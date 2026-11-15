"""Pipeline de Extracción de Entidades Técnicas.

Recorrido de un texto:

    texto ─► prompt (ChatPromptTemplate)
          ─► modelo.with_structured_output(ExtraccionTecnica, include_raw=True)
          ─► verificar_respuesta (mira finish_reason y errores de parseo)
          ─► ExtraccionTecnica validada

Todo eso va envuelto en .with_retry(): si verificar_respuesta lanza un
ErrorReintentable (JSON roto, incompleto o cortado por tokens), LangChain
vuelve a correr la cadena entera, hasta INTENTOS_MAXIMOS veces.

La función pública es process_text(text), asíncrona, que usa .ainvoke().
"""

import logging
import sys
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.exceptions import ModelRefusalError, OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

import entorno
from schemas import ExtraccionTecnica

logger = logging.getLogger("pipeline")

# --- Números que la consigna no fijó (pide "al menos un reintento") ---------
INTENTOS_MAXIMOS = 3  # 1 intento normal + 2 reintentos.
MAX_TOKENS_RESPUESTA = 512  # Tope de tokens de salida del modelo.
TIMEOUT_SEGUNDOS = 30.0
# Espera entre intentos: ~1 s, después ~2 s (tope 4 s), más hasta 0,5 s al azar.
# No se pasa "initial" porque la versión actual de tenacity lo marca como obsoleto.
ESPERA_ENTRE_INTENTOS = {"max": 4.0, "exp_base": 2.0, "jitter": 0.5}

# Nombre de la cadena completa: lo usa RegistroDeIntentos para saber qué loguear.
NOMBRE_PIPELINE = "pipeline_extraccion"

# Valores de corte por tokens según el proveedor:
# OpenAI pone finish_reason="length"; Anthropic pone stop_reason="max_tokens".
_CORTES_POR_TOKENS = {"length", "max_tokens"}


# --- Excepciones propias -----------------------------------------------------
class ErrorDelPipeline(Exception):
    """Base de los errores de este proyecto."""


class ErrorReintentable(ErrorDelPipeline):
    """Algo salió mal en la respuesta, pero otro intento puede salir bien."""


class RespuestaIncompleta(ErrorReintentable):
    """El modelo cortó por límite de tokens: el JSON puede estar a medias."""


class RespuestaMalFormada(ErrorReintentable):
    """El JSON no se pudo leer o no cumple el esquema de Pydantic."""


class RespuestaRechazada(ErrorDelPipeline):
    """El modelo se negó a responder. NO se reintenta: el mismo pedido daría lo mismo."""


class ExtraccionFallida(ErrorDelPipeline):
    """Se agotaron los intentos. Es el error que ve quien llama a process_text."""


# Lo que with_retry considera reintentable. ValidationError y
# OutputParserException se agregan por las dudas: normalmente ya llegan
# convertidos en RespuestaMalFormada por verificar_respuesta.
ERRORES_REINTENTABLES: tuple[type[BaseException], ...] = (
    ErrorReintentable,
    ValidationError,
    OutputParserException,
)


# --- Prompt: plantilla modular, sin f-strings --------------------------------
# Las llaves {instrucciones_formato} y {texto} son variables de LangChain.
# Python no las rellena: lo hace ChatPromptTemplate al invocar la cadena.
MENSAJE_SISTEMA = (
    "Sos un analista técnico de software. Leés un párrafo (un log de error "
    "o una descripción de arquitectura) y extraés entidades técnicas.\n\n"
    "{instrucciones_formato}"
)

MENSAJE_USUARIO = "Texto a analizar:\n<texto>\n{texto}\n</texto>"

INSTRUCCIONES_FORMATO = (
    "Reglas de formato:\n"
    "1. tecnologias: solo nombres de tecnologías que aparecen en el texto, "
    "sin repetir. No inventes ninguna.\n"
    "2. nivel_de_criticidad: alta si hay caída, pérdida de datos o riesgo de "
    "seguridad; media si hay degradación o errores intermitentes; baja si es "
    "informativo o una mejora.\n"
    "3. resumen_tecnico: una o dos oraciones, en español, sin adornos.\n"
    "4. Si el texto es ambiguo, elegí la opción más prudente y decilo en el resumen."
)


def construir_prompt(instrucciones: str = INSTRUCCIONES_FORMATO) -> ChatPromptTemplate:
    """Arma la plantilla y deja fijas las instrucciones de formato con .partial().

    Así la cadena solo necesita recibir {"texto": ...}. Si querés otras
    instrucciones, pasalas por parámetro y no tocás el resto.
    """

    plantilla = ChatPromptTemplate.from_messages(
        [("system", MENSAJE_SISTEMA), ("human", MENSAJE_USUARIO)]
    )
    return plantilla.partial(instrucciones_formato=instrucciones)


# --- Modelo: OpenAI o Anthropic según LLM_PROVIDER (como en el Módulo 1) -----
def crear_modelo(proveedor: str | None = None) -> BaseChatModel:
    """Crea ChatOpenAI o ChatAnthropic. La clave sale SOLO del entorno."""

    proveedor = proveedor or entorno.proveedor_elegido()
    if proveedor == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=entorno.leer_modelo("openai"),
            api_key=entorno.leer_clave("openai"),
            temperature=0,
            max_tokens=MAX_TOKENS_RESPUESTA,
            timeout=TIMEOUT_SEGUNDOS,
        )
    if proveedor == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Igual que en el Módulo 1: a Anthropic no le mandamos temperature.
        return ChatAnthropic(
            model=entorno.leer_modelo("anthropic"),
            api_key=entorno.leer_clave("anthropic"),
            max_tokens=MAX_TOKENS_RESPUESTA,
            timeout=TIMEOUT_SEGUNDOS,
        )
    raise ValueError(f"No sé crear un modelo real para '{proveedor}'.")


# --- Control de la respuesta: finish_reason + errores de parseo --------------
def motivo_de_corte(mensaje: BaseMessage | None) -> str | None:
    """Devuelve "length" o "max_tokens" si el modelo cortó por tokens; si no, None."""

    if mensaje is None:
        return None
    metadata = getattr(mensaje, "response_metadata", None) or {}
    for clave in ("finish_reason", "stop_reason"):
        valor = metadata.get(clave)
        if valor in _CORTES_POR_TOKENS:
            return valor
    return None


def verificar_respuesta(salida: dict[str, Any]) -> ExtraccionTecnica:
    """Revisa lo que devolvió with_structured_output(..., include_raw=True).

    `salida` trae tres claves:
    - raw: el AIMessage crudo, con response_metadata (finish_reason, etc.).
    - parsed: el objeto ExtraccionTecnica si se pudo armar, o None.
    - parsing_error: la excepción si el JSON no se pudo leer o validar, o None.

    El orden importa: primero se mira si la respuesta se cortó por tokens.
    Si se cortó, no se intenta "transformar" nada: se pide de nuevo.
    """

    raw = salida.get("raw")
    metadata = getattr(raw, "response_metadata", None) or {}
    error = salida.get("parsing_error")

    # 1) ¿Se cortó por tokens? Va primero: un JSON a medias no se transforma.
    corte = motivo_de_corte(raw)
    if corte:
        logger.warning("Respuesta cortada por tokens (motivo: %s). Se descarta.", corte)
        raise RespuestaIncompleta(
            f"El modelo cortó la respuesta por límite de tokens ({corte})."
        )

    # 2) ¿El modelo se negó? No tiene sentido reintentar.
    if isinstance(error, ModelRefusalError) or metadata.get("stop_reason") == "refusal":
        logger.error("El modelo se negó a responder. No se reintenta.")
        raise RespuestaRechazada("El modelo se negó a procesar el texto.") from error

    # 3) ¿El JSON vino roto o no cumple el esquema? Se reintenta.
    if error is not None:
        logger.warning(
            "Respuesta mal formada: %s: %s", type(error).__name__, _resumir(error)
        )
        raise RespuestaMalFormada(f"No pasó la validación: {_resumir(error)}") from error

    # 4) ¿Contestó texto suelto en vez del objeto? Se reintenta.
    resultado = salida.get("parsed")
    if resultado is None:
        logger.warning("El modelo no devolvió ningún objeto estructurado.")
        raise RespuestaMalFormada("El modelo no devolvió el objeto pedido.")

    # Doble chequeo barato: re-validar asegura que es exactamente nuestro esquema.
    resultado = ExtraccionTecnica.model_validate(resultado.model_dump())
    logger.info(
        "Validación OK: %d tecnologías, criticidad %s.",
        len(resultado.tecnologias),
        resultado.nivel_de_criticidad.value,
    )
    return resultado


def _resumir(error: BaseException, largo: int = 200) -> str:
    texto = " ".join(str(error).split())
    return texto if len(texto) <= largo else texto[: largo - 3] + "..."


# --- La cadena LCEL ----------------------------------------------------------
def construir_cadena(
    modelo: BaseChatModel,
    *,
    intentos: int = INTENTOS_MAXIMOS,
    esperar: bool = True,
) -> Runnable:
    """prompt | modelo.with_structured_output(Schema) | verificación, con reintentos.

    method="function_calling" se usa en los dos proveedores para que el
    finish_reason llegue siempre al paso de verificación (ver README).
    """

    if intentos < 2:
        raise ValueError("La consigna pide al menos un reintento: intentos >= 2.")

    estructurado = modelo.with_structured_output(
        ExtraccionTecnica, include_raw=True, method="function_calling"
    )
    # La cadena LCEL que pide la consigna, más el paso de verificación.
    cadena = construir_prompt() | estructurado | RunnableLambda(verificar_respuesta)

    return cadena.with_retry(
        retry_if_exception_type=ERRORES_REINTENTABLES,
        stop_after_attempt=intentos,
        wait_exponential_jitter=esperar,
        exponential_jitter_params=ESPERA_ENTRE_INTENTOS if esperar else None,
    ).with_config(run_name=NOMBRE_PIPELINE)


class RegistroDeIntentos(AsyncCallbackHandler):
    """Loguea cada intento.

    LangChain avisa a este "callback" cada vez que arranca un paso de la cadena.
    El primer intento es el arranque del pipeline completo; los reintentos
    llegan con la etiqueta "retry:attempt:N" que pone with_retry.
    """

    async def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("name") == NOMBRE_PIPELINE:
            logger.info("Intento 1: mando el texto al modelo.")
            return
        for tag in tags or []:
            if tag.startswith("retry:attempt:"):
                numero = int(tag.rsplit(":", 1)[1])
                logger.info("Reintento automático: intento %d.", numero)
        # Los demás pasos internos (prompt, modelo, parser) no se loguean.


# --- Función pública ---------------------------------------------------------
_cadena_por_defecto: Runnable | None = None


async def process_text(text: str, cadena: Runnable | None = None) -> ExtraccionTecnica:
    """Extrae entidades técnicas de `text` y devuelve un objeto validado.

    `cadena` es opcional: los tests y el demo pasan una con modelo falso.
    Si no se pasa, se arma una con el proveedor de LLM_PROVIDER.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("process_text necesita un texto no vacío.")

    if cadena is None:
        global _cadena_por_defecto
        if _cadena_por_defecto is None:
            _cadena_por_defecto = construir_cadena(crear_modelo())
        cadena = _cadena_por_defecto

    logger.info("Procesando texto de %d caracteres.", len(text))
    try:
        resultado = await cadena.ainvoke(
            {"texto": text.strip()}, config={"callbacks": [RegistroDeIntentos()]}
        )
    except ERRORES_REINTENTABLES as exc:
        logger.error("Se agotaron los intentos. Último error: %s", exc)
        raise ExtraccionFallida(
            f"No se obtuvo un objeto válido después de los reintentos: {exc}"
        ) from exc
    logger.info("Listo: objeto validado.")
    return resultado


def configurar_logs(nivel: int = logging.INFO) -> None:
    """Formato simple para ver la validación y los reintentos en la consola."""

    logging.basicConfig(
        stream=sys.stdout,  # Mismo canal que los print, así no se mezclan.
        level=nivel,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Las bibliotecas HTTP hablan mucho en INFO; las bajamos a WARNING.
    for ruidoso in ("httpx", "httpcore", "openai", "anthropic"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
