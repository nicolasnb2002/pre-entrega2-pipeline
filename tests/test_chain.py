"""La cadena LCEL con dobles: sin claves, sin red."""

import logging

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

import chain
from chain import (
    ExtraccionFallida,
    RespuestaIncompleta,
    RespuestaMalFormada,
    RespuestaRechazada,
    construir_cadena,
    construir_prompt,
    motivo_de_corte,
    process_text,
    verificar_respuesta,
)
from modelo_falso import (
    ModeloFalso,
    respuesta_cortada,
    respuesta_json_roto,
    respuesta_ok,
    respuesta_negativa,
    respuesta_sin_objeto,
)
from schemas import ExtraccionTecnica

DATOS = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL.",
}
TEXTO = "FastAPI tira 503: el pool de PostgreSQL está lleno y Redis no responde."


def _cadena(*respuestas, intentos=3):
    modelo = ModeloFalso(respuestas=list(respuestas))
    return modelo, construir_cadena(modelo, intentos=intentos, esperar=False)


# --- Prompt --------------------------------------------------------------------
def test_prompt_es_plantilla_con_variables():
    prompt = construir_prompt()
    assert isinstance(prompt, ChatPromptTemplate)
    # Las instrucciones quedan fijas con partial: solo falta el texto.
    assert prompt.input_variables == ["texto"]
    mensajes = prompt.format_messages(texto="hola mundo")
    assert "Reglas de formato" in mensajes[0].content
    assert "hola mundo" in mensajes[1].content


def test_prompt_acepta_otras_instrucciones():
    mensajes = construir_prompt("Respondé corto.").format_messages(texto="x")
    assert "Respondé corto." in mensajes[0].content


# --- process_text: camino feliz ------------------------------------------------
async def test_process_text_devuelve_objeto_validado():
    modelo, cadena = _cadena(respuesta_ok(DATOS))
    resultado = await process_text(TEXTO, cadena=cadena)
    assert isinstance(resultado, ExtraccionTecnica)
    assert resultado.model_dump(mode="json") == DATOS
    assert modelo.llamadas == 1


async def test_process_text_rechaza_texto_vacio():
    modelo, cadena = _cadena(respuesta_ok(DATOS))
    with pytest.raises(ValueError):
        await process_text("   ", cadena=cadena)
    assert modelo.llamadas == 0  # No se gastó ninguna llamada.


# --- Reintentos ----------------------------------------------------------------
async def test_reintenta_tras_json_mal_formado(caplog):
    caplog.set_level(logging.INFO, logger="pipeline")
    modelo, cadena = _cadena(respuesta_json_roto('{"tecnologias": ["Redis"'), respuesta_ok(DATOS))
    resultado = await process_text(TEXTO, cadena=cadena)
    assert resultado.tecnologias == DATOS["tecnologias"]
    assert modelo.llamadas == 2
    assert "Respuesta mal formada" in caplog.text
    assert "Reintento automático: intento 2" in caplog.text


async def test_reintenta_tras_validacion_fallida():
    invalido = {**DATOS, "tecnologias": []}
    modelo, cadena = _cadena(respuesta_ok(invalido), respuesta_ok(DATOS))
    await process_text(TEXTO, cadena=cadena)
    assert modelo.llamadas == 2


async def test_reintenta_si_no_hay_objeto():
    modelo, cadena = _cadena(respuesta_sin_objeto(), respuesta_ok(DATOS))
    await process_text(TEXTO, cadena=cadena)
    assert modelo.llamadas == 2


async def test_agota_intentos_y_lanza_extraccion_fallida(caplog):
    caplog.set_level(logging.INFO, logger="pipeline")
    modelo, cadena = _cadena(
        respuesta_json_roto("{"), respuesta_json_roto("{"), respuesta_json_roto("{"), respuesta_ok(DATOS)
    )
    with pytest.raises(ExtraccionFallida):
        await process_text(TEXTO, cadena=cadena)
    assert modelo.llamadas == 3  # Paró en INTENTOS_MAXIMOS; no usó la cuarta.
    assert "Se agotaron los intentos" in caplog.text


async def test_error_no_reintentable_no_se_reintenta():
    class ModeloQueRompe(ModeloFalso):
        def _generate(self, *args, **kwargs):
            self._llamadas += 1
            raise KeyError("bug del programa, no de la respuesta")

    modelo = ModeloQueRompe(respuestas=[])
    cadena = construir_cadena(modelo, esperar=False)
    with pytest.raises(KeyError):
        await process_text(TEXTO, cadena=cadena)
    assert modelo.llamadas == 1


def test_exige_al_menos_un_reintento():
    with pytest.raises(ValueError):
        construir_cadena(ModeloFalso(respuestas=[]), intentos=1)


# --- finish_reason -------------------------------------------------------------
@pytest.mark.parametrize("proveedor, esperado", [("openai", "length"), ("anthropic", "max_tokens")])
def test_motivo_de_corte_detecta_tokens(proveedor, esperado):
    assert motivo_de_corte(respuesta_cortada(proveedor)) == esperado


@pytest.mark.parametrize("metadata", [{"finish_reason": "stop"}, {"finish_reason": "tool_calls"}, {"stop_reason": "tool_use"}, {}])
def test_motivo_de_corte_normal(metadata):
    assert motivo_de_corte(AIMessage(content="", response_metadata=metadata)) is None


def test_corte_por_tokens_gana_aunque_el_json_parezca_bien():
    # Aunque parsed tenga un objeto, si se cortó por tokens no se usa.
    mensaje = respuesta_ok(DATOS, finish_reason="length")
    salida = {"raw": mensaje, "parsed": ExtraccionTecnica(**DATOS), "parsing_error": None}
    with pytest.raises(RespuestaIncompleta):
        verificar_respuesta(salida)


def test_parsing_error_se_convierte_en_mal_formada():
    salida = {"raw": AIMessage(content=""), "parsed": None, "parsing_error": ValueError("roto")}
    with pytest.raises(RespuestaMalFormada):
        verificar_respuesta(salida)


@pytest.mark.parametrize("proveedor", ["openai", "anthropic"])
async def test_reintenta_tras_respuesta_cortada(proveedor, caplog):
    caplog.set_level(logging.INFO, logger="pipeline")
    modelo, cadena = _cadena(respuesta_cortada(proveedor), respuesta_ok(DATOS))
    resultado = await process_text(TEXTO, cadena=cadena)
    assert resultado.nivel_de_criticidad.value == "alta"
    assert modelo.llamadas == 2
    assert "cortada por tokens" in caplog.text


async def test_cortada_siempre_termina_en_fallo():
    modelo, cadena = _cadena(respuesta_cortada(), respuesta_cortada(), respuesta_cortada())
    with pytest.raises(ExtraccionFallida) as info:
        await process_text(TEXTO, cadena=cadena)
    assert isinstance(info.value.__cause__, RespuestaIncompleta)


# --- Modelos reales: solo se construyen, no se llaman ---------------------------
@pytest.mark.parametrize("proveedor, variable", [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")])
def test_crear_modelo_real_sin_llamar(monkeypatch, proveedor, variable):
    monkeypatch.setenv(variable, "clave-falsa-de-test")
    modelo = chain.crear_modelo(proveedor)
    assert modelo.max_tokens == chain.MAX_TOKENS_RESPUESTA
    # Construir la cadena tampoco hace ninguna llamada de red.
    construir_cadena(modelo)


@pytest.mark.parametrize("valor", [None, "", "tu_clave_aca"])
def test_crear_modelo_sin_clave_falla(monkeypatch, valor):
    if valor is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", valor)
    with pytest.raises(ValueError):
        chain.crear_modelo("openai")


async def test_process_text_sin_cadena_usa_crear_modelo(monkeypatch):
    # Sin pasar cadena, process_text arma la suya con crear_modelo().
    modelo = ModeloFalso(respuestas=[respuesta_ok(DATOS)])
    monkeypatch.setattr(chain, "crear_modelo", lambda: modelo)
    monkeypatch.setattr(chain, "_cadena_por_defecto", None)
    resultado = await process_text(TEXTO)
    assert resultado.tecnologias == DATOS["tecnologias"]
    assert modelo.llamadas == 1


async def test_negativa_del_modelo_no_se_reintenta():
    modelo, cadena = _cadena(respuesta_negativa(), respuesta_ok(DATOS))
    with pytest.raises(RespuestaRechazada):
        await process_text(TEXTO, cadena=cadena)
    assert modelo.llamadas == 1


def test_stop_reason_refusal_de_anthropic():
    mensaje = AIMessage(content="", response_metadata={"stop_reason": "refusal"})
    with pytest.raises(RespuestaRechazada):
        verificar_respuesta({"raw": mensaje, "parsed": None, "parsing_error": None})
