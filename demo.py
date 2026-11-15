"""Mini-script de prueba asíncrono.

Uso:  python demo.py

- Sin clave (o con LLM_PROVIDER=demo): MODO DEMOSTRACIÓN con un modelo falso.
  No viaja nada a internet y no se gasta crédito. Muestra cinco casos:
  ejemplo normal, JSON roto que se recupera, respuesta cortada por tokens que
  se recupera, prueba de estrés con texto ambiguo y un caso que agota los intentos.
- Con clave y LLM_PROVIDER=openai|anthropic: corre el ejemplo y la prueba de
  estrés contra el modelo real (son dos llamadas, como mínimo).
"""

import asyncio
import json
import logging

from dotenv import load_dotenv

import entorno
from chain import ExtraccionFallida, configurar_logs, construir_cadena, crear_modelo, process_text
from modelo_falso import (
    ModeloFalso,
    respuesta_cortada,
    respuesta_json_roto,
    respuesta_ok,
    respuesta_sin_objeto,
)

logger = logging.getLogger("demo")

TEXTO_EJEMPLO = (
    "ERROR 2026-11-14 18:03:12 api-pagos: FastAPI devuelve 503 en /checkout. "
    "El pool de PostgreSQL llegó a 100 conexiones y Redis tarda 2 s en responder "
    "el caché de sesiones. El 40% de los pagos están fallando."
)

TEXTO_AMBIGUO = (
    "Anoche algo anduvo lento. Puede ser la base, o el redis, o capaz el nginx "
    "de adelante; algunos usuarios dicen que no pasa nada. Hay que mirarlo."
)

TEXTO_SIN_TECNOLOGIAS = "El sistema hizo cosas raras ayer y nadie sabe bien por qué."

# Lo que "contesta" el modelo falso en cada caso, en orden de llamada.
RESPUESTA_EJEMPLO = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": (
        "API de pagos en FastAPI con caché en Redis y persistencia en PostgreSQL; "
        "el pool de conexiones saturado provoca errores 503 en el checkout."
    ),
}

CASOS_DEMO = [
    ("1. Ejemplo de la consigna", TEXTO_EJEMPLO, [respuesta_ok(RESPUESTA_EJEMPLO)]),
    (
        "2. JSON mal formado en el primer intento",
        TEXTO_EJEMPLO,
        [respuesta_json_roto('{"tecnologias": ["FastAPI", "Redis"'), respuesta_ok(RESPUESTA_EJEMPLO)],
    ),
    (
        "3. Respuesta cortada por tokens (finish_reason=length)",
        TEXTO_EJEMPLO,
        [respuesta_cortada("openai"), respuesta_ok(RESPUESTA_EJEMPLO)],
    ),
    (
        "4. Prueba de estrés: texto ambiguo",
        TEXTO_AMBIGUO,
        [
            # Primer intento: lista vacía y criticidad inventada -> Pydantic rechaza.
            respuesta_ok({"tecnologias": [], "nivel_de_criticidad": "urgente", "resumen_tecnico": "No sé."}),
            # Segundo intento: duplicados y mayúsculas -> el esquema los normaliza.
            respuesta_ok(
                {
                    "tecnologias": ["Redis", " redis ", "Nginx", ""],
                    "nivel_de_criticidad": "Media",
                    "resumen_tecnico": (
                        "Lentitud intermitente sin causa confirmada; sospechas sobre la base "
                        "de datos, Redis o el proxy Nginx. Criticidad media por prudencia."
                    ),
                }
            ),
        ],
    ),
    (
        "5. Prueba de estrés: texto sin tecnologías (agota los intentos)",
        TEXTO_SIN_TECNOLOGIAS,
        [respuesta_sin_objeto(), respuesta_json_roto("{"), respuesta_cortada("anthropic")],
    ),
]


def _mostrar(resultado) -> None:
    print(json.dumps(resultado.model_dump(mode="json"), ensure_ascii=False, indent=2))


async def _correr(titulo: str, texto: str, cadena=None) -> None:
    print("\n" + "=" * 70)
    print(titulo)
    print("Texto:", texto)
    print("-" * 70, flush=True)
    try:
        resultado = await process_text(texto, cadena=cadena)
    except ExtraccionFallida as exc:
        print(f"Sin resultado válido: {exc}")
        return
    _mostrar(resultado)


async def modo_demostracion() -> None:
    print("=" * 70)
    print("MODO DEMOSTRACIÓN: modelo falso, sin red y sin claves.")
    print("Para usar un modelo real: LLM_PROVIDER=openai|anthropic y su clave en .env.")
    print("=" * 70)
    for titulo, texto, respuestas in CASOS_DEMO:
        modelo = ModeloFalso(respuestas=respuestas)
        # esperar=False: sin pausa entre reintentos, para que el demo no tarde.
        # Con un modelo real se usa la espera exponencial de chain.py.
        await _correr(titulo, texto, construir_cadena(modelo, esperar=False))
        veces = "vez" if modelo.llamadas == 1 else "veces"
        print(f"(El modelo falso fue llamado {modelo.llamadas} {veces}.)")


async def modo_real(proveedor: str) -> None:
    print(f"Proveedor real: {proveedor}, modelo {entorno.leer_modelo(proveedor)}.")
    cadena = construir_cadena(crear_modelo(proveedor))
    await _correr("1. Ejemplo de la consigna", TEXTO_EJEMPLO, cadena)
    await _correr("2. Prueba de estrés: texto ambiguo", TEXTO_AMBIGUO, cadena)


async def main() -> None:
    load_dotenv()  # Carga .env (si existe) antes de leer las variables.
    configurar_logs()
    proveedor = entorno.proveedor_elegido()
    if proveedor == "demo" or not entorno.hay_clave_para(proveedor):
        await modo_demostracion()
    else:
        await modo_real(proveedor)


if __name__ == "__main__":
    asyncio.run(main())
