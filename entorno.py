"""Lectura de configuración desde variables de entorno (misma lógica que el Módulo 1).

Las claves NO viven en el código: salen solo de OPENAI_API_KEY o
ANTHROPIC_API_KEY, que load_dotenv() carga desde el archivo .env.
Un placeholder del ejemplo tampoco cuenta como clave, así no se manda
"tu_clave_aca" a un servicio pago.
"""

import os

PROVEEDORES_VALIDOS = {"openai", "anthropic", "demo"}

# Modelos por defecto. NO están verificados contra una cuenta real.
MODELO_POR_DEFECTO = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}

VARIABLE_DE_CLAVE = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

VARIABLE_DE_MODELO = {
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
}

_PLACEHOLDERS = {
    "tu_clave",
    "tu_clave_aca",
    "tu_clave_aqui",
    "changeme",
    "sk-...",
    "pega_tu_clave",
    "pegá_tu_clave",
}


def clave_util(valor: str | None) -> bool:
    """True si el texto parece una clave cargada de verdad."""

    if valor is None:
        return False
    limpio = valor.strip()
    if not limpio:
        return False
    return limpio.lower() not in _PLACEHOLDERS


def proveedor_elegido() -> str:
    """Lee LLM_PROVIDER. Por defecto openai, como en el Módulo 1."""

    proveedor = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if proveedor not in PROVEEDORES_VALIDOS:
        raise ValueError(
            f"LLM_PROVIDER='{proveedor}' no sirve. Opciones: openai, anthropic o demo."
        )
    return proveedor


def hay_clave_para(proveedor: str) -> bool:
    variable = VARIABLE_DE_CLAVE.get(proveedor)
    return variable is not None and clave_util(os.getenv(variable))


def leer_clave(proveedor: str) -> str:
    """Devuelve la clave del proveedor leída del entorno. Nunca la imprime."""

    variable = VARIABLE_DE_CLAVE[proveedor]
    valor = os.getenv(variable)
    if not clave_util(valor):
        raise ValueError(f"Falta {variable} en el entorno (o es el texto de ejemplo).")
    return valor.strip()


def leer_modelo(proveedor: str) -> str:
    variable = VARIABLE_DE_MODELO[proveedor]
    return (os.getenv(variable) or MODELO_POR_DEFECTO[proveedor]).strip()
