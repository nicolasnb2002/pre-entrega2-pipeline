"""OPENAI_BASE_URL y LLM_MAX_TOKENS: proveedores compatibles con OpenAI (Gemini, Groq).

Solo se construye ChatOpenAI con una clave falsa. No hay ninguna llamada de red.
"""

import pytest

import chain
import entorno

URL_OFICIAL_OPENAI = "https://api.openai.com/v1"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ = "https://api.groq.com/openai/v1"


@pytest.fixture
def entorno_limpio(monkeypatch):
    for variable in ("OPENAI_BASE_URL", "OPENAI_MODEL", "LLM_MAX_TOKENS", "OPENAI_API_BASE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "clave-falsa-de-test")
    return monkeypatch


def _parametros_pasados(monkeypatch):
    """Reemplaza ChatOpenAI por un espía que guarda con qué se lo llamó."""

    import langchain_openai

    recibidos = {}

    class Espia:
        def __init__(self, **kwargs):
            recibidos.update(kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", Espia)
    return recibidos


def test_sin_base_url_no_se_pasa(entorno_limpio):
    recibidos = _parametros_pasados(entorno_limpio)
    chain.crear_modelo("openai")
    assert "base_url" not in recibidos
    assert recibidos["api_key"] == "clave-falsa-de-test"


def test_base_url_vacia_no_se_pasa(entorno_limpio):
    entorno_limpio.setenv("OPENAI_BASE_URL", "   ")
    recibidos = _parametros_pasados(entorno_limpio)
    chain.crear_modelo("openai")
    assert "base_url" not in recibidos


@pytest.mark.parametrize("url, modelo", [(GEMINI, "gemini-3.5-flash-lite"), (GROQ, "openai/gpt-oss-20b")])
def test_con_base_url_se_pasa(entorno_limpio, url, modelo):
    entorno_limpio.setenv("OPENAI_BASE_URL", url)
    entorno_limpio.setenv("OPENAI_MODEL", modelo)
    recibidos = _parametros_pasados(entorno_limpio)
    chain.crear_modelo("openai")
    assert recibidos["base_url"] == url
    assert recibidos["model"] == modelo
    # La clave sigue saliendo de OPENAI_API_KEY.
    assert recibidos["api_key"] == "clave-falsa-de-test"


def test_chatopenai_real_apunta_a_la_url(entorno_limpio):
    # Con el ChatOpenAI de verdad (sin llamarlo), el cliente queda apuntando a Groq.
    entorno_limpio.setenv("OPENAI_BASE_URL", GROQ)
    modelo = chain.crear_modelo("openai")
    assert str(modelo.client._client.base_url).rstrip("/") == GROQ
    chain.construir_cadena(modelo)  # Armar la cadena tampoco llama a nadie.


def test_chatopenai_real_sin_url_usa_openai(entorno_limpio):
    modelo = chain.crear_modelo("openai")
    assert str(modelo.client._client.base_url).rstrip("/") == URL_OFICIAL_OPENAI


def test_base_url_invalida(entorno_limpio):
    entorno_limpio.setenv("OPENAI_BASE_URL", "api.groq.com/openai/v1")
    with pytest.raises(ValueError):
        chain.crear_modelo("openai")


def test_anthropic_ignora_base_url(entorno_limpio):
    entorno_limpio.setenv("OPENAI_BASE_URL", GROQ)
    entorno_limpio.setenv("ANTHROPIC_API_KEY", "clave-falsa-de-test")
    modelo = chain.crear_modelo("anthropic")
    assert "groq" not in str(getattr(modelo, "anthropic_api_url", "")).lower()


def test_max_tokens_por_defecto_y_desde_env(entorno_limpio):
    assert entorno.leer_max_tokens(512) == 512
    entorno_limpio.setenv("LLM_MAX_TOKENS", "2048")
    assert chain.crear_modelo("openai").max_tokens == 2048


@pytest.mark.parametrize("valor", ["mucho", "0", "-5"])
def test_max_tokens_invalido(entorno_limpio, valor):
    entorno_limpio.setenv("LLM_MAX_TOKENS", valor)
    with pytest.raises(ValueError):
        entorno.leer_max_tokens(512)


@pytest.mark.parametrize("valor", ["pegá_acá_la_clave_de_gemini", "pegá_acá_la_clave_de_groq", "tu_clave_de_gemini", "tu_clave_de_groq"])
def test_placeholders_del_ejemplo_no_cuentan_como_clave(valor):
    assert entorno.clave_util(valor) is False
