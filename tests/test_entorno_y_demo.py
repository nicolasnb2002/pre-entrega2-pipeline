"""Configuración por variables de entorno y el mini-script en modo demostración."""

import pytest

import demo
import entorno


def test_proveedor_por_defecto(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert entorno.proveedor_elegido() == "openai"


def test_proveedor_invalido(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError):
        entorno.proveedor_elegido()


def test_modelos_por_defecto(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert entorno.leer_modelo("openai") == "gpt-4o-mini"
    assert entorno.leer_modelo("anthropic") == "claude-haiku-5-5"


@pytest.mark.parametrize("valor, esperado", [(None, False), ("", False), ("tu_clave_aca", False), ("sk-algo", True)])
def test_clave_util(valor, esperado):
    assert entorno.clave_util(valor) is esperado


async def test_demo_sin_clave_corre_los_cinco_casos(monkeypatch, capsys):
    # Sin clave: entra solo en modo demostración, sin red.
    monkeypatch.setattr(demo, "load_dotenv", lambda: None)  # Ignora un .env local.
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    await demo.main()
    salida = capsys.readouterr().out
    assert "MODO DEMOSTRACIÓN" in salida
    assert salida.count('"nivel_de_criticidad"') == 4  # Casos 1 a 4 dan objeto.
    assert "Sin resultado válido" in salida  # El caso 5 agota los intentos.
