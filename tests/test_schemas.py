"""El contrato de Pydantic: qué acepta y qué rechaza."""

import pytest
from pydantic import ValidationError

from schemas import ExtraccionTecnica, NivelDeCriticidad

VALIDO = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL.",
}


def test_objeto_valido():
    obj = ExtraccionTecnica.model_validate(VALIDO)
    assert obj.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert obj.nivel_de_criticidad is NivelDeCriticidad.ALTA
    assert obj.model_dump(mode="json")["nivel_de_criticidad"] == "alta"


def test_limpia_espacios_vacios_y_duplicados():
    obj = ExtraccionTecnica.model_validate(
        {**VALIDO, "tecnologias": [" Redis ", "redis", "", "  ", "Nginx", "REDIS"]}
    )
    assert obj.tecnologias == ["Redis", "Nginx"]


def test_criticidad_en_mayusculas_se_normaliza():
    obj = ExtraccionTecnica.model_validate({**VALIDO, "nivel_de_criticidad": " Media "})
    assert obj.nivel_de_criticidad is NivelDeCriticidad.MEDIA


@pytest.mark.parametrize(
    "cambio",
    [
        {"tecnologias": []},  # lista vacía
        {"tecnologias": ["", "   "]},  # vacía después de limpiar
        {"tecnologias": "Redis"},  # no es lista
        {"tecnologias": ["Redis", 3]},  # elemento que no es texto
        {"tecnologias": [f"t{i}" for i in range(21)]},  # más de 20
        {"nivel_de_criticidad": "urgente"},  # fuera del enum
        {"resumen_tecnico": "   "},  # vacío
        {"resumen_tecnico": "corto"},  # menos de 10 caracteres
        {"resumen_tecnico": "x" * 601},  # demasiado largo
        {"comentario": "campo extra"},  # campo no pedido
    ],
)
def test_rechaza_datos_invalidos(cambio):
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**VALIDO, **cambio})


@pytest.mark.parametrize("campo", ["tecnologias", "nivel_de_criticidad", "resumen_tecnico"])
def test_campos_obligatorios(campo):
    datos = dict(VALIDO)
    del datos[campo]
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate(datos)
