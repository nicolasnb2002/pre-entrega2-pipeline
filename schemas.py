"""El contrato de salida del pipeline.

Acá se define QUÉ tiene que devolver el modelo. Si la respuesta no cumple
este contrato, Pydantic lanza ValidationError y el pipeline la trata como
"respuesta mal formada" (se puede reintentar).

Campos que pide la consigna:
- tecnologias: lista de strings, no vacía.
- nivel_de_criticidad: enum con baja, media o alta.
- resumen_tecnico: string.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Límites que la consigna no fijó. Están acá para cambiarlos en un solo lugar.
MAX_TECNOLOGIAS = 20
MIN_LARGO_RESUMEN = 10
MAX_LARGO_RESUMEN = 600


class NivelDeCriticidad(str, Enum):
    """Tres niveles y nada más. Heredar de str hace que el JSON salga como "alta"."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un párrafo (log de error o arquitectura).

    El docstring y las descripciones de cada campo viajan al LLM como parte
    del esquema: son las "instrucciones" que ve el modelo sobre cada campo.
    """

    # str_strip_whitespace saca espacios de los costados en todos los str.
    # extra="forbid" rechaza campos que no pedimos (ej. un "comentario" inventado).
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    tecnologias: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_TECNOLOGIAS,
        description=(
            "Tecnologías concretas nombradas en el texto: lenguajes, frameworks, "
            "bases de datos, colas, servicios cloud. Sin repetir."
        ),
    )
    nivel_de_criticidad: NivelDeCriticidad = Field(
        ...,
        description="baja, media o alta, según el impacto del problema descripto.",
    )
    resumen_tecnico: str = Field(
        ...,
        min_length=MIN_LARGO_RESUMEN,
        max_length=MAX_LARGO_RESUMEN,
        description="Una o dos oraciones con lo técnico importante del texto.",
    )

    @field_validator("tecnologias", mode="before")
    @classmethod
    def limpiar_tecnologias(cls, valor: object) -> object:
        """Saca espacios, vacíos y duplicados, respetando el orden original.

        "Redis" y " redis " cuentan como la misma: queda la primera que apareció.
        Si después de limpiar la lista queda vacía, min_length=1 la rechaza.
        """

        if not isinstance(valor, list):
            return valor  # Que Pydantic diga "esto no es una lista".
        vistas: set[str] = set()
        limpias: list[object] = []
        for item in valor:
            if not isinstance(item, str):
                limpias.append(item)  # Pydantic lo rechaza después como no-string.
                continue
            nombre = item.strip()
            if not nombre or nombre.casefold() in vistas:
                continue
            vistas.add(nombre.casefold())
            limpias.append(nombre)
        return limpias

    @field_validator("nivel_de_criticidad", mode="before")
    @classmethod
    def criticidad_en_minusculas(cls, valor: object) -> object:
        """Acepta "Alta" o " ALTA " y lo pasa a "alta". Cualquier otra palabra falla."""

        if isinstance(valor, str):
            return valor.strip().lower()
        return valor
