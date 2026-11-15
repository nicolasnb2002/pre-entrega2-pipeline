"""Modelo de chat falso para el modo demostración y los tests.

No usa red ni claves. Devuelve, en orden, las respuestas que le pasás.
Imita lo justo de un modelo real:
- tool_calls con los argumentos (como hace function_calling).
- response_metadata con finish_reason, para probar el corte por tokens.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

NOMBRE_HERRAMIENTA = "ExtraccionTecnica"  # El nombre que LangChain le da al esquema.


class ModeloFalso(BaseChatModel):
    """Chat model que responde una lista fija de AIMessage, uno por llamada."""

    respuestas: list[AIMessage]
    _llamadas: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "modelo-falso"

    @property
    def llamadas(self) -> int:
        """Cuántas veces se lo invocó. Sirve para comprobar los reintentos."""

        return self._llamadas

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ModeloFalso":
        # Un modelo real "ata" el esquema como herramienta. El falso ya sabe
        # qué contestar, así que se devuelve a sí mismo.
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        if self._llamadas >= len(self.respuestas):
            raise RuntimeError("El modelo falso se quedó sin respuestas preparadas.")
        mensaje = self.respuestas[self._llamadas]
        self._llamadas += 1
        return ChatResult(generations=[ChatGeneration(message=mensaje)])


def respuesta_ok(datos: dict[str, Any], finish_reason: str = "tool_calls") -> AIMessage:
    """Respuesta bien formada: el modelo "llamó a la herramienta" con estos datos."""

    return AIMessage(
        content="",
        tool_calls=[{"name": NOMBRE_HERRAMIENTA, "args": datos, "id": "call_ok", "type": "tool_call"}],
        response_metadata={"finish_reason": finish_reason},
    )


def respuesta_json_roto(json_crudo: str, finish_reason: str = "tool_calls") -> AIMessage:
    """Respuesta con JSON que no se puede leer (por ejemplo, falta cerrar llaves)."""

    return AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_roto",
                    "type": "function",
                    "function": {"name": NOMBRE_HERRAMIENTA, "arguments": json_crudo},
                }
            ]
        },
        response_metadata={"finish_reason": finish_reason},
    )


def respuesta_cortada(proveedor: str = "openai") -> AIMessage:
    """Respuesta cortada por tokens, con el campo que usa cada proveedor."""

    metadata = (
        {"finish_reason": "length"} if proveedor == "openai" else {"stop_reason": "max_tokens"}
    )
    mensaje = respuesta_json_roto('{"tecnologias": ["Kafka", "Spar')
    mensaje.response_metadata = metadata
    return mensaje


def respuesta_sin_objeto() -> AIMessage:
    """El modelo contestó texto suelto en vez del objeto pedido."""

    return AIMessage(content="No encontré tecnologías.", response_metadata={"finish_reason": "stop"})


def respuesta_negativa() -> AIMessage:
    """El modelo se niega a responder (bloque de tipo "refusal")."""

    return AIMessage(
        content=[{"type": "refusal", "refusal": "No puedo ayudar con eso."}],
        response_metadata={"finish_reason": "stop"},
    )
