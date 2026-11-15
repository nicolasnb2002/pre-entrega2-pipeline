# Pre-entrega 2 — Pipeline de procesamiento validado

Trabajo de **Nazareno Bergamo** para la pre-entrega 2 de Coderhouse (AI Engineering).
Es un **Pipeline de Extracción de Entidades Técnicas** en Python 3.12 con LangChain. Recibe un párrafo sin procesar (un log de error o una descripción de arquitectura) y devuelve un objeto validado con Pydantic: tecnologías, nivel de criticidad y resumen técnico.

La configuración del proveedor es la misma del Módulo 1 (pre-entrega 1): la variable `LLM_PROVIDER` elige `openai` o `anthropic`, y las claves salen solo del entorno. Además, con `OPENAI_BASE_URL` se puede usar un proveedor **gratuito** compatible con OpenAI (Gemini o Groq) sin cambiar el código: ver "Probarlo gratis con Gemini o Groq".

No hay informe ni PDF, como pide la consigna. Todo está en este README.

## Qué pide la consigna y dónde está

| Pedido | Dónde |
| --- | --- |
| Modelo Pydantic en `schemas.py` | `ExtraccionTecnica` en `schemas.py` |
| `tecnologias` lista de strings, no vacía | `schemas.py`: `min_length=1` y el validador `limpiar_tecnologias` |
| `nivel_de_criticidad` enum baja, media, alta | `NivelDeCriticidad` en `schemas.py` |
| `resumen_tecnico` string | `schemas.py` |
| Cliente `ChatOpenAI` o `ChatAnthropic`, reutilizando el Módulo 1 | `crear_modelo()` en `chain.py` y `entorno.py`. Gemini y Groq entran por el mismo `ChatOpenAI` con `base_url` |
| `ChatPromptTemplate` modular que acepta el texto y las instrucciones de formato | `construir_prompt()` en `chain.py`: variables `{texto}` e `{instrucciones_formato}` |
| Sin f-strings dentro de la cadena | El prompt usa variables de LangChain. Los únicos f-strings del proyecto están en mensajes de error y logs |
| Cadena LCEL `prompt \| model.with_structured_output(Schema)` | `construir_cadena()` en `chain.py` |
| Al menos un reintento automático con `.with_retry()` ante JSON mal formado o incompleto | `construir_cadena()`: `with_retry(retry_if_exception_type=..., stop_after_attempt=3)` |
| No ignorar el `finish_reason` | `motivo_de_corte()` y `verificar_respuesta()` en `chain.py` |
| Función async `process_text(text: str)` con `.ainvoke()` | `process_text()` en `chain.py` |
| Logs para ver la validación y los reintentos | Módulo `logging`, logger `pipeline`, y `RegistroDeIntentos` en `chain.py` |
| Mini-script de prueba asíncrono que ejecute un ejemplo | `demo.py` |
| Prueba de estrés con texto ambiguo | Casos 4 y 5 de `demo.py` y los tests de reintento |
| Ejemplo de salida JSON e instrucciones en el README | Este archivo |

## Decisiones que la consigna no fijó

Si la cátedra pide otro valor, se cambia en el lugar indicado.

| Tema | Qué hice | Dónde cambiarlo |
| --- | --- | --- |
| Nombre del modelo Pydantic | `ExtraccionTecnica` | `schemas.py` |
| Restricciones de `tecnologias` | Sin espacios en los costados, sin vacíos, sin duplicados (sin distinguir mayúsculas), de 1 a 20 elementos | `limpiar_tecnologias` y `MAX_TECNOLOGIAS` en `schemas.py` |
| Mayúsculas en la criticidad | `"Alta"` se acepta y se pasa a `"alta"`. Otra palabra (ej. `"urgente"`) falla | `criticidad_en_minusculas` en `schemas.py` |
| Largo de `resumen_tecnico` | De 10 a 600 caracteres | `MIN_LARGO_RESUMEN` y `MAX_LARGO_RESUMEN` en `schemas.py` |
| Campos de más | Se rechazan (`extra="forbid"`) | `model_config` en `schemas.py` |
| Cantidad de intentos | 3 en total (1 normal + 2 reintentos). La consigna pide "al menos uno" | `INTENTOS_MAXIMOS` en `chain.py` |
| Espera entre intentos | Exponencial con azar: ~1 s, ~2 s (tope 4 s) | `ESPERA_ENTRE_INTENTOS` en `chain.py` |
| Qué se reintenta | Solo errores de la respuesta: cortada por tokens, JSON roto, no cumple el esquema o falta el objeto. Un bug del programa no se reintenta | `ERRORES_REINTENTABLES` en `chain.py` |
| Errores de red | Los reintenta el SDK de cada proveedor (trae sus propios reintentos), no `with_retry` | `crear_modelo()` en `chain.py` |
| Tope de tokens de salida | 512. Para Gemini y Groq recomiendo 2048, porque esos modelos razonan y el razonamiento gasta tokens del mismo tope | `LLM_MAX_TOKENS` en `.env` o `MAX_TOKENS_RESPUESTA` en `chain.py` |
| Proveedores gratuitos | Se usan con el mismo `ChatOpenAI`, cambiando la URL con `OPENAI_BASE_URL`. No hay una clase nueva ni un tercer valor de `LLM_PROVIDER` | `leer_base_url()` en `entorno.py` y `crear_modelo()` en `chain.py` |
| Sin `OPENAI_BASE_URL` | No se le pasa `base_url` a `ChatOpenAI`, así usa la URL oficial de OpenAI | `crear_modelo()` en `chain.py` |
| URL mal escrita | Si `OPENAI_BASE_URL` no empieza con `https://` o `http://`, falla antes de llamar | `leer_base_url()` en `entorno.py` |
| Modelo gratuito de Gemini | `gemini-3.5-flash-lite` (estable, con nivel gratuito y function calling según la doc de Google) | `OPENAI_MODEL` en `.env` |
| Modelo gratuito de Groq | `openai/gpt-oss-20b` (modelo de producción, con nivel gratuito y tool use según la doc de Groq) | `OPENAI_MODEL` en `.env` |
| Tiempo máximo por pedido | 30 segundos | `TIMEOUT_SEGUNDOS` en `chain.py` |
| Método de salida estructurada | `method="function_calling"` en los dos proveedores (ver "El finish_reason") | `construir_cadena()` en `chain.py` |
| Temperatura | 0 en OpenAI. A Anthropic no se le manda, igual que en el Módulo 1 | `crear_modelo()` en `chain.py` |
| Qué devuelve `process_text` | El objeto `ExtraccionTecnica`. Para JSON: `.model_dump(mode="json")` | `chain.py` |
| Si el modelo se niega a responder | Lanza `RespuestaRechazada` sin reintentar | `verificar_respuesta()` en `chain.py` |
| Qué pasa si se agotan los intentos | Lanza `ExtraccionFallida` con el último motivo | `process_text()` en `chain.py` |
| Nombre de la variable de proveedor | `LLM_PROVIDER` (`openai`, `anthropic` o `demo`) | `entorno.py` y `.env.example` |
| Modelo de OpenAI por defecto | `gpt-4o-mini` | `OPENAI_MODEL` o `MODELO_POR_DEFECTO` en `entorno.py` |
| Modelo de Anthropic por defecto | `claude-haiku-5-5` | `ANTHROPIC_MODEL` o `MODELO_POR_DEFECTO` en `entorno.py` |
| Nombre del mini-script | `demo.py` | — |
| Modo demostración | Sin clave, `demo.py` usa un modelo falso y no llama a nadie | `modelo_falso.py` y `demo.py` |

Los ids de modelo son los mismos defaults del Módulo 1. **No los verifiqué contra una cuenta real.** Antes de gastar crédito, contrastalos con los modelos habilitados en la cuenta.

## Ejemplo de salida

Entrada (es el caso 1 de `demo.py`):

```text
ERROR 2026-11-14 18:03:12 api-pagos: FastAPI devuelve 503 en /checkout. El pool de PostgreSQL llegó a 100 conexiones y Redis tarda 2 s en responder el caché de sesiones. El 40% de los pagos están fallando.
```

Salida validada:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API de pagos en FastAPI con caché en Redis y persistencia en PostgreSQL; el pool de conexiones saturado provoca errores 503 en el checkout."
}
```

En modo demostración esa salida la arma el modelo falso. Con un modelo real el texto del resumen va a cambiar, pero la forma es la misma porque la impone Pydantic.

## Instalación

Hace falta **Python 3.12**. En una terminal, parado en esta carpeta:

```bash
python3.12 -m venv .venv
```

Activar el entorno:

- Linux o macOS: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Las versiones pinneadas son las que se usaron para correr las pruebas: `langchain-core` 1.8.0, `langchain-openai` 1.8.0, `langchain-anthropic` 1.7.10, `pydantic` 2.14.3, `python-dotenv` 1.2.4, `pytest` 9.2.0 y `pytest-asyncio` 1.5.0.

Opcional, para tener un `.env`:

```bash
# Linux o macOS
cp .env.example .env
# Windows
copy .env.example .env
```

El `.env` está en `.gitignore`. No lo subas: ahí irían las claves.

## Cómo ejecutarlo

Sin claves, que es la forma segura:

```bash
python demo.py
```

Vas a ver el cartel **MODO DEMOSTRACIÓN** y cinco casos, con sus logs:

1. El ejemplo de la consigna: sale bien en el primer intento.
2. El primer intento devuelve JSON roto: se loguea "Respuesta mal formada", hay un reintento automático y el segundo sale bien.
3. El primer intento se corta por tokens (`finish_reason="length"`): se descarta sin leerlo y se reintenta.
4. Prueba de estrés con texto ambiguo: el primer intento trae una lista vacía y una criticidad inventada, Pydantic lo rechaza, y el segundo trae duplicados y mayúsculas que el esquema limpia.
5. Prueba de estrés con un texto sin tecnologías: los tres intentos fallan y el pipeline termina con `ExtraccionFallida`, sin romper el programa.

En el modo demostración no hay pausa entre reintentos, para que termine rápido. Con un modelo real sí se espera.

Para una llamada real, más adelante y con crédito propio:

1. En `.env`, poné `LLM_PROVIDER=openai` o `LLM_PROVIDER=anthropic`.
2. Completá solo la clave de ese proveedor (`OPENAI_API_KEY` o `ANTHROPIC_API_KEY`).
3. Corré `python demo.py`. Hace el ejemplo y la prueba de estrés con texto ambiguo: dos llamadas, más los reintentos si hacen falta.

Este armado **no hizo esa llamada**. Si la clave está vacía o es el texto de ejemplo (`tu_clave_aca`), el programa se queda en demostración.

Las pruebas:

```bash
python -m pytest
```

Usan dobles: no necesitan claves ni internet.

## Probarlo gratis con Gemini o Groq

Gemini (Google AI Studio) y Groq tienen nivel gratuito y aceptan pedidos con el mismo formato que OpenAI. Por eso alcanza con cambiar la URL: el código sigue usando `ChatOpenAI`, y la clave sigue saliendo de `OPENAI_API_KEY`.

**No está probado con una clave real.** Las URLs, los modelos y el soporte de tool calling salen de la documentación oficial consultada el 14 de noviembre de 2026 (links abajo). Los niveles gratuitos tienen límites de pedidos por minuto y por día, y pueden cambiar.

### Gemini

1. Sacá una clave gratis en https://aistudio.google.com/apikey.
2. Poné esto en el `.env`:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=tu_clave_de_gemini
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-3.5-flash-lite
LLM_MAX_TOKENS=2048
```

3. Corré `python demo.py`.

Fuentes: [compatibilidad con OpenAI](https://ai.google.dev/gemini-api/docs/openai) (URL base y function calling por esa vía, que Google todavía marca como beta), [ficha de Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) (estable, function calling y structured outputs soportados) y [precios](https://ai.google.dev/gemini-api/docs/pricing) (nivel gratuito "Free of charge").

### Groq

1. Sacá una clave gratis en https://console.groq.com/keys.
2. Poné esto en el `.env`:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=tu_clave_de_groq
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=openai/gpt-oss-20b
LLM_MAX_TOKENS=2048
```

3. Corré `python demo.py`.

Fuentes: [compatibilidad con OpenAI](https://console.groq.com/docs/openai) (URL base), [tool use](https://console.groq.com/docs/tool-use) (`openai/gpt-oss-20b` soporta tool use, no en paralelo), [modelos](https://console.groq.com/docs/models) (es modelo de producción) y [límites](https://console.groq.com/docs/rate-limits) (el plan gratuito incluye ese modelo con 30 pedidos por minuto y 1000 por día).

### Por qué tool calling importa acá

`with_structured_output(..., method="function_calling")` le pasa el esquema al modelo como una "herramienta" y le exige que la use (`tool_choice` forzado, `parallel_tool_calls=False`). Si el proveedor no soportara tool calling, no habría objeto y el pipeline lo trataría como respuesta mal formada. Los dos proveedores documentan soporte de tool calling para los modelos elegidos. Lo que no pude verificar sin clave es que acepten exactamente los parámetros que manda `langchain-openai` (por ejemplo `max_completion_tokens` y `parallel_tool_calls`). Si alguno los rechaza, la API responde 400 y se ve en el error.

## Cómo funciona por dentro

```text
texto ─► prompt ─► modelo.with_structured_output(ExtraccionTecnica, include_raw=True) ─► verificar_respuesta ─► objeto
          └────────────── todo esto envuelto en .with_retry() ──────────────┘
```

1. `process_text(text)` rechaza un texto vacío antes de gastar una llamada y llama a `cadena.ainvoke({"texto": ...})`.
2. `construir_prompt()` arma un `ChatPromptTemplate` con un mensaje de sistema y uno de usuario. Las instrucciones de formato se fijan con `.partial()`, así la cadena solo recibe el texto.
3. `with_structured_output(..., include_raw=True)` devuelve tres cosas: `raw` (el mensaje crudo del modelo), `parsed` (el objeto, si se pudo armar) y `parsing_error` (el error, si no).
4. `verificar_respuesta()` mira primero el `finish_reason`. Después mira si hubo error de parseo o si falta el objeto. Si algo anda mal, lanza `RespuestaIncompleta` o `RespuestaMalFormada`.
5. `.with_retry()` atrapa esas excepciones y vuelve a correr la cadena entera, hasta 3 veces.
6. Si se agotan los intentos, `process_text` lanza `ExtraccionFallida`.

## El finish_reason

Si el modelo se queda sin tokens, corta la respuesta a la mitad y el JSON queda incompleto. Cada proveedor lo avisa distinto:

- OpenAI: `response_metadata["finish_reason"] == "length"`.
- Anthropic: `response_metadata["stop_reason"] == "max_tokens"`.

`motivo_de_corte()` revisa los dos. Si la respuesta se cortó, se descarta **aunque parezca que el JSON se leyó bien**, porque podría faltarle información. Eso lo prueba `test_corte_por_tokens_gana_aunque_el_json_parezca_bien`.

Por qué `method="function_calling"`: en OpenAI el método por defecto de `with_structured_output` es `json_schema`. Con ese método, si la respuesta se corta por tokens, el SDK de OpenAI lanza su propio error (`LengthFinishReasonError`) adentro de la llamada al modelo, y el paso de verificación nunca ve el `finish_reason`. Con `function_calling` el mensaje crudo llega siempre a `verificar_respuesta()`, y el control es el mismo para los dos proveedores.

Un detalle: si una respuesta se cortó por tokens, el reintento usa el mismo tope (512 por defecto). Para este esquema chico alcanza con un modelo que no razona; si se cortara seguido (pasa más con modelos que razonan, como Gemini 3.x o gpt-oss), subí `LLM_MAX_TOKENS` en el `.env`.

## Mapa de archivos

- `schemas.py`: el contrato de Pydantic.
- `entorno.py`: lee `LLM_PROVIDER`, las claves, los modelos, `OPENAI_BASE_URL` y `LLM_MAX_TOKENS` del entorno (lógica del Módulo 1).
- `chain.py`: prompt, modelo, cadena LCEL, verificación, reintentos, logs y `process_text()`.
- `modelo_falso.py`: un chat model de mentira para el demo y las pruebas.
- `demo.py`: el mini-script asíncrono.
- `tests/`: pruebas con pytest.
- `requirements.txt`, `.env.example`, `.gitignore`, `pytest.ini`.

## Cómo defenderlo en clase

- **Schemas (Pydantic):** `ExtraccionTecnica` es el contrato: tres campos, la lista no puede estar vacía y la criticidad solo puede ser baja, media o alta.
  Si el modelo devuelve algo que no cumple, Pydantic lo rechaza y ese error se puede reintentar.
- **Proveedores gratuitos:** Gemini y Groq hablan el mismo "idioma" que OpenAI, así que uso el mismo `ChatOpenAI` y solo cambio la URL con `OPENAI_BASE_URL`.
  Si la variable no está, no se pasa `base_url` y va a OpenAI; la clave sale siempre de `OPENAI_API_KEY`.
- **Prompt template:** `ChatPromptTemplate` con las variables `{texto}` e `{instrucciones_formato}`, sin f-strings.
  LangChain rellena las variables al invocar, y las instrucciones se pueden cambiar sin tocar la cadena.
- **Cadena LCEL:** `prompt | modelo.with_structured_output(ExtraccionTecnica) | verificar_respuesta`.
  El `|` conecta pasos: la salida de uno es la entrada del siguiente.
- **with_retry:** envuelve la cadena entera y la vuelve a correr si sale un error de respuesta (JSON roto, incompleto o inválido).
  Son 3 intentos con espera exponencial; un error de programación no se reintenta.
- **Control de finish_reason:** con `include_raw=True` veo el mensaje crudo y reviso si cortó por tokens (`length` en OpenAI, `max_tokens` en Anthropic).
  Si cortó, descarto la respuesta antes de transformarla y lanzo un error que dispara el reintento.
- **process_text async:** es la función pública: valida que el texto no esté vacío y llama a `cadena.ainvoke()`.
  Al ser async, no bloquea el programa mientras espera al modelo; si se agotan los intentos lanza `ExtraccionFallida`.
- **Logs:** el logger `pipeline` muestra cada intento, cada rechazo con su motivo y la validación OK.
  Los reintentos se ven gracias a un callback que lee la etiqueta `retry:attempt:N` que pone LangChain.
- **Demo y tests:** `demo.py` corre cinco casos con un modelo falso, sin red ni claves, incluida la prueba de estrés.
  Los tests usan ese mismo modelo falso para comprobar validaciones, reintentos y el corte por tokens.

## Pruebas que hay

- Pydantic: objeto válido, limpieza de espacios y duplicados, mayúsculas en la criticidad, lista vacía, criticidad fuera del enum, resumen vacío o muy largo, campos de más y campos faltantes.
- Prompt: variables de la plantilla e instrucciones reemplazables.
- Reintentos: se recupera tras JSON roto, tras validación fallida y tras respuesta sin objeto; agota los 3 intentos y lanza `ExtraccionFallida`; un error que no es de la respuesta no se reintenta; una negativa del modelo no se reintenta.
- `finish_reason`: detecta `length` y `max_tokens`, no marca respuestas normales, y descarta una respuesta cortada aunque el JSON parezca bien.
- `process_text`: es async, devuelve el objeto, rechaza texto vacío sin llamar al modelo, y arma su propia cadena si no se le pasa una.
- Configuración: proveedor por defecto, proveedor inválido, modelos por defecto, claves placeholder, y construcción de `ChatOpenAI` y `ChatAnthropic` con una clave falsa sin llamarlos.
- Demo: corre los cinco casos en modo demostración.
- `OPENAI_BASE_URL`: con la URL de Gemini o Groq se pasa `base_url` a `ChatOpenAI`; sin la variable (o vacía) no se pasa y queda la URL oficial de OpenAI; una URL mal escrita falla; Anthropic no la usa. `LLM_MAX_TOKENS` por defecto, desde el entorno e inválido.

## Límites conocidos

- No se hizo ninguna llamada real a OpenAI, Anthropic, Gemini ni Groq. Los ids de modelo no están verificados contra una cuenta.
- La compatibilidad OpenAI de Gemini está en beta según Google. No verifiqué que Gemini o Groq acepten todos los parámetros que manda `langchain-openai`, ni que devuelvan `finish_reason="length"` igual que OpenAI cuando se cortan.
- El modelo falso imita el formato de respuesta de LangChain (tool calls y `response_metadata`). Un modelo real puede devolver campos extra en la metadata; el control solo mira `finish_reason` y `stop_reason`.
- Si el modelo se niega a responder (un bloque "refusal" o `stop_reason="refusal"` en Anthropic), `process_text` lanza `RespuestaRechazada` y no reintenta, porque repetir el mismo pedido daría lo mismo. Esto está probado con el modelo falso, no con una negativa real.
