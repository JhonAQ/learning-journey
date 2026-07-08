# Laboratorio: Agente de Triage RAG - SaludPlus Perú

Proyecto académico de un agente conversacional inteligente para una red de clínicas peruanas. El agente utiliza **RAG (Retrieval-Augmented Generation)** con **LangChain**, **ChromaDB** y **Streamlit**.

## Arquitectura

```
usuario -> Streamlit (app.py)
              |
              v
      agent_engine.py
              |
      +-------+-------+
      |               |
  Triage           RAG + LLM
  deterministico   (ChromaDB + Gemini)
      |               |
      +-------+-------+
              |
      Respuesta / Derivacion
```

## Estructura de archivos

| Archivo | Descripción |
|---------|-------------|
| `config.py` | Carga variables de entorno y crea LLM/embeddings |
| `ingest.py` | Genera datos ficticios y construye la base vectorial |
| `agent_engine.py` | Motor de triage, RAG, memoria y guardrails |
| `app.py` | Interfaz de chat con Streamlit |
| `requirements.txt` | Dependencias del proyecto |
| `.env.example` | Plantilla de variables de entorno |
| `data/saludplus_data.json` | Datos generados por `ingest.py` |
| `chroma_db/` | Base de datos vectorial persistida |

## Requisitos

- Python 3.13 (o superior)
- Una API key de [Google AI Studio](https://aistudio.google.com/app/apikey) (para Gemini)

## Instalación

```bash
# 1. Entrar al directorio
cd C:\Users\JhonAQ\Desktop\Projects\learning-journey\RAG

# 2. Activar entorno virtual (ya creado)
.venv\Scripts\activate

# 3. Instalar dependencias
python -m pip install -r requirements.txt
```

## Configuración

Copia el archivo de ejemplo y pega tu API key:

```bash
copy .env.example .env
```

Edita `.env`:

```ini
GOOGLE_API_KEY=tu_api_key_aqui
```

> También puedes dejar `.env` sin key e ingresarla directamente en la barra lateral de Streamlit.

## Ejecución

### Paso 1: Generar la base de conocimiento vectorial

```bash
python ingest.py
```

Esto crea:
- `data/saludplus_data.json`
- `chroma_db/` (base vectorial con embeddings locales)

### Paso 2: Ejecutar la aplicación

```bash
streamlit run app.py
```

Abre tu navegador en `http://localhost:8501`.

## Casos de prueba

| Tipo de consulta | Ejemplo | Comportamiento esperado |
|------------------|---------|--------------------------|
| Administrativa | "¿Qué seguros aceptan en Lima Centro?" | RESPONDER con datos de RAG |
| Urgencia médica | "Me duele mucho el pecho y tengo mareos" | DERIVAR inmediatamente |
| Enojo | "Quiero hablar con un humano" | DERIVAR a asesor |
| Complejidad médica | "¿Qué pastilla me tomo para la migraña?" | DERIVAR por seguridad |
| Memoria | "¿Y en Arequipa?" | Responder con contexto previo |

## Decisiones técnicas

- **Embeddings locales**: se usan por defecto (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) para no depender de cuota de API.
- **Triage determinístico**: capa de palabras clave que actúa antes del LLM, garantizando derivación rápida en casos de riesgo.
- **Memoria**: `ConversationBufferMemory` permite mantener contexto dentro de una sesión de chat.
- **Fallback seguro**: cualquier error técnico o JSON malformado retorna `DERIVAR` con riesgo ALTO.

## Notas para el estudiante

- Nunca se debe hardcodear una API key en el código; usa `.env`.
- `chroma_db/` no se versiona (está en `.gitignore`) porque se regenera con `ingest.py`.
- Puedes cambiar el proveedor de LLM a OpenAI editando `LLM_PROVIDER` en `.env` e instalando las dependencias opcionales.

## Licencia

Proyecto educativo. Datos completamente ficticios.
