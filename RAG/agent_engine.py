"""
agent_engine.py
==========================================================
FASE 2: Motor del Agente IA (RAG + Triage + Memoria)

Este módulo contiene toda la lógica del agente:
  1. Reglas de triage determinísticas (urgencia, enojo, complejidad).
  2. Recuperación de documentos desde ChromaDB (Top-3).
  3. Cadena de LangChain que combina contexto + historial + pregunta.
  4. Parseo estricto de la salida JSON con Pydantic.
  5. Fallback seguro: cualquier error retorna DERIVAR.

La interfaz principal es `process_query(pregunta, memory)`.
"""

import json
import re
import unicodedata
from typing import Literal

from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field, ValidationError

from config import get_embeddings, get_llm, get_settings


# ============================================================
# 1. ESQUEMA DE SALIDA (Pydantic)
# ============================================================
class TriageOutput(BaseModel):
    """
    Esquema que el LLM debe respetar en su respuesta JSON.

    - accion: RESPONDER (usa RAG) o DERIVAR (escala a humano).
    - nivel_riesgo_clinico: BAJO, MEDIO o ALTO.
    - respuesta_al_usuario: texto empático y claro.
    - resumen_para_humano: solo se llena cuando accion == DERIVAR.
    """
    accion: Literal["RESPONDER", "DERIVAR"] = Field(
        ..., description="Acción a tomar: RESPONDER o DERIVAR"
    )
    nivel_riesgo_clinico: Literal["BAJO", "MEDIO", "ALTO"] = Field(
        ..., description="Nivel de riesgo clínico estimado"
    )
    respuesta_al_usuario: str = Field(
        ..., description="Mensaje final que verá el usuario"
    )
    resumen_para_humano: str = Field(
        default="",
        description="Resumen técnico de 1 línea para el asesor humano (solo si se deriva)",
    )


# ============================================================
# 2. REGLAS DE TRIAGE DETERMINÍSTICO
# ============================================================
class TriageRules:
    """
    Capa de seguridad primaria: detecta palabras clave de urgencia,
    enojo o complejidad y fuerza la derivación sin pasar por RAG.
    """

    # Palabras relacionadas con emergencias médicas
    URGENT_KEYWORDS = [
        "infarto", "ataque cardiaco", "paro cardiaco", "no respira",
        "sangrado", "sangrando", "hemorragia", "convulsion", "convulsiones",
        "perdi el conocimiento", "perdida de conocimiento", "desmayo",
        "accidente", "accidente cerebrovascular", "derrame", "quemadura",
        "fractura", "roto", "hueso roto", "dificultad para respirar",
        "fiebre alta", "fiebre de", "temperatura alta", "dolor intenso",
        "dolor fuerte", "dolor extremo", "muy mal", "emergencia",
        "urgencia", "ambulancia", "llamar al 105",
    ]

    # Palabras de enojo o solicitud explícita de humano
    ANGER_KEYWORDS = [
        "estupido", "estupida", "idiota", "imbecil", "mierda", "carajo",
        "reclamo", "queja", "denuncia", "indignado", "indignada", "furioso",
        "hablar con humano", "hablar con una persona", "con un humano",
        "operador", "supervisor", "gerente", "responsable", "no me atiende",
        "pesimo", "pésimo", "terrible", "horrible", "nunca mas",
    ]

    # Palabras de complejidad fuera de la base (diagnóstico, receta, tratamiento)
    COMPLEXITY_KEYWORDS = [
        "diagnostico", "diagnosticar", "que tengo", "me esta dando",
        "recetar", "receta", "medicamento", "pastilla", "pildora",
        "tratamiento", "tomar para", "curar", "cura", "sintoma de",
        "embarazo de riesgo", "cronico", "crónico", "enfermedad grave",
        "tumor", "cancer", "cáncer", "quimioterapia", "cirugia", "operacion",
    ]

    @classmethod
    def _normalize(cls, text: str) -> str:
        """
        Normaliza el texto: minúsculas, sin tildes y sin signos de puntuación.
        Esto hace que la detección de palabras clave sea más robusta.
        """
        text = text.lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    @classmethod
    def evaluate(cls, message: str) -> TriageOutput | None:
        """
        Evalúa el mensaje del usuario contra las listas de palabras clave.
        Si encuentra coincidencia, retorna una derivación inmediata.
        Si no, retorna None y se continúa con el flujo RAG.
        """
        normalized = cls._normalize(message)

        # --- Urgencias médicas ---
        for kw in cls.URGENT_KEYWORDS:
            if kw in normalized:
                return TriageOutput(
                    accion="DERIVAR",
                    nivel_riesgo_clinico="ALTO",
                    respuesta_al_usuario=(
                        "Entiendo que estás pasando por una situación difícil. "
                        "Por tu seguridad, te transfiero de inmediato con nuestro personal médico."
                    ),
                    resumen_para_humano=f"URGENCIA detectada por palabra clave: '{kw}'",
                )

        # --- Enojo / solicitud de humano ---
        for kw in cls.ANGER_KEYWORDS:
            if kw in normalized:
                return TriageOutput(
                    accion="DERIVAR",
                    nivel_riesgo_clinico="MEDIO",
                    respuesta_al_usuario=(
                        "Lamento mucho la situación. Voy a transferirte con un asesor humano "
                        "para que pueda atenderte personalmente."
                    ),
                    resumen_para_humano=f"ENOJO/SOLICITUD HUMANA detectada por palabra clave: '{kw}'",
                )

        # --- Complejidad médica ---
        for kw in cls.COMPLEXITY_KEYWORDS:
            if kw in normalized:
                return TriageOutput(
                    accion="DERIVAR",
                    nivel_riesgo_clinico="ALTO",
                    respuesta_al_usuario=(
                        "Entiendo tu consulta, pero no puedo brindarte un diagnóstico ni recomendar medicamentos. "
                        "Te derivaré con un profesional de salud para que te oriente de manera segura."
                    ),
                    resumen_para_humano=f"CONSULTA MÉDICA COMPLEJA detectada por palabra clave: '{kw}'",
                )

        return None


# ============================================================
# 3. CARGA DEL VECTOR STORE Y RETRIEVER
# ============================================================
def load_vector_store(embeddings=None, persist_dir: str | None = None) -> Chroma:
    """Carga una base ChromaDB previamente persistida."""
    settings = get_settings()
    if embeddings is None:
        embeddings = get_embeddings(settings)
    if persist_dir is None:
        persist_dir = settings.chroma_persist_dir

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="saludplus",
    )


def get_retriever(vector_store: Chroma, k: int = 3):
    """Retorna un retriever configurado para traer los top-k documentos."""
    return vector_store.as_retriever(search_kwargs={"k": k})


# ============================================================
# 4. FORMATO DE DOCUMENTOS RECUPERADOS
# ============================================================
def format_docs(docs: list[Document]) -> str:
    """
    Une los chunks recuperados en un solo string numerado.
    Este string es el "contexto" que se inyecta en el prompt.
    """
    if not docs:
        return "NO HAY INFORMACIÓN RELEVANTE EN LA BASE DE CONOCIMIENTO."

    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[{i}] {doc.page_content.strip()}")

    return "\n\n".join(formatted)


# ============================================================
# 5. CONSTRUCCIÓN DE LA CADENA RAG
# ============================================================
def build_chain(llm, retriever, memory: ConversationBufferMemory):
    """
    Construye la cadena LCEL:

        pregunta -> {context, question, history} -> prompt -> LLM -> parser JSON

    La memoria se lee en cada invocación mediante una lambda, lo que permite
    que el historial de la sesión se actualice dinámicamente.
    """

    # Parser que fuerza la salida al esquema TriageOutput
    parser = JsonOutputParser(pydantic_object=TriageOutput)

    # Prompt con guardrails estrictos
    system_template = """Eres el "Agente de Triage Inteligente" de SaludPlus Perú.

CONTEXTO RECUPERADO DE LA BASE DE CONOCIMIENTO:
{context}

REGLAS ESTRICTAS (GUARDRAILS):
1. NO eres médico. NUNCA diagnosticques, recetes medicamentos ni recomiendes tratamientos.
2. Responde ÚNICAMENTE con información presente en el CONTEXTO RECUPERADO.
3. Si la pregunta no puede responderse con el contexto, o trata sobre salud grave, DERIVA.
4. Si el usuario muestra enojo, pide hablar con humano o hace una consulta compleja, DERIVA.
5. Sé empático, claro y breve en tus respuestas.

HISTORIAL DE LA CONVERSACIÓN:
{history}

Tu tarea es clasificar la consulta del usuario y responder en español.

{format_instructions}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "PREGUNTA DEL PACIENTE: {question}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    # Runnable que lee la memoria actual
    def _load_memory(_) -> list:
        return memory.load_memory_variables({}).get("history", [])

    chain = (
        RunnableParallel(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "history": RunnableLambda(_load_memory),
            }
        )
        | prompt
        | llm
        | parser
    )

    return chain


# ============================================================
# 6. ORQUESTACIÓN PRINCIPAL
# ============================================================
def process_query(question: str, memory: ConversationBufferMemory) -> dict:
    """
    Punto de entrada principal del agente.

    Retorna un diccionario con:
      - output: objeto TriageOutput
      - sources: lista de Documentos recuperados (solo en modo RAG)
      - routing: "deterministic" o "rag"

    Flujo:
      1. Triage determinístico (bypass si es urgencia/enojo/complejidad).
      2. Cargar vector store y retriever.
      3. Invocar cadena RAG.
      4. Parsear/validar con Pydantic.
      5. Guardar en memoria.
      6. Fallback seguro ante cualquier error.
    """
    settings = get_settings()

    try:
        # 1. Triage determinístico primero (cortocircuito)
        triage_result = TriageRules.evaluate(question)
        if triage_result:
            memory.save_context(
                {"input": question},
                {"output": triage_result.respuesta_al_usuario},
            )
            return {
                "output": triage_result,
                "sources": [],
                "routing": "deterministic",
            }

        # 2. Cargar RAG
        embeddings = get_embeddings(settings)
        vector_store = load_vector_store(embeddings, settings.chroma_persist_dir)
        retriever = get_retriever(vector_store, k=3)

        # Recuperar documentos para mostrarlos en la UI
        sources = retriever.invoke(question)

        # 3. Cadena
        llm = get_llm(settings)
        chain = build_chain(llm, retriever, memory)

        # 4. Invocar LLM
        raw_output = chain.invoke(question)

        # 5. Validar con Pydantic
        if isinstance(raw_output, dict):
            result = TriageOutput(**raw_output)
        else:
            result = TriageOutput(**json.loads(raw_output))

        # 6. Guardar en memoria
        memory.save_context(
            {"input": question},
            {"output": result.respuesta_al_usuario},
        )

        return {
            "output": result,
            "sources": sources,
            "routing": "rag",
        }

    except ValidationError as e:
        # El LLM devolvió JSON pero no cumple el esquema
        fallback = TriageOutput(
            accion="DERIVAR",
            nivel_riesgo_clinico="ALTO",
            respuesta_al_usuario=(
                "No pude procesar tu consulta de forma segura. "
                "Te transferiré con un asesor humano."
            ),
            resumen_para_humano=f"Error de validación JSON: {e}",
        )
        return {"output": fallback, "sources": [], "routing": "error"}

    except Exception as e:
        # Cualquier otro error: API, red, vector store vacío, etc.
        fallback = TriageOutput(
            accion="DERIVAR",
            nivel_riesgo_clinico="ALTO",
            respuesta_al_usuario=(
                "Ocurrió un problema técnico. Por seguridad, te derivaré con un asesor humano."
            ),
            resumen_para_humano=f"Error técnico en process_query: {e}",
        )
        return {"output": fallback, "sources": [], "routing": "error"}


# ============================================================
# 7. HELPER PARA CREAR MEMORIA LIMPIA
# ============================================================
def create_memory() -> ConversationBufferMemory:
    """Crea una nueva memoria de conversación vacía."""
    return ConversationBufferMemory(memory_key="history", return_messages=True)


# Permite probar el motor desde la terminal
if __name__ == "__main__":
    import os

    # Requiere una API key de Gemini para funcionar
    if not os.getenv("GOOGLE_API_KEY"):
        print("[ERROR] Define la variable de entorno GOOGLE_API_KEY antes de probar.")
        raise SystemExit(1)

    mem = create_memory()
    test_questions = [
        "¿Qué seguros aceptan en Lima Centro?",
        "Me duele mucho el pecho y tengo mareos",
        "¿Y en Arequipa?",
    ]

    for q in test_questions:
        print(f"\n[PACIENTE] {q}")
        response = process_query(q, mem)
        result = response["output"]
        print(f"[AGENTE] {result.accion} | Riesgo: {result.nivel_riesgo_clinico} | Routing: {response['routing']}")
        print(f"         {result.respuesta_al_usuario}")
        if result.resumen_para_humano:
            print(f"         Resumen humano: {result.resumen_para_humano}")
