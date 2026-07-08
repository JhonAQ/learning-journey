"""
app.py
==========================================================
FASE 3: Interfaz de Usuario con Streamlit
Diseño limpio y funcional para SaludPlus Peru - Triage IA
"""

import os

import streamlit as st

from agent_engine import create_memory, process_query
from config import ensure_project_dirs, get_llm, get_settings


# ============================================================
# 1. CONFIGURACION DE PAGINA
# ============================================================
st.set_page_config(
    page_title="SaludPlus Peru | Triage IA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. ESTILOS MINIMALES Y FUNCIONALES
# ============================================================
st.markdown(
    """
    <style>
        .main-header {
            background-color: #1A5F7A;
            padding: 1.2rem 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1rem;
        }
        .main-header h1 {
            margin: 0;
            font-size: 1.5rem;
            font-weight: 600;
        }
        .main-header p {
            margin: 0.2rem 0 0 0;
            opacity: 0.9;
            font-size: 0.9rem;
        }
        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            background-color: #2D6A4F;
            border-radius: 50%;
            margin-right: 6px;
        }
        .route-label {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            display: inline-block;
            margin-top: 0.4rem;
        }
        .route-rag { background-color: #E8F5E9; color: #2D6A4F; }
        .route-det { background-color: #FFF3E0; color: #D68C45; }
        .route-err { background-color: #FFEBEE; color: #B42318; }
        .source-card {
            background-color: #FAFAFA;
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
        }
        .source-tag {
            font-size: 0.65rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #1A5F7A;
            background-color: #E3F2FD;
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. ESTADO DE SESION
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = create_memory()


# ============================================================
# 4. BARRA LATERAL
# ============================================================
with st.sidebar:
    st.title("SaludPlus")
    st.caption("Triage IA v1.0")

    settings = get_settings()
    default_key = (
        settings.google_api_key if settings.llm_provider == "google" else settings.openai_api_key
    )

    api_key = st.text_input(
        "API key",
        value=default_key,
        type="password",
        help="Sobrescribe la variable de entorno para esta sesion.",
        label_visibility="collapsed",
        placeholder="Pega tu API key aqui...",
    )

    connection_ok = False
    if api_key:
        if settings.llm_provider == "google":
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            os.environ["OPENAI_API_KEY"] = api_key
        try:
            get_llm(settings)
            connection_ok = True
        except Exception:
            connection_ok = False

    if api_key and connection_ok:
        st.success("Conectado")
    elif api_key and not connection_ok:
        st.warning("Sin conexion")
    else:
        st.info("Key requerida")

    st.divider()

    st.markdown(f"**LLM:** {settings.llm_provider.upper()}")
    st.caption(settings.gemini_model if settings.llm_provider == "google" else settings.openai_model)
    st.markdown(f"**Embeddings:** {settings.embedding_provider.upper()}")

    st.divider()

    if st.button("Nueva conversacion", use_container_width=True, type="primary"):
        st.session_state.chat_history = []
        st.session_state.memory = create_memory()
        st.rerun()


# ============================================================
# 5. HEADER
# ============================================================
st.markdown(
    """
    <div class="main-header">
        <h1>Centro de Triage IA</h1>
        <p><span class="status-dot"></span>Sistema activo — Red de clinicas SaludPlus Peru</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. VERIFICAR BASE VECTORIAL
# ============================================================
ensure_project_dirs(settings)

if not os.path.exists(settings.chroma_persist_dir) or not os.listdir(settings.chroma_persist_dir):
    st.error("No se encontro la base de conocimiento. Ejecuta: python ingest.py")
    st.stop()


# ============================================================
# 7. CONTENIDO PRINCIPAL EN DOS COLUMNAS
# ============================================================
chat_col, context_col = st.columns([1.8, 1])

with chat_col:
    st.subheader("Conversacion")

    if not st.session_state.chat_history:
        st.info(
            "Bienvenido al asistente de triage. Puedes consultar horarios, seguros, "
            "especialidades o el proceso de citas. Si tu caso es una urgencia o requiere "
            "diagnostico, te derivaremos con un profesional.",
        )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg.get("role") == "assistant":
                routing = msg.get("routing", "rag")
                badge_class = "route-rag" if routing == "rag" else "route-det" if routing == "deterministic" else "route-err"
                badge_text = "Respuesta RAG" if routing == "rag" else "Derivacion automatica" if routing == "deterministic" else "Error de sistema"
                st.markdown(
                    f'<span class="route-label {badge_class}">{badge_text}</span>',
                    unsafe_allow_html=True,
                )

                if msg.get("derivado"):
                    st.error(
                        f"**Derivacion a asesor humano**  \n"
                        f"Riesgo clinico: {msg['riesgo']}  \n"
                        f"Nota: {msg['resumen_tecnico']}"
                    )

    pregunta = st.chat_input("Escribe tu consulta...")

    if pregunta:
        if not api_key:
            st.error("Ingresa tu API key en el panel lateral.")
            st.stop()

        st.session_state.chat_history.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.write(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Analizando..."):
                response = process_query(pregunta, st.session_state.memory)

            result = response["output"]
            es_derivado = result.accion == "DERIVAR"
            routing = response["routing"]

            badge_class = "route-rag" if routing == "rag" else "route-det" if routing == "deterministic" else "route-err"
            badge_text = "Respuesta RAG" if routing == "rag" else "Derivacion automatica" if routing == "deterministic" else "Error de sistema"

            st.write(result.respuesta_al_usuario)
            st.markdown(
                f'<span class="route-label {badge_class}">{badge_text}</span>',
                unsafe_allow_html=True,
            )

            if es_derivado:
                st.error(
                    f"**Derivacion a asesor humano**  \n"
                    f"Riesgo clinico: {result.nivel_riesgo_clinico}  \n"
                    f"Nota: {result.resumen_para_humano}"
                )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": result.respuesta_al_usuario,
                    "derivado": es_derivado,
                    "riesgo": result.nivel_riesgo_clinico,
                    "resumen_tecnico": result.resumen_para_humano,
                    "routing": routing,
                }
            )

            if response["sources"]:
                st.session_state.last_sources = response["sources"]

with context_col:
    st.subheader("Contexto")

    with st.container(border=True):
        st.markdown("**Documentos recuperados**")
        sources = st.session_state.get("last_sources", [])
        if sources:
            for src in sources:
                category = src.metadata.get("category", "general")
                with st.container():
                    st.markdown(
                        f'<span class="source-tag">{category}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(src.page_content[:200] + "...")
        else:
            st.caption(
                "Aun no hay documentos recuperados. Escribe una consulta administrativa "
                "para ver los chunks que usa el RAG."
            )

    with st.container(border=True):
        st.markdown("**Sobre SaludPlus**")
        st.caption(
            "Red de 6 clinicas en Peru. Atencion presencial, teleconsulta y "
            "emergencias 24/7 en sedes seleccionadas."
        )
        st.caption("Seguros: EPS Rimac, EPS Pacifico, EsSalud (referidos), Mapfre, particular.")

    with st.container(border=True):
        st.markdown("**Como funciona**")
        st.caption(
            "1. **Triage:** detectamos urgencias, enojo o complejidad medica.  \n"
            "2. **Recuperacion:** buscamos los 3 documentos mas relevantes.  \n"
            "3. **Respuesta:** el LLM responde solo con informacion validada."
        )


# ============================================================
# 8. FOOTER
# ============================================================
st.divider()
st.caption("Laboratorio academico — Datos completamente ficticios — LangChain + ChromaDB + Streamlit")
