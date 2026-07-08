import streamlit as st
import google.generativeai as genai
import json

# ==========================================
# 1. BASE DE CONOCIMIENTO (RAG Simulado)
# ==========================================
# En la vida real esto vendría de una base de datos SQL o MongoDB.
# Aquí lo inyectamos como contexto para que el modelo no alucine.
BASE_CONOCIMIENTO = {
    "empresa": "SaludPlus Perú",
    "sedes": ["Lima Centro", "Lima Sur", "Arequipa Norte", "Trujillo Centro"],
    "horarios_atencion": "Lunes a Sábado de 8:00 AM a 8:00 PM. Emergencias 24/7.",
    "seguros_aceptados": ["EPS Rimac", "EPS Pacifico", "EsSalud (solo referidos)", "Mapfre"],
    "especialidades": ["Medicina General", "Pediatría", "Cardiología", "Ginecología"],
    "proceso_citas": "Las citas se agendan mediante la App SaludPlus o presencialmente con DNI."
}

# ==========================================
# 2. SYSTEM PROMPT MAESTRO (Guardrails & Triage)
# ==========================================
SYSTEM_INSTRUCTION = f"""
Eres el 'Agente de Triage Inteligente' de la red de clínicas SaludPlus Perú.
Tu misión es clasificar la consulta del paciente y decidir si la respondes tú (administrativa) o si la derivas a un humano (urgencia médica o queja compleja).

REGLAS ESTRICTAS (GUARDRAILS):
1. NO ERES MÉDICO. NUNCA des diagnósticos, ni recetes medicamentos.
2. Si el paciente menciona dolor fuerte, sangrado, accidentes, fiebre altísima o síntomas graves, es una URGENCIA.
3. Si el paciente pide hablar con un humano, está muy enojado, o hace una pregunta compleja que no está en la base de datos, DERÍVALO.
4. Para consultas administrativas, usa ESTA base de datos exclusivamente: {json.dumps(BASE_CONOCIMIENTO, ensure_ascii=False)}

FORMATO DE SALIDA OBLIGATORIO:
Debes responder SIEMPRE y ÚNICAMENTE con un objeto JSON válido. No incluyas texto fuera del JSON.
Estructura del JSON:
{{
    "accion": "RESPONDER" o "DERIVAR",
    "nivel_riesgo_clinico": "BAJO", "MEDIO", o "ALTO",
    "respuesta_al_usuario": "Tu respuesta empática y cordial aquí (solo si la acción es RESPONDER). Si es DERIVAR, indícale brevemente que lo transferirás.",
    "resumen_para_humano": "Resumen técnico de 1 línea del problema (solo si la acción es DERIVAR. Si es RESPONDER, déjalo vacío.)"
}}
"""

# ==========================================
# 3. CONFIGURACIÓN DE LA INTERFAZ (Streamlit)
# ==========================================
st.set_page_config(page_title="SaludPlus - IA Triage", page_icon="🏥", layout="centered")

st.title("🏥 SaludPlus Perú - Asistente IA")
st.markdown("Este agente clasifica tu consulta y decide si responder o escalar a un humano.")

# Configuración de API Key en la barra lateral
api_key = st.sidebar.text_input("Ingresa tu API Key de Gemini:", type="password")
if not api_key:
    st.warning("👈 Por favor, ingresa tu API Key de Google Gemini en la barra lateral para comenzar.")
    st.stop()

# Configurar el modelo de Google Gemini
genai.configure(api_key=api_key)
# Usamos un modelo reciente y le inyectamos las instrucciones del sistema
# NOTA: Los nombres de modelo cambian con el tiempo. Si falla, verifica los disponibles en:
# https://ai.google.dev/gemini-api/docs/models/gemini
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_INSTRUCTION,
    generation_config={"response_mime_type": "application/json"} # Forzamos salida en JSON
)

# Inicializar el historial de chat en la sesión
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mostrar el historial de mensajes
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Si el mensaje fue derivado, mostramos una alerta visual
        if msg.get("derivado"):
            st.error(f"🚨 **CASO DERIVADO A HUMANO**\n\n**Riesgo:** {msg['riesgo']}\n\n**Resumen técnico:** {msg['resumen_tecnico']}")

# ==========================================
# 4. LÓGICA DE PROCESAMIENTO (El Chat)
# ==========================================
pregunta = st.chat_input("Escribe tu consulta médica o administrativa...")

if pregunta:
    # 1. Mostrar la pregunta del usuario
    st.session_state.chat_history.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.write(pregunta)

    # 2. Procesar con Gemini
    with st.chat_message("assistant"):
        with st.spinner("Analizando urgencia e intención..."):
            try:
                # Enviamos la pregunta al modelo
                response = model.generate_content(pregunta)

                # Parseamos la respuesta JSON del modelo
                datos_ia = json.loads(response.text)

                accion = datos_ia.get("accion", "DERIVAR")
                riesgo = datos_ia.get("nivel_riesgo_clinico", "ALTO")
                texto_respuesta = datos_ia.get("respuesta_al_usuario", "Lo siento, necesito transferirte con un humano.")
                resumen_tecnico = datos_ia.get("resumen_para_humano", "")

                # Imprimir la respuesta al usuario
                st.write(texto_respuesta)

                # Si el modelo decide derivar, mostramos la interfaz del "Human in the Loop"
                es_derivado = accion == "DERIVAR"
                if es_derivado:
                    st.error(f"🚨 **ALERTA DE SISTEMA: TRANSFERENCIA A ASESOR HUMANO/MÉDICO**\n\n**Riesgo Clínico:** {riesgo}\n\n**Nota para el asesor:** {resumen_tecnico}")

                # Guardar en el historial
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": texto_respuesta,
                    "derivado": es_derivado,
                    "riesgo": riesgo,
                    "resumen_tecnico": resumen_tecnico
                })

            except json.JSONDecodeError as e:
                st.error(f"⚠️ La IA no respondió en formato JSON válido. Se deriva automáticamente por seguridad. Detalle: {e}")
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg or "not found" in error_msg.lower():
                    st.error(f"⚠️ El modelo de IA no está disponible o el nombre cambió. Detalle: {e}")
                elif "API key" in error_msg or "permission" in error_msg.lower():
                    st.error(f"⚠️ Problema con tu API Key. Verifica que sea válida y tenga permisos. Detalle: {e}")
                else:
                    st.error(f"⚠️ Error de procesamiento. Se deriva automáticamente por seguridad. Detalle: {e}")