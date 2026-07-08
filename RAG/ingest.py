"""
ingest.py
==========================================================
FASE 1: Generación de Datos e Ingesta a ChromaDB

Este script hace 4 cosas:
  1. Genera un JSON ficticio pero realista de la red de clínicas
     SaludPlus Perú.
  2. Convierte ese JSON en documentos de LangChain.
  3. Divide los documentos en chunks usando RecursiveCharacterTextSplitter.
  4. Genera embeddings locales y persiste la base vectorial en ./chroma_db.

Ejecutar:
    python ingest.py

Requiere:
    - Las dependencias de requirements.txt instaladas.
    - config.py en la misma carpeta.
"""

import json
import os
from datetime import datetime

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from config import ensure_project_dirs, get_embeddings, get_settings


# ============================================================
# 1. GENERADOR DE DATOS DE SALUDPLUS PERÚ
# ============================================================
def generate_saludplus_data() -> dict:
    """
    Crea un diccionario con datos estructurados de la red de clínicas.
    En un proyecto real estos datos vendrían de una base de datos SQL o API.
    """
    return {
        "empresa": {
            "nombre": "SaludPlus Perú",
            "razon_social": "SaludPlus S.A.C.",
            "ruc": "20548796321",
            "mision": "Brindar atención médica accesible, oportuna y de calidad a familias peruanas.",
        },
        "sedes": [
            {
                "nombre": "Lima Centro",
                "direccion": "Av. Arequipa 1234, Cercado de Lima",
                "telefono": "01-612-3456",
                "horario_atencion": "Lunes a Sábado de 8:00 AM a 8:00 PM. Emergencias 24/7.",
                "servicios": ["Medicina General", "Pediatría", "Cardiología", "Ginecología"],
            },
            {
                "nombre": "Lima Norte",
                "direccion": "Av. Tomas Valle 4321, Independencia",
                "telefono": "01-623-4567",
                "horario_atencion": "Lunes a Sábado de 7:00 AM a 9:00 PM. Domingos de 8:00 AM a 2:00 PM.",
                "servicios": ["Medicina General", "Pediatría", "Dermatología", "Nutrición"],
            },
            {
                "nombre": "Lima Sur",
                "direccion": "Av. Javier Prado Sur 987, Surco",
                "telefono": "01-634-5678",
                "horario_atencion": "Lunes a Sábado de 8:00 AM a 8:00 PM. Emergencias 24/7.",
                "servicios": ["Medicina General", "Cardiología", "Traumatología", "Neumología"],
            },
            {
                "nombre": "Arequipa Norte",
                "direccion": "Av. Ejército 567, Cayma",
                "telefono": "054-123-456",
                "horario_atencion": "Lunes a Sábado de 8:00 AM a 7:00 PM.",
                "servicios": ["Medicina General", "Pediatría", "Ginecología"],
            },
            {
                "nombre": "Trujillo Centro",
                "direccion": "Jr. Pizarro 345, Trujillo",
                "telefono": "044-987-654",
                "horario_atencion": "Lunes a Sábado de 8:00 AM a 8:00 PM. Domingos de 9:00 AM a 1:00 PM.",
                "servicios": ["Medicina General", "Cardiología", "Dermatología"],
            },
            {
                "nombre": "Cusco Valle",
                "direccion": "Av. La Cultura 789, Cusco",
                "telefono": "084-456-789",
                "horario_atencion": "Lunes a Sábado de 8:00 AM a 6:00 PM.",
                "servicios": ["Medicina General", "Pediatría", "Medicina Interna"],
            },
        ],
        "seguros_aceptados": [
            {
                "nombre": "EPS Rimac",
                "tipo": "EPS",
                "notas": "Se acepta en todas las sedes. Paciente debe presentar DNI y carnet vigente.",
            },
            {
                "nombre": "EPS Pacífico",
                "tipo": "EPS",
                "notas": "Se acepta en todas las sedes. Requierte orden médica para especialidades.",
            },
            {
                "nombre": "EsSalud",
                "tipo": "Seguro social",
                "notas": "Solo atención con referido de EsSalud. No se atiende sin documento de derivación.",
            },
            {
                "nombre": "Mapfre",
                "tipo": "Seguro privado",
                "notas": "Cobertura sujeta a cartilla. Coordinar previamente con el área de seguros.",
            },
            {
                "nombre": "Particular",
                "tipo": "Pago directo",
                "notas": "Se acepta pago en efectivo, tarjeta o Yape/Plin.",
            },
        ],
        "procesos": [
            {
                "titulo": "Agendamiento de citas",
                "descripcion": "Las citas se agendan mediante la App SaludPlus, el portal web o presencialmente con DNI. "
                               "Para especialidades se requiere orden médica o derivación según el seguro.",
            },
            {
                "titulo": "Cancelación o reprogramación",
                "descripcion": "Puedes cancelar o reprogramar hasta 2 horas antes de tu cita sin costo. "
                               "Después de ese plazo se cobra una penalidad del 30%.",
            },
            {
                "titulo": "Atención de emergencias",
                "descripcion": "Las sedes con emergencias 24/7 son Lima Centro y Lima Sur. "
                               "Para emergencias médicas graves se recomienda llamar al 105 o acudir al hospital más cercano.",
            },
        ],
        "reglas_triage": [
            {
                "titulo": "Urgencias médicas",
                "descripcion": "Síntomas como dolor intenso en el pecho, sangrado, convulsiones, pérdida de conocimiento, "
                               "dificultad para respirar o fiebre muy alta deben derivarse inmediatamente a un médico humano.",
            },
            {
                "titulo": "Consultas administrativas",
                "descripcion": "Horarios, seguros aceptados, especialidades, direcciones y procesos de citas pueden ser respondidas "
                               "por el asistente virtual usando la base de conocimiento.",
            },
            {
                "titulo": "Diagnósticos y recetas",
                "descripcion": "El asistente NUNCA debe diagnosticar, recetar medicamentos o recomendar tratamientos. "
                               "Estas solicitudes deben derivarse a un profesional de salud.",
            },
        ],
        "preguntas_frecuentes": [
            {
                "pregunta": "¿Qué documentos debo llevar a mi primera cita?",
                "respuesta": "DNI, carnet del seguro (si aplica) y orden médica o referido según la especialidad.",
            },
            {
                "pregunta": "¿Puedo pagar con tarjeta?",
                "respuesta": "Sí, en todas las sedes se acepta pago con tarjeta, efectivo, Yape y Plin.",
            },
            {
                "pregunta": "¿Tienen teleconsulta?",
                "respuesta": "Sí, disponible para Medicina General y Nutrición en horario de lunes a sábado de 9:00 AM a 6:00 PM.",
            },
        ],
    }


# ============================================================
# 2. CONVERTIR JSON A DOCUMENTOS DE LANGCHAIN
# ============================================================
def flatten_records(data: dict) -> list[Document]:
    """
    Transforma el diccionario anidado en una lista de Documentos planos.
    Cada Documento tiene:
      - page_content: texto legible para el modelo.
      - metadata: campos estructurados para filtrar o debuggear.
    """
    docs: list[Document] = []

    # --- Empresa ---
    empresa = data["empresa"]
    docs.append(
        Document(
            page_content=f"Empresa: {empresa['nombre']}\n"
                         f"Razón social: {empresa['razon_social']}\n"
                         f"RUC: {empresa['ruc']}\n"
                         f"Misión: {empresa['mision']}",
            metadata={"category": "empresa", "source": "empresa"},
        )
    )

    # --- Sedes ---
    for sede in data["sedes"]:
        servicios = ", ".join(sede["servicios"])
        docs.append(
            Document(
                page_content=f"Sede: {sede['nombre']}\n"
                             f"Dirección: {sede['direccion']}\n"
                             f"Teléfono: {sede['telefono']}\n"
                             f"Horario: {sede['horario_atencion']}\n"
                             f"Especialidades disponibles: {servicios}",
                metadata={
                    "category": "sede",
                    "source": f"sede_{sede['nombre']}",
                    "sede": sede["nombre"],
                },
            )
        )

    # --- Seguros ---
    for seguro in data["seguros_aceptados"]:
        docs.append(
            Document(
                page_content=f"Seguro aceptado: {seguro['nombre']}\n"
                             f"Tipo: {seguro['tipo']}\n"
                             f"Notas: {seguro['notas']}",
                metadata={
                    "category": "seguro",
                    "source": f"seguro_{seguro['nombre']}",
                },
            )
        )

    # --- Procesos ---
    for proceso in data["procesos"]:
        docs.append(
            Document(
                page_content=f"Proceso: {proceso['titulo']}\n"
                             f"Descripción: {proceso['descripcion']}",
                metadata={
                    "category": "proceso",
                    "source": f"proceso_{proceso['titulo']}",
                },
            )
        )

    # --- Reglas de triage ---
    for regla in data["reglas_triage"]:
        docs.append(
            Document(
                page_content=f"Regla de triage: {regla['titulo']}\n"
                             f"Descripción: {regla['descripcion']}",
                metadata={
                    "category": "regla_triage",
                    "source": f"regla_{regla['titulo']}",
                },
            )
        )

    # --- Preguntas frecuentes ---
    for faq in data["preguntas_frecuentes"]:
        docs.append(
            Document(
                page_content=f"Pregunta frecuente: {faq['pregunta']}\n"
                             f"Respuesta: {faq['respuesta']}",
                metadata={
                    "category": "faq",
                    "source": "faq",
                },
            )
        )

    return docs


# ============================================================
# 3. DIVISIÓN EN CHUNKS Y CONSTRUCCIÓN DEL VECTOR STORE
# ============================================================
def build_vector_store(
    docs: list[Document],
    embeddings,
    persist_dir: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> Chroma:
    """
    Divide los documentos en chunks y los indexa en ChromaDB.

    Args:
        docs: Documentos originales de LangChain.
        embeddings: Modelo de embeddings (HuggingFace o Google).
        persist_dir: Carpeta donde se persistirá ChromaDB.
        chunk_size: Tamaño máximo de cada chunk (en caracteres).
        chunk_overlap: Superposición entre chunks consecutivos.

    Returns:
        Instancia de Chroma con la colección indexada.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(docs)

    print(f"   [DOC] Documentos originales : {len(docs)}")
    print(f"   [CHK] Chunks generados      : {len(chunks)}")

    # Creamos (o sobrescribimos) la colección en ChromaDB.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="saludplus",
    )

    return vector_store


# ============================================================
# 4. FUNCIÓN PRINCIPAL
# ============================================================
def main() -> None:
    """Orquesta la generación de datos y la ingestas a ChromaDB."""
    print("[INFO] Iniciando ingestas de SaludPlus Perú...")

    settings = get_settings()
    ensure_project_dirs(settings)

    # 1. Generar datos
    data = generate_saludplus_data()

    # 2. Guardar JSON (para inspección del estudiante)
    with open(settings.data_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Datos guardados en: {settings.data_json_path}")

    # 3. Crear embeddings
    embeddings = get_embeddings(settings)
    print(f"[EMB] Modelo de embeddings: {settings.embedding_provider}")

    # 4. Flatten y chunks
    docs = flatten_records(data)

    # 5. Vector store
    vector_store = build_vector_store(
        docs=docs,
        embeddings=embeddings,
        persist_dir=settings.chroma_persist_dir,
    )

    # 6. Resumen final
    print("\n[OK] Ingestas completada exitosamente.")
    print(f"   Base vectorial persistida en: {settings.chroma_persist_dir}")
    print(f"   Fecha de generación         : {datetime.now().isoformat()}")

    # Pequeña prueba de retrieval para demostrar que funciona
    print("\n[SRC] Prueba de búsqueda: 'horarios de atención'")
    results = vector_store.similarity_search("horarios de atención", k=2)
    for i, doc in enumerate(results, 1):
        print(f"\n--- Resultado {i} ---")
        print(doc.page_content[:250] + "...")


if __name__ == "__main__":
    main()
