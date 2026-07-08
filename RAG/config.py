"""
config.py
==========================================================
Configuración central del laboratorio SaludPlus Perú.

Este módulo carga las variables de entorno desde un archivo .env
y expone dos fábricas (funciones creadoras):
  - get_embeddings()
  - get_llm()

De esta forma, si más adelante quieres cambiar de proveedor
(Gemini -> OpenAI, o embeddings locales -> Google), solo tocas
variables de entorno, no la lógica de negocio.
"""

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# 1. DEFINICIÓN DE VARIABLES DE ENTORNO
# ============================================================
class Settings(BaseSettings):
    """
    Pydantic-settings se encarga de:
      1. Leer el archivo .env (si existe).
      2. Convertir los valores a los tipos de Python.
      3. Proveer valores por defecto razonables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignora variables de entorno que no estén aquí
    )

    # Proveedor de LLM: "google" o "openai"
    llm_provider: Literal["google", "openai"] = "google"

    # Google Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # OpenAI (opcional)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Proveedor de embeddings: "local" o "google"
    # - local:  descarga un modelo de sentence-transformers (gratis, offline).
    # - google: usa GoogleGenerativeAIEmbeddings (consume cuota de API).
    embedding_provider: Literal["local", "google"] = "local"
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Rutas del proyecto
    chroma_persist_dir: str = "./chroma_db"
    data_json_path: str = "./data/saludplus_data.json"


# ============================================================
# 2. CARGA DE CONFIGURACIÓN
# ============================================================
def get_settings() -> Settings:
    """
    Retorna la configuración cargada desde el archivo .env y variables de entorno.

    NOTA: No usamos @lru_cache porque la API key puede cambiar en runtime
    (por ejemplo, cuando el usuario la ingresa en la sidebar de Streamlit).
    """
    return Settings()


# ============================================================
# 3. FÁBRICA DE EMBEDDINGS
# ============================================================
def get_embeddings(settings: Settings | None = None):
    """
    Crea y retorna el modelo de embeddings según la configuración.

    - EMBEDDING_PROVIDER=local  -> HuggingFaceEmbeddings (modelo descargado localmente)
    - EMBEDDING_PROVIDER=google -> GoogleGenerativeAIEmbeddings (API de Gemini)
    """
    if settings is None:
        settings = get_settings()

    if settings.embedding_provider == "local":
        # Embeddings locales: no consumen API, ideal para laboratorios.
        # El modelo se descarga automáticamente la primera vez.
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.local_embedding_model,
            model_kwargs={"device": "cpu"},  # Forzamos CPU para compatibilidad
            encode_kwargs={"normalize_embeddings": True},
        )

    if settings.embedding_provider == "google":
        # Embeddings vía API de Gemini.
        if not settings.google_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=google requiere la variable GOOGLE_API_KEY"
            )

        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.google_api_key,
        )

    raise ValueError(f"Proveedor de embeddings no soportado: {settings.embedding_provider}")


# ============================================================
# 4. FÁBRICA DE LLM
# ============================================================
def get_llm(settings: Settings | None = None):
    """
    Crea y retorna el modelo de lenguaje (LLM) según la configuración.

    - LLM_PROVIDER=google -> ChatGoogleGenerativeAI (Gemini)
    - LLM_PROVIDER=openai -> ChatOpenAI
    """
    if settings is None:
        settings = get_settings()

    if settings.llm_provider == "google":
        if not settings.google_api_key:
            raise ValueError(
                "LLM_PROVIDER=google requiere la variable GOOGLE_API_KEY"
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.2,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
        )

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai requiere la variable OPENAI_API_KEY"
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=0.2,
            api_key=settings.openai_api_key,
        )

    raise ValueError(f"Proveedor de LLM no soportado: {settings.llm_provider}")


# ============================================================
# 5. HELPER: CREAR DIRECTORIOS SI NO EXISTEN
# ============================================================
def ensure_project_dirs(settings: Settings | None = None) -> None:
    """Crea las carpetas data/ y chroma_db/ si aún no existen."""
    if settings is None:
        settings = get_settings()

    os.makedirs(os.path.dirname(settings.data_json_path) or ".", exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)


# Permite ejecutar: python config.py (solo para verificar carga)
if __name__ == "__main__":
    cfg = get_settings()
    print("Configuración cargada correctamente:")
    print(f"  LLM provider       : {cfg.llm_provider}")
    print(f"  Modelo Gemini      : {cfg.gemini_model}")
    print(f"  Embedding provider : {cfg.embedding_provider}")
    print(f"  Modelo local       : {cfg.local_embedding_model}")
    print(f"  Chroma persist dir : {cfg.chroma_persist_dir}")
    print(f"  Data JSON path     : {cfg.data_json_path}")
    print(f"  GOOGLE_API_KEY     : {'*** configurada ***' if cfg.google_api_key else '--- vacía ---'}")
