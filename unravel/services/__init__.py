"""Public service exports, loaded only when requested.

Importing a policy service should not load the optional document playground.
"""

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "Chunk": "chunking",
    "get_chunks": "chunking",
    "get_available_providers": "chunking",
    "get_provider_splitters": "chunking",
    "EmbedderBackend": "embedding_backends",
    "get_backend": "embedding_backends",
    "Embedder": "embedders",
    "get_embedder": "embedders",
    "list_available_models": "embedders",
    "EMBEDDING_MODELS": "embedders",
    "DEFAULT_MODEL": "embedders",
    "VectorStore": "vector_store",
    "SearchResult": "vector_store",
    "create_vector_store": "vector_store",
    "LLMConfig": "llm",
    "RAGContext": "llm",
    "LLM_PROVIDERS": "llm",
    "DEFAULT_SYSTEM_PROMPT": "llm",
    "generate_response": "llm",
    "generate_response_stream": "llm",
    "get_api_key_from_env": "llm",
    "list_providers": "llm",
    "get_provider_models": "llm",
    "validate_config": "llm",
    "get_model": "llm",
    "ModelWrapper": "llm",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value
