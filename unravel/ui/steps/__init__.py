"""UI step exports, imported on demand."""

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "render_upload_step": "upload",
    "render_chunks_step": "chunks",
    "render_embeddings_step": "embeddings",
    "render_query_step": "query",
    "render_policy_step": "policy",
    "render_export_step": "export",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value
