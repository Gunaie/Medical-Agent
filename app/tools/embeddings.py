# ⚠️ 废弃文件：此文件已弃用，请使用 app.tools.llm_tool.get_embeddings()

import warnings
from app.tools.llm_tool import get_embeddings

warnings.warn(
    "app.tools.embeddings is deprecated. Use app.tools.llm_tool.get_embeddings() instead.",
    DeprecationWarning,
    stacklevel=2
)

def get_embedding_function():
    """已废弃，请使用 app.tools.llm_tool.get_embeddings()"""
    return get_embeddings()