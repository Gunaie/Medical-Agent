from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os
from pathlib import Path
from typing import Optional


# 获取 app/ 的上一级，即项目根目录的绝对路径
# settings.py 在 app/config/ 下，parents[2] 就是项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    # --- 模型配置 ---
    LLM_MODEL: str = "qwen-plus"
    EMBEDDING_MODEL: str = "text-embedding-v4"
    DASHSCOPE_API_KEY: str = Field(..., validation_alias="DASHSCOPE_API_KEY")

    # --- Neo4j 图数据库配置 ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = Field(..., validation_alias="NEO4J_PASSWORD")

    # --- ChromaDB ---
    CHROMA_PERSIST_DIR: str = os.path.join(BASE_DIR, "chroma_db")

    # --- 服务端口与地址 ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    GRADIO_PORT: int = 7860

    # --- CORS 白名单 ---
    CORS_ORIGINS: str = "*"

    # --- 环境标识 ---
    ENV: str = "development"

    # ✅ 【新增】Redis 配置
    REDIS_URL: Optional[str] = "redis://localhost:6379/0"

    # Pydantic V2 标准配置方式
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

settings = Settings()