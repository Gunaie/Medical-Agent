from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os
from pathlib import Path


# ✅ 获取 app/ 的上一级，即项目根目录的绝对路径
# settings.py 在 app/config/ 下，parents[2] 就是项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    """医疗智能问答系统配置管理"""

    # --- 模型配置 ---
    LLM_MODEL: str = "qwen-plus"
    EMBEDDING_MODEL: str = "text-embedding-v4"
    DASHSCOPE_API_KEY: str = Field(..., validation_alias="DASHSCOPE_API_KEY")

    # --- Neo4j 图数据库配置 (规范要求) ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password123"

    # 无论从哪里运行脚本，都固定指向根目录下的 chroma_db
    CHROMA_PERSIST_DIR: str = os.path.join(BASE_DIR, "chroma_db")

    # --- 服务端口与地址 ---
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8080
    GRADIO_PORT: int = 7860

    # Pydantic V2 标准配置方式
    model_config = SettingsConfigDict(
        env_file=".env",  # 默认读取当前项目根目录下的 .env
        env_file_encoding="utf-8",
        case_sensitive=False  # 允许环境变量大小写不敏感
    )


# 实例化全局配置对象
settings = Settings()