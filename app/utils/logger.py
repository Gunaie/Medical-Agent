import structlog
import logging
import os
from datetime import datetime
from pathlib import Path

# ✅ P1-1 修复：使用项目根目录的绝对路径，避免相对路径导致的日志位置不可预期
# logger.py 位于 app/utils/logger.py，向上两级即为项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent
log_dir = BASE_DIR / "logs"
os.makedirs(log_dir, exist_ok=True)

current_date = datetime.now().strftime("%Y%m%d")
log_file = log_dir / f"medagent_{current_date}.jsonl"


def setup_logging():
    # 配置标准库 logging 作为底层输出通道
    # structlog.stdlib.LoggerFactory() 依赖此配置将日志写入文件
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        format="%(message)s",  # JSONRenderer 已格式化，此处无需额外格式
        force=True,  # 覆盖 Uvicorn/其他库可能已设置的 root logger
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,      # 现在 logger 有 .name 属性了
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        # 关键修复：改用 stdlib 工厂，替代 WriteLoggerFactory
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


# 调用配置
setup_logging()
logger = structlog.get_logger("medical_agent")