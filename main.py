import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# 显式导入路由模块
from app.api.routes import router as api_router
from app.config.settings import settings  # ✅ P1-3 新增：导入配置

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 MedAgent Enterprise 正在启动...")
    print(f"   🌍 环境: {settings.ENV}")

    # ✅ P1-6 预留：启动时验证 Neo4j 连接（取消注释即可启用）
    # from app.tools.neo4j_tool import neo4j_client
    # neo4j_client.connect()

    # 启动时打印已注册的路由，方便核对
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            print(f"   📍 已注册路由: {list(route.methods)} {route.path}")
    yield

    # ✅ P1-6 预留：关闭时释放 Neo4j 连接（取消注释即可启用）
    # neo4j_client.close()

    print("🛑 MedAgent Enterprise 正在关闭...")

app = FastAPI(
    title="MedAgent - 企业级医疗问答",
    version="0.1.0",
    description="基于 AI 的医疗智能体系统",
    lifespan=lifespan
)

# ✅ P1-3 修复：CORS 白名单从配置读取，不再允许所有来源
cors_origins = settings.CORS_ORIGINS.split(",") if "," in settings.CORS_ORIGINS else [settings.CORS_ORIGINS]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # 从 settings.CORS_ORIGINS 读取，生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# prefix="" 表示根路径直接可用
app.include_router(api_router, prefix="")

if __name__ == "__main__":
    # ✅ P1-3 修复：参数从配置读取，生产环境自动关闭 reload
    reload_flag = settings.ENV == "development"
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,    # 从配置读取，默认 0.0.0.0
        port=settings.API_PORT,    # 从配置读取，默认 8080
        reload=reload_flag         # development 时 True，production 时 False
    )