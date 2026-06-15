import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# 显式导入路由模块
from app.api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 MedAgent Enterprise 正在启动...")
    # ✅ 启动时打印已注册的路由，方便核对
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            print(f"   📍 已注册路由: {list(route.methods)} {route.path}")
    yield
    print("🛑 MedAgent Enterprise 正在关闭...")

app = FastAPI(
    title="MedAgent - 企业级医疗问答",
    version="0.1.0",
    description="基于 AI 的医疗智能体系统",
    lifespan=lifespan
)

# ✅ 【新增】添加 CORS 中间件，避免跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# prefix="" 表示根路径直接可用
app.include_router(api_router, prefix="")

if __name__ == "__main__":
    # ✅ 端口固定为 8080，与前端 settings.py 必须一致
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)