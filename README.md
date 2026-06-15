🏥 Medical AI Agent：基于知识图谱与 RAG 的医疗问诊助手
📝 项目简介
这是一个基于 LangGraph 和 FastAPI 构建的医疗领域 AI 助手。系统结合了知识图谱（KG）的结构化查询与 RAG 的非结构化检索能力，并针对“药物相互作用”等复杂问题设计了多级降级与推理机制。
✨ 核心亮点 (Highlights)
三级降级检索架构：实现了“KG 结构化查询 -> RAG 指南检索 -> LLM 预训练推理”的无缝降级链路。
高可用流式输出：基于 FastAPI SSE 与局部异常防御机制，彻底解决流式解析中断问题，实现丝滑输出。
医疗安全护栏：通过 Prompt Engineering 强制模型输出免责声明，从业务逻辑层保障医疗问答合规。
性能优化：采用 LRU 缓存策略优化 Agent 实例化开销，降低重复构建带来的性能损耗。
🛠️ 技术栈
后端框架：FastAPI, LangGraph (ReAct Agent)
数据存储：Neo4j (知识图谱), Milvus/Chroma (向量数据库)
AI 核心：LangChain, 预训练大模型 API
🚀 快速开始
1. 克隆项目
git clone https://github.com/你的用户名/medical-ai-agent.git
cd medical-ai-agent
2. 配置环境变量
cp .env.example .env
编辑 .env 文件，填入真实的 Neo4j 和 LLM 密钥
3. 安装依赖并启动
pip install -r requirements.txt
uvicorn app.main:app --reload
