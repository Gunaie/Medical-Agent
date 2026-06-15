from app.tools.llm_tool import get_llm, get_embeddings

if __name__ == "__main__":
    print("🔍 测试 Qwen-Plus...")
    llm = get_llm()
    res = llm.invoke("请用一句话介绍阿司匹林")
    print(f"✅ LLM 响应: {res.content}\n")

    print("🔍 测试 text-embedding-v4...")
    emb = get_embeddings()
    vec = emb.embed_query("高血压")
    print(f"✅ Embedding 维度: {len(vec)}")