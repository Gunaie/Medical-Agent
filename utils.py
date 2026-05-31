from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from neo4j import GraphDatabase
import config

def get_llm(temperature=0.0):
    """获取 Qwen-Plus LLM 实例"""
    return ChatTongyi(
        model=config.LLM_MODEL,
        dashscope_api_key=config.DASHSCOPE_API_KEY,
        temperature=temperature,
        streaming=True
    )

def get_embeddings():
    """获取 text-embedding-v4 实例 (1024维)"""
    return DashScopeEmbeddings(
        model=config.EMBEDDING_MODEL,
        dashscope_api_key=config.DASHSCOPE_API_KEY
    )

def get_neo4j_driver():
    """获取 Neo4j 驱动单例"""
    if not hasattr(get_neo4j_driver, "_driver"):
        get_neo4j_driver._driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
    return get_neo4j_driver._driver

def query_neo4j(cypher: str, params: dict = None):
    """执行 Cypher 查询并返回记录列表"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [record.data() for record in result]