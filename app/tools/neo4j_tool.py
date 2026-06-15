from neo4j import GraphDatabase
from app.config.settings import settings

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )

    def run_query(self, cypher: str, params: dict = None):
        """执行Cypher查询并返回结果"""
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def close(self):
        self.driver.close()

# 全局单例
neo4j_client = Neo4jClient()