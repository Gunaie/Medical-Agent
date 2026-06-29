from neo4j import GraphDatabase
from app.config.settings import settings


class Neo4jClient:
    """Neo4j 客户端（单例模式 + 延迟初始化）

    使用方式：
        # 首次调用时自动连接
        neo4j_client.run_query("MATCH (n) RETURN n LIMIT 1")

        # 应用关闭时释放连接
        neo4j_client.close()

    注意：
        不要在模块导入时调用 run_query()，应在应用启动后（lifespan 中）使用。
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        # 单例模式：确保只有一个实例
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 防止重复初始化
        if Neo4jClient._initialized:
            return
        self._driver = None
        Neo4jClient._initialized = True

    def connect(self):
        """延迟连接，避免模块导入时初始化失败

        首次调用 run_query() 时自动触发连接。
        如果连接已存在，直接返回现有连接。
        """
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
                )
                # 验证连接可用性
                self._driver.verify_connectivity()
                print("✅ Neo4j 连接成功")
            except Exception as e:
                print(f"❌ Neo4j 连接失败: {e}")
                raise
        return self._driver

    def run_query(self, cypher: str, params: dict = None):
        """执行 Cypher 查询并返回结果

        Args:
            cypher: Cypher 查询语句
            params: 查询参数（防止注入）

        Returns:
            List[dict]: 查询结果列表
        """
        driver = self.connect()
        with driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def close(self):
        """关闭连接，释放资源

        建议在应用关闭时调用（如 FastAPI lifespan 的 yield 之后）。
        """
        if self._driver:
            self._driver.close()
            self._driver = None
            print("🛑 Neo4j 连接已关闭")

    @property
    def is_connected(self) -> bool:
        """检查当前是否已连接"""
        return self._driver is not None


# 全局单例（延迟初始化，导入时不连接）
neo4j_client = Neo4jClient()