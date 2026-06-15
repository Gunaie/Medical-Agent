from app.tools.kg_tool import knowledge_graph_search

# StructuredTool 需要使用 invoke 方法，参数以字典形式传入
result = knowledge_graph_search.invoke({"entity": "乳泣", "relation": "症状"})
print(result)