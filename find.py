from app.tools.embeddings import get_embedding_function

embedder = get_embedding_function()
# sentence-transformers 会将本地路径或缓存路径存储在 model_path 属性中
print("✅ 模型实际加载路径:", embedder.model[0].auto_model.config._name_or_path)