# scripts/ingest_medical_guides_fixed.py
import os
import sys
import glob
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ 【路径修复】动态获取项目根目录，替代硬编码绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.db.chroma_init import medical_collection

# ==================== 0. 指南元数据映射（按文件名关键词匹配） ====================
GUIDE_METADATA_MAP = {
    "儿童肺炎支原体肺炎": {"publish_year": 2025, "target_population": "儿童"},
    "肥胖症": {"publish_year": 2024, "target_population": "成人/儿童"},
    "唇裂": {"publish_year": 2024, "target_population": "新生儿/婴幼儿"},
}

# ==================== 1. 全局初始化 OCR 引擎 ====================
print("⏳ 正在加载 RapidOCR 模型...")
ocr_engine = RapidOCR()
print("✅ RapidOCR 模型加载完成")

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)


# ==================== 2. OCR 感知文本提取函数 ====================
def _extract_page_text(page: fitz.Page) -> str:
    """
    智能提取PDF页面文本：
    - 优先使用原生文本（速度快）
    - 当有效字符数低于阈值时，自动降级为 OCR 识别扫描页
    """
    native_text = page.get_text("text").strip()
    clean_len = len(native_text.replace('\n', '').replace(' ', ''))
    print(f"   [DEBUG] 第{page.number + 1}页 | 原始长度:{len(native_text)} | 有效字符:{clean_len}")

    # 阈值判断：有效字符极少时判定为扫描页
    if clean_len > 30:
        print(f"   [DEBUG] → 判定为文本页，直接返回")
        return native_text

    print(f"   [DEBUG] → 触发OCR降级，渲染DPI=300图片...")
    try:
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")

        result, elapse = ocr_engine(img_bytes)

        # ✅ 安全提取耗时，避免 rapidocr 返回 list 时触发 __format__ 异常
        if isinstance(elapse, (list, tuple)):
            elapsed_str = "/".join([f"{e:.3f}" for e in elapse])
        else:
            elapsed_str = f"{elapse:.3f}"

        ocr_lines = len(result) if result else 0
        print(f"   [DEBUG] → OCR耗时:{elapsed_str}s | 识别行数:{ocr_lines}")

        if not result:
            print(f"   [WARN] OCR返回空结果，回退原生文本")
            return native_text

        # 按阅读顺序排序：先从上到下(y)，再从左到右(x)
        sorted_lines = sorted(result, key=lambda x: (x[0][0][1], x[0][0][0]))
        ocr_text = "\n".join([line[1] for line in sorted_lines])
        print(f"   🔍 第{page.number + 1}页为扫描页，OCR识别 {len(sorted_lines)} 行")
        return ocr_text

    except Exception as e:
        print(f"   ❌ OCR异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return native_text


# ==================== 3. 主解析与入库函数 ====================
def parse_medical_pdf(pdf_path: str):
    """解析医疗指南PDF并写入向量数据库"""
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在: {pdf_path}")
        return

    filename = os.path.basename(pdf_path)
    print(f"\n📖 开始处理: {filename}")

    # 根据文件名匹配元数据，未匹配则使用默认值
    guide_meta = {"publish_year": 2024, "target_population": "通用"}
    for key, meta in GUIDE_METADATA_MAP.items():
        if key in filename:
            guide_meta = meta
            break

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    success_chunks = 0

    for page_num in range(total_pages):
        page = doc[page_num]
        print(f"\n--- 处理进度: {page_num + 1}/{total_pages} ---")

        text = _extract_page_text(page)

        if not text.strip():
            print(f"   ⚠️ 第{page_num + 1}页无有效内容，跳过")
            continue

        chunks = text_splitter.split_text(text)

        if chunks:
            metadatas = [{
                "domain": "medical_clinical_docs",
                "source_file": filename,
                "page": page_num + 1,
                "publish_year": guide_meta["publish_year"],
                "target_population": guide_meta["target_population"]
            }] * len(chunks)

            ids = [f"{filename}_p{page_num+1}_c{i}" for i in range(len(chunks))]

            medical_collection.upsert(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )

        success_chunks += len(chunks)
        print(f"   ✅ 第{page_num + 1}页生成 {len(chunks)} 个chunk")

    doc.close()
    print(f"\n🎉 处理完成！共处理 {total_pages} 页，成功入库 {success_chunks} 个chunk")


# ==================== 4. 脚本入口（自动遍历目录） ====================
if __name__ == "__main__":
    # ✅ 【核心修复】使用 BASE_DIR 动态拼接路径，不再硬编码
    GUIDES_DIR = os.path.join(BASE_DIR, "guides")

    if not os.path.exists(GUIDES_DIR):
        print(f"❌ 指南目录不存在: {GUIDES_DIR}")
    else:
        pdf_files = glob.glob(os.path.join(GUIDES_DIR, "*.pdf"))

        if not pdf_files:
            print(f"⚠️ 在 {GUIDES_DIR} 中未找到任何 PDF 文件")
        else:
            print(f"📂 发现 {len(pdf_files)} 个医疗指南文件:")
            for f in pdf_files:
                print(f"   - {os.path.basename(f)}")
            print("=" * 50)

            for pdf_path in pdf_files:
                parse_medical_pdf(pdf_path)

            print("\n" + "=" * 50)
            print("🎉 全部指南处理完毕！")