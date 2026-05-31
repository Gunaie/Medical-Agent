import gradio as gr
from service import chat

EXAMPLES = [
    ["鼻炎有什么症状？"],
    ["阿莫西林的禁忌是什么？"],
    ["寻医问药网怎么联系医生？"],
    ["你好，你是谁？"],
]

with gr.Blocks(title="医疗问诊机器人") as demo:
    gr.Markdown("# 🏥 医疗问诊机器人\n基于 Qwen-Plus + 知识图谱 + RAG")
    chatbot = gr.ChatInterface(
        fn=chat,
        examples=EXAMPLES,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )