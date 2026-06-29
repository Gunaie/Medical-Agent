# app.py (或你的 gradio 前端文件)
import gradio as gr
import requests
import uuid
import json
from app.config.settings import settings

STREAM_API_URL = f"http://127.0.0.1:{settings.API_PORT}/v1/chat/stream"


def generate_new_session_id():
    """生成新的 UUID 会话ID"""
    return str(uuid.uuid4())


def stream_chat_with_backend(message, history, session_id):
    """兼容当前 Gradio 版本的流式生成器（含状态提示自动清除）"""
    if not message.strip():
        yield history, session_id
        return

    if not session_id:
        session_id = generate_new_session_id()

    # ✅ 使用字典格式，满足 Gradio 运行时的 postprocess 校验
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    payload = {"session_id": session_id, "query": message}
    headers = {"Content-Type": "application/json"}

    try:
        with requests.post(STREAM_API_URL, json=payload, headers=headers, stream=True, timeout=120) as response:
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                try:
                    parsed = json.loads(data_str)
                    event = parsed.get("event")
                    content = parsed.get("content", "")

                    if event == "message":
                        # ✅ 【体验优化】收到正式回复时，自动清除斜体状态提示
                        current_content = history[-1]["content"]
                        if current_content.startswith("_") and current_content.endswith("_"):
                            history[-1]["content"] = ""

                        history[-1]["content"] += content
                        yield history, session_id

                    elif event == "source":
                        history[-1]["content"] += f"\n\n📚 参考来源：{content}"
                        yield history, session_id

                    elif event == "status":
                        # 仅在内容为空时显示状态提示，避免覆盖已有正文
                        if not history[-1]["content"]:
                            history[-1]["content"] = f"_{content}_"
                            yield history, session_id

                    elif event == "error":
                        history[-1]["content"] = f"❌ 流式响应异常: {content}"
                        yield history, session_id
                        break

                except json.JSONDecodeError:
                    continue

    except requests.exceptions.ConnectionError:
        history[-1]["content"] = f"❌ 无法连接到后端服务 ({STREAM_API_URL})"
        yield history, session_id
    except Exception as e:
        history[-1]["content"] = f"❌ API 调用失败: {str(e)}"
        yield history, session_id


# ================= UI 界面构建 =================
with gr.Blocks(title="🏥 MedAgent - 医疗智能问答系统") as demo:
    gr.Markdown("# 🏥 MedAgent - 企业级医疗智能问答系统")
    with gr.Row():
        session_id_box = gr.Textbox(label="Session ID", placeholder="留空则自动生成新会话")
        new_session_btn = gr.Button("🔄 开启新会话")

    # ✅ 不传 type 参数以兼容初始化，数据使用字典格式
    chatbot = gr.Chatbot(height=500)
    msg_input = gr.Textbox(label="请输入您的医疗问题", placeholder="例如：感冒了吃什么药？")
    clear_btn = gr.ClearButton([msg_input, chatbot])

    def submit_message(message, history, session_id):
        """流式输出过程中持续 yield 更新界面"""
        for updated_history, updated_sid in stream_chat_with_backend(message, history, session_id):
            yield "", updated_history, updated_sid

    msg_input.submit(
        fn=submit_message,
        inputs=[msg_input, chatbot, session_id_box],
        outputs=[msg_input, chatbot, session_id_box]
    )
    new_session_btn.click(generate_new_session_id, None, session_id_box)


if __name__ == "__main__":
    print(f"\n🔗 Gradio 前端将请求流式后端地址: {STREAM_API_URL}")
    demo.launch(server_port=settings.GRADIO_PORT)