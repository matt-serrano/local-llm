# app_gui.py
import os
import gradio as gr
from llama_cpp import Llama

MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Load once at startup
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=max(2, os.cpu_count() // 2),
    verbose=False,
)

DEFAULT_SYSTEM = (
    "You are a concise, helpful local assistant. "
    "Be direct, avoid fluff, and keep code examples minimal."
)

def build_messages(history, user, system_prompt):
    msgs = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in history:
        if user_msg:
            msgs.append({"role": "user", "content": user_msg})
        if assistant_msg:
            msgs.append({"role": "assistant", "content": assistant_msg})
    msgs.append({"role": "user", "content": user})
    return msgs

def chat_fn(user, history, temperature, max_tokens, system_prompt):
    messages = build_messages(history, user, system_prompt)
    stream = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        stream=True,
    )
    partial = ""
    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            partial += delta
            yield partial  # stream to UI as it generates

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Local LLM (Offline)",
    description="Runs fully on your machine using llama.cpp",
    additional_inputs=[
        gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(32, 1024, value=256, step=32, label="Max tokens"),
        gr.Textbox(value=DEFAULT_SYSTEM, lines=3, label="System prompt"),
    ],
)

if __name__ == "__main__":
    demo.launch()
