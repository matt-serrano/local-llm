from llama_cpp import Llama
import os

MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=max(2, os.cpu_count() // 2),
    verbose=False,
)

prompt = "You are a concise assistant. Reply in one sentence: What is an LLM?"

out = llm.create_completion(
    prompt=prompt,
    max_tokens=128,
    temperature=0.7,
    stop=["</s>"]
)

print(out["choices"][0]["text"].strip())
