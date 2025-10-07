from llama_cpp import Llama

MODEL_PATH = "models/llama-3.2-3b-instruct.Q$_K_M.gguf"

llm = Llama (
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=max(2, __import__("os").cpu_count() // 2),
    verbose=False,                  
)

prompt = "You are a conise assistant. Reply in one sentence: What is an LLM?"
out = llm.create_completion(
    prompt=prompt
    max_tokens=128
    temperature=0.7
    stop=["</s>]
)

print(out["choices"][0]["text"].strip())