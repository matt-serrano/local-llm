from llama_cpp import Llama
import os, sys

MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=max(2, os.cpu_count() // 2),
    verbose=False,
    # If you have Apple Silicon or CUDA builds later, you can add:
    # n_gpu_layers=-1,
)

SYSTEM_PROMPT = (
    "You are a concise, helpful local assistant. "
    "Be direct, avoid fluff, and keep code examples minimal."
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

def chat(user_text: str) -> str:
    messages.append({"role": "user", "content": user_text})
    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
    )
    reply = out["choices"][0]["message"]["content"].strip()
    messages.append({"role": "assistant", "content": reply})
    return reply

if __name__ == "__main__":
    print("Local Chat — type '/exit' to quit\n")
    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in {"/exit", "exit", "quit", ":q"}:
                print("Goodbye!")
                sys.exit(0)
            answer = chat(user)
            print(f"Assistant: {answer}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
