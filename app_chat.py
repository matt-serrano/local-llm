from llama_cpp import Llama
import os, sys, readline  # readline = arrow-key history on Linux/macOS
from colorama import Fore, Style, init as colorama_init

MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=max(2, os.cpu_count() // 2),
    verbose=False,
)

SYSTEM_PROMPT = (
    "You are a concise, helpful local assistant. "
    "Be direct, avoid fluff, and keep code examples minimal."
)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
temperature = 0.7
max_tokens = 256

colorama_init(autoreset=True)

HELP_TEXT = f"""\
{Fore.CYAN}/help{Style.RESET_ALL}  show this help
{Fore.CYAN}/reset{Style.RESET_ALL} clear chat history (keeps system prompt)
{Fore.CYAN}/save <file>{Style.RESET_ALL} save the transcript to a file
{Fore.CYAN}/temp <0-2>{Style.RESET_ALL} set temperature (current: {{temp}})
{Fore.CYAN}/max <tokens>{Style.RESET_ALL} set max tokens (current: {{max_toks}})
{Fore.CYAN}/exit{Style.RESET_ALL}  quit
"""

def show_help():
    print(HELP_TEXT.format(temp=temperature, max_toks=max_tokens))

def stream_reply():
    # stream tokens as they arrive
    for chunk in llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        stream=True,  # generator of incremental chunks
    ):
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            yield delta

def save_transcript(path: str):
    with open(path, "w", encoding="utf-8") as f:
        for m in messages:
            role = m["role"].upper()
            f.write(f"{role}: {m['content']}\n\n")

def main():
    print(f"{Fore.GREEN}Local Chat — type /help for commands, /exit to quit{Style.RESET_ALL}\n")
    global temperature, max_tokens

    try:
        while True:
            try:
                user = input(f"{Fore.YELLOW}You:{Style.RESET_ALL} ").strip()
            except EOFError:
                print("\nGoodbye!")
                break

            if not user:
                continue

            # commands
            if user.lower() in {"/exit", "exit", "quit", ":q"}:
                print("Goodbye!")
                break
            if user.startswith("/help"):
                show_help(); continue
            if user.startswith("/reset"):
                del messages[1:]  # keep system prompt
                print(f"{Fore.MAGENTA}History cleared.{Style.RESET_ALL}\n")
                continue
            if user.startswith("/save"):
                parts = user.split(maxsplit=1)
                if len(parts) == 2:
                    path = parts[1].strip()
                    try:
                        save_transcript(path)
                        print(f"{Fore.MAGENTA}Saved to {path}.{Style.RESET_ALL}\n")
                    except Exception as e:
                        print(f"{Fore.RED}Save failed: {e}{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.RED}Usage: /save transcript.txt{Style.RESET_ALL}\n")
                continue
            if user.startswith("/temp"):
                parts = user.split()
                if len(parts) == 2:
                    try:
                        temperature = float(parts[1])
                        print(f"{Fore.MAGENTA}Temperature set to {temperature}.{Style.RESET_ALL}\n")
                    except ValueError:
                        print(f"{Fore.RED}Invalid temperature.{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.RED}Usage: /temp 0.7{Style.RESET_ALL}\n")
                continue
            if user.startswith("/max"):
                parts = user.split()
                if len(parts) == 2 and parts[1].isdigit():
                    max_tokens = int(parts[1])
                    print(f"{Fore.MAGENTA}Max tokens set to {max_tokens}.{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.RED}Usage: /max 256{Style.RESET_ALL}\n")
                continue

            # normal chat turn
            messages.append({"role": "user", "content": user})
            print(f"{Fore.CYAN}Assistant:{Style.RESET_ALL} ", end="", flush=True)

            reply_chunks = []
            try:
                for piece in stream_reply():
                    reply_chunks.append(piece)
                    print(piece, end="", flush=True)
                print("\n")
            except Exception as e:
                print(f"\n{Fore.RED}Generation failed: {e}{Style.RESET_ALL}\n")
                messages.pop()  # drop last user msg on failure
                continue

            reply_text = "".join(reply_chunks).strip()
            messages.append({"role": "assistant", "content": reply_text})

    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
