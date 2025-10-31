import itertools
import sys
import threading
import time
from typing import List
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL = "openai/gpt-oss-20b"

class Spinner:
    def __init__(self, message: str = "Processing..."):
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])
        self.busy = False
        self.delay = 0.1
        self.message = message
        self.thread: threading.Thread | None = None

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def _spin(self) -> None:
        while self.busy:
            self.write(f"\r{self.message} {next(self.spinner)}")
            time.sleep(self.delay)
        self.write("\r\033[K")

    def __enter__(self):
        self.busy = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.busy = False
        time.sleep(self.delay)
        if self.thread:
            self.thread.join()
        self.write("\r")

def chat_loop() -> None:
    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. Keep responses concise unless asked "
                "for more detail."
            ),
        }
    ]

    print(
        "Assistant: "
        "Hi! I'm ready to chat. Ask me anything, or say 'quit' to exit."
    )

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        try:
            print("\nAssistant:", end=" ", flush=True)
            with Spinner("Thinking..."):
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    stream=True,
                )

            collected = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content = delta.content
                    print(content, end="", flush=True)
                    collected += content
            print()

            messages.append({"role": "assistant", "content": collected})

        except Exception as e:
            print(
                f"\nError chatting with the LM Studio server!\n\n"
                f"Please ensure:\n"
                f"1. LM Studio server is running at 127.0.0.1:1234 (hostname:port)\n"
                f"2. Model '{MODEL}' is downloaded\n"
                f"3. Model '{MODEL}' is loaded, or that just-in-time model loading is enabled\n\n"
                f"Error details: {str(e)}\n"
                "See https://lmstudio.ai/docs/basics/server for more information"
            )
            break

if __name__ == "__main__":
    chat_loop()