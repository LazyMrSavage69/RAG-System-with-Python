"""
chat.py — Interactive conversational interface for the RAG pipeline.

Run: python chat.py
"""

from rag import RAGPipeline


def main():
    print("=" * 60)
    print("  RAG Chat — powered by DeepSeek + ChromaDB")
    print("  Commands:  /ingest <path_or_url>  |  /status  |  /quit")
    print("=" * 60)

    pipeline = RAGPipeline()
    pipeline.status()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/quit", "/exit", "exit", "quit"}:
            print("Bye!")
            break

        elif user_input.lower().startswith("/ingest "):
            source = user_input[8:].strip()
            pipeline.ingest(source)

        elif user_input.lower() == "/status":
            pipeline.status()

        else:
            answer = pipeline.ask(user_input, verbose=False)
            print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()
