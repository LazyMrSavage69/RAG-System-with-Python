"""
rag.py — Top-level RAG pipeline.

Orchestrates: document loading → embedding → retrieval → generation.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

from loader import load_document
from vector_store import VectorStore
from generator import DeepSeekGenerator


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest("my_doc.pdf")
        answer = pipeline.ask("What is the main topic?")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_collection",
        embed_model: str = "all-MiniLM-L6-v2",
        deepseek_model: str = "deepseek-chat",
        top_k: int = 5,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        load_dotenv()
        resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise ValueError(
                "DeepSeek API key not found. "
                "Set DEEPSEEK_API_KEY in your .env file or pass api_key=."
            )

        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embed_model=embed_model,
        )
        self.generator = DeepSeekGenerator(
            api_key=resolved_key,
            model=deepseek_model,
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, source: str) -> int:
        """
        Load, chunk, embed, and store a document.

        Args:
            source: File path (PDF/TXT/MD) or HTTP(S) URL.

        Returns:
            Number of newly indexed chunks.
        """
        print(f"\n[RAG] Ingesting: {source}")
        chunks = load_document(
            source,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        print(f"[RAG] {len(chunks)} chunks created.")
        added = self.store.add_chunks(chunks)
        return added

    def ingest_many(self, sources: List[str]) -> Dict[str, int]:
        """Ingest multiple documents. Returns {source: chunks_added}."""
        return {src: self.ingest(src) for src in sources}

    # ── Querying ──────────────────────────────────────────────────────────────

    def ask(self, question: str, verbose: bool = False) -> str:
        """
        Answer a question using retrieved context.

        Args:
            question: Natural language question.
            verbose:  If True, print retrieved chunks before answering.

        Returns:
            Generated answer string.
        """
        print(f"\n[RAG] Query: {question}")

        if self.store.count() == 0:
            return "⚠️  No documents indexed yet. Call ingest() first."

        chunks = self.store.query(question, top_k=self.top_k)

        if verbose:
            print(f"\n[RAG] Top-{len(chunks)} retrieved chunks:")
            for i, c in enumerate(chunks, 1):
                print(f"  [{i}] (score={c['score']}) {c['source']}")
                print(f"       {c['text'][:120]}…")

        answer = self.generator.generate(question, chunks)
        return answer

    # ── Utilities ─────────────────────────────────────────────────────────────

    def status(self):
        print(f"[RAG] Indexed chunks: {self.store.count()}")


# ── CLI / Quick demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pipeline = RAGPipeline()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rag.py ingest <file_or_url>")
        print("  python rag.py ask    '<question>'")
        print("  python rag.py status")
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Please provide a file path or URL.")
            sys.exit(1)
        pipeline.ingest(sys.argv[2])

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Please provide a question.")
            sys.exit(1)
        answer = pipeline.ask(sys.argv[2], verbose=True)
        print(f"\n{'='*60}\nAnswer:\n{answer}\n{'='*60}")

    elif command == "status":
        pipeline.status()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
