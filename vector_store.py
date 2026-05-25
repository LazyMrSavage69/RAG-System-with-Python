"""
vector_store.py — ChromaDB vector store with sentence-transformers embeddings.

The embedding model runs locally (no API key needed).
Default model: all-MiniLM-L6-v2  (fast, ~80 MB, strong multilingual quality)
"""

import hashlib
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ── Embedding helper ──────────────────────────────────────────────────────────

class LocalEmbedder:
    """Thin wrapper so we can swap models easily."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[Embedder] Loading '{model_name}' …")
        self.model = SentenceTransformer(model_name)
        print("[Embedder] Ready.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()


# ── Vector store ──────────────────────────────────────────────────────────────

class VectorStore:
    """
    Persistent ChromaDB collection with local sentence-transformer embeddings.

    Args:
        persist_dir:    Directory where ChromaDB stores its files.
        collection_name: Name of the ChromaDB collection.
        embed_model:    Sentence-transformers model name.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_collection",
        embed_model: str = "all-MiniLM-L6-v2",
    ):
        self.embedder = LocalEmbedder(embed_model)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        print(f"[VectorStore] Collection '{collection_name}' ready "
              f"({self.collection.count()} docs).")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_id(chunk: Dict) -> str:
        """Stable, deterministic ID from source + chunk index."""
        raw = f"{chunk['source']}::{chunk['chunk_index']}"
        return hashlib.md5(raw.encode()).hexdigest()

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Embed and store a list of chunks.  Skips duplicates (same ID).

        Returns:
            Number of newly added chunks.
        """
        if not chunks:
            return 0

        ids = [self._make_id(c) for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]}
                     for c in chunks]

        # Avoid re-inserting chunks that are already stored
        existing = set(self.collection.get(ids=ids)["ids"])
        new_mask = [i for i, id_ in enumerate(ids) if id_ not in existing]

        if not new_mask:
            print("[VectorStore] All chunks already indexed — skipping.")
            return 0

        new_ids = [ids[i] for i in new_mask]
        new_texts = [texts[i] for i in new_mask]
        new_meta = [metadatas[i] for i in new_mask]

        print(f"[VectorStore] Embedding {len(new_texts)} chunks …")
        embeddings = self.embedder.embed(new_texts)

        self.collection.add(
            ids=new_ids,
            documents=new_texts,
            embeddings=embeddings,
            metadatas=new_meta,
        )
        print(f"[VectorStore] ✓ Stored {len(new_texts)} chunks.")
        return len(new_texts)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the top-k most relevant chunks for a question.

        Returns:
            List of dicts with keys: 'text', 'source', 'score'.
        """
        embedding = self.embedder.embed([question])[0]
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta["source"],
                "score": round(1 - dist, 4),   # convert cosine distance → similarity
            })

        return chunks

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Delete all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        print("[VectorStore] Collection reset.")
