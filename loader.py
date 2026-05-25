"""
loader.py — Document loading and chunking for multiple file types.
Supported: PDF, TXT, Markdown (.md), Web URLs
"""

import re
import fitz  # PyMuPDF
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict]:
    """
    Split text into overlapping chunks and attach metadata.

    Args:
        text:       Raw text to split.
        source:     Origin label (file path or URL) stored in metadata.
        chunk_size: Approximate number of words per chunk.
        overlap:    Number of words shared between consecutive chunks.

    Returns:
        List of dicts with keys: 'text', 'source', 'chunk_index'.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({
            "text": chunk,
            "source": source,
            "chunk_index": len(chunks),
        })
        start += chunk_size - overlap  # slide forward with overlap

    return chunks


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_pdf(path: str) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def load_text(path: str) -> str:
    """Read a plain text or Markdown file."""
    return Path(path).read_text(encoding="utf-8")


def load_url(url: str, timeout: int = 10) -> str:
    """
    Fetch a web page and extract its visible text via BeautifulSoup.
    Strips scripts, styles, and navigation noise.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Public API ────────────────────────────────────────────────────────────────

def load_document(source: str, **chunk_kwargs) -> List[Dict]:
    """
    Auto-detect source type and return a list of text chunks.

    Args:
        source:        File path (PDF / TXT / MD) or HTTP(S) URL.
        **chunk_kwargs: Forwarded to chunk_text (chunk_size, overlap).

    Returns:
        List of chunk dicts ready for embedding and storage.

    Raises:
        ValueError: If the file extension is not supported.
    """
    if source.startswith("http://") or source.startswith("https://"):
        raw = load_url(source)
    else:
        suffix = Path(source).suffix.lower()
        if suffix == ".pdf":
            raw = load_pdf(source)
        elif suffix in {".txt", ".md", ".markdown"}:
            raw = load_text(source)
        else:
            raise ValueError(f"Unsupported file type: '{suffix}'")

    return chunk_text(raw, source=source, **chunk_kwargs)
