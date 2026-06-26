"""
Text chunking and deduplication for law-book indexing.

Splits extracted statute text into overlapping word windows and removes
near-duplicate chunks before embedding into the FAISS vector store.
"""

import hashlib
from typing import List


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping word-based chunks suitable for embedding.

    Args:
        text: Full statute text from a law-book PDF.
        size: Target chunk size in words.
        overlap: Words shared between consecutive chunks for context continuity.

    Returns:
        List of chunk strings longer than 50 characters.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i : i + size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks


def remove_duplicates(chunks: List[str]) -> List[str]:
    """
    Remove duplicate chunks using a normalized MD5 hash of whitespace-collapsed text.

    Args:
        chunks: List of text chunks from chunk_text().

    Returns:
        Deduplicated list preserving first occurrence order.
    """
    seen = set()
    unique = []
    for chunk in chunks:
        h = hashlib.md5(" ".join(chunk.lower().split()).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    removed = len(chunks) - len(unique)
    if removed:
        print(
            f"  Dedup: {len(chunks)} → {len(unique)} chunks "
            f"({removed} duplicates hataaye)"
        )
    return unique
