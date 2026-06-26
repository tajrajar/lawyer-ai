"""
One-time FAISS index builder for Pakistani law-book PDFs.

Reads statute PDFs from data/law_books/, chunks and deduplicates the text,
embeds chunks with sentence-transformers, and writes the vector store used by
the RAG search pipeline. Run once before using analyze.py or the API.

Usage: python law_indexer.py
"""

import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from data.pdf_engine import extract_text_auto
from data.indexing.dedup import chunk_text, remove_duplicates

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Expected filenames in data/law_books/ mapped to their full statute titles.
LAW_FILES = {
    "ppc.pdf":          "Pakistan Penal Code 1860",
    "crpc.pdf":         "Code of Criminal Procedure 1898",
    "constitution.pdf": "Constitution of Pakistan 1973",
    "family_law.pdf":   "Muslim Family Laws Ordinance 1961",
    "contract_act.pdf": "Contract Act 1872",
    "cpc.pdf":          "Code of Civil Procedure 1908",
}


def build_index(books_dir="data/law_books", save_dir="data/vector_store"):
    """
    Build and persist the FAISS vector index from law-book PDFs.

    Args:
        books_dir: Directory containing the statute PDF files listed in LAW_FILES.
        save_dir: Output directory for law.faiss, chunks.pkl, and meta.pkl.
    """
    print("=" * 50)
    print("Pakistani Legal AI — Law Index Builder")
    print("=" * 50)

    model = SentenceTransformer(MODEL_NAME)
    all_chunks, all_meta = [], []

    for fname, law_name in LAW_FILES.items():
        path = os.path.join(books_dir, fname)
        if not os.path.exists(path):
            print(f"\n[SKIP] {fname} nahi mila — law_books/ mein rakhein")
            continue

        print(f"\n[PROCESSING] {law_name}")
        text = extract_text_auto(path)
        chunks = remove_duplicates(chunk_text(text))
        for c in chunks:
            all_chunks.append(c)
            all_meta.append({"law": law_name, "file": fname})
        print(f"  → {len(chunks)} chunks add kiye")

    if not all_chunks:
        print("\n[ERROR] Koi bhi PDF process nahi hua. law_books/ folder check karein.")
        return

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Embeddings bana raha hoon — 10-20 minute lag sakte hain...")

    embeddings = model.encode(
        all_chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    # L2-normalize so inner-product search approximates cosine similarity.
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(save_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(save_dir, "law.faiss"))
    with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)
    with open(os.path.join(save_dir, "meta.pkl"), "wb") as f:
        pickle.dump(all_meta, f)

    print(f"\n[DONE] Index saved → {save_dir}/")
    print("Ab analyze.py se documents analyze kar sakte hain.")


if __name__ == "__main__":
    build_index()
