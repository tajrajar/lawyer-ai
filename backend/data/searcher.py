"""
Hybrid vector + keyword search over indexed Pakistani law books.

Combines FAISS semantic search with optional BM25 keyword scoring, query expansion
for common legal terms, and per-statute relevance weighting.
"""

import faiss, pickle, numpy as np, logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# When a case keyword appears in the query, append related statute terms to improve recall.
EXPANSIONS = {
    "fraud":           ["PPC 420", "cheating", "misrepresentation", "dishonest"],
    "cheating":        ["PPC 420", "cheating", "deception"],
    "breach of trust": ["PPC 406", "criminal breach", "trust"],
    "murder":          ["PPC 302", "qatl", "homicide", "killing"],
    "theft":           ["PPC 379", "PPC 380", "stolen"],
    "bail":            ["CrPC 497", "CrPC 498", "surety", "bail bond"],
    "divorce":         ["MFLO", "talaq", "khula", "dissolution"],
    "contract":        ["Contract Act", "Section 73", "agreement breach"],
    "forgery":         ["PPC 468", "PPC 471", "forged documents"],
    "property":        ["sale deed", "Transfer of Property", "possession"],
    "fir":             ["CrPC 154", "CrPC 156", "first information report"],
    "harassment":      ["PPC 354", "harassment", "assault"],
    "kidnapping":      ["PPC 359", "PPC 362", "abduction"],
    "check bounce":    ["Section 489-F", "dishonoured cheque", "Negotiable Instruments"],
    "double selling":  ["PPC 420", "PPC 406", "fraud", "property"],
}

# Boost criminal statutes for typical case queries; down-weight CPC for non-civil matters.
CATEGORY_WEIGHTS = {
    "Pakistan Penal Code 1860":          1.5,  # core criminal offences (PPC)
    "Code of Criminal Procedure 1898":   1.3,  # FIR, bail, investigation (CrPC)
    "Constitution of Pakistan 1973":     1.0,  # baseline for constitutional refs
    "Muslim Family Laws Ordinance 1961": 1.2,  # family/divorce cases
    "Contract Act 1872":                 1.2,  # commercial and property disputes
    "Code of Civil Procedure 1908":      0.6,  # less relevant for criminal/FIR uploads
}


def expand_query(query: str) -> str:
    """
    Append domain-specific legal terms when known case keywords appear in the query.

    Args:
        query: Raw document excerpt or search string.

    Returns:
        Original query with appended expansion terms (space-separated).
    """
    q = query
    for kw, terms in EXPANSIONS.items():
        if kw in query.lower():
            q += " " + " ".join(terms)
    return q


class LawSearcher:
    """
    FAISS + BM25 hybrid retriever over pre-built law-book chunk indexes.

    Loads law.faiss, chunks.pkl, and meta.pkl from the vector store directory.
    """

    def __init__(self, store="data/vector_store"):
        """
        Load the FAISS index, chunk corpus, metadata, and optional BM25 index.

        Args:
            store: Directory containing law.faiss, chunks.pkl, and meta.pkl.
        """
        self.encoder = SentenceTransformer(MODEL)
        self.index   = faiss.read_index(f"{store}/law.faiss")
        self.chunks  = pickle.load(open(f"{store}/chunks.pkl", "rb"))
        self.meta    = pickle.load(open(f"{store}/meta.pkl",   "rb"))

        try:
            from rank_bm25 import BM25Okapi
            self.bm25    = BM25Okapi([c.lower().split() for c in self.chunks])
            self.bm25_ok = True
        except ImportError:
            self.bm25_ok = False

        print(f"Searcher ready — {len(self.chunks)} chunks | BM25: {self.bm25_ok}")

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Retrieve the top_k most relevant law chunks for a query.

        Args:
            query: Document text or search string (first ~2000 chars used upstream).
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, law, section_ref, relevance_pct.
        """
        expanded = expand_query(query)

        vec = self.encoder.encode([expanded], convert_to_numpy=True)
        faiss.normalize_L2(vec)
        scores, idxs = self.index.search(vec, top_k * 3)

        faiss_sc = {}
        for score, i in zip(scores[0], idxs[0]):
            if i >= 0:
                law     = self.meta[i]["law"]
                weight  = CATEGORY_WEIGHTS.get(law, 1.0)
                faiss_sc[i] = float(score) * weight

        bm25_sc = {}
        if self.bm25_ok:
            raw = self.bm25.get_scores(expanded.lower().split())
            top = np.argsort(raw)[::-1][:top_k * 3]
            mx  = raw[top[0]] if len(top) > 0 else 1
            for i in top:
                if raw[i] > 0:
                    law        = self.meta[i]["law"]
                    weight     = CATEGORY_WEIGHTS.get(law, 1.0)
                    bm25_sc[i] = (float(raw[i]) / max(mx, 1)) * weight

        # 60% semantic (FAISS) + 40% keyword (BM25): embeddings catch paraphrases;
        # BM25 rewards exact section numbers and statute names in the query.
        combined = {
            i: 0.6 * faiss_sc.get(i, 0) + 0.4 * bm25_sc.get(i, 0)
            for i in set(faiss_sc) | set(bm25_sc)
        }
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "text":          self.chunks[i],
                "law":           self.meta[i]["law"],
                "section_ref":   self.meta[i].get("section_ref", ""),
                "relevance_pct": round(score * 100, 1),
            }
            for i, score in ranked
        ]


_searcher = None


def get_searcher(store="data/vector_store"):
    """
    Return a module-level singleton LawSearcher, loading it on first call.

    Args:
        store: Path to the vector store directory.

    Returns:
        LawSearcher instance, or None if the index files are missing or corrupt.
    """
    global _searcher
    if _searcher is None:
        try:
            _searcher = LawSearcher(store)
        except Exception as e:
            logger.warning(f"Searcher fail: {e}")
    return _searcher
