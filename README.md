# Pakistani Legal AI Analyzer ⚖️

A RAG-based AI assistant that analyzes legal documents (FIRs, contracts, case files) against Pakistani law and produces a structured, bilingual case analysis — with built-in safeguards against hallucinated citations.

> **Disclaimer:** This tool provides AI-generated informational analysis only. It is **not a substitute for professional legal advice**. Always consult a qualified lawyer before taking any legal action.

## What It Does

Upload a legal document (e.g. an FIR or contract), and the system:

1. Extracts text from the PDF (with automatic OCR fallback for scanned documents)
2. Masks sensitive personal information (CNIC, phone, email, account numbers) before sending anything to an LLM
3. Retrieves relevant sections from Pakistani law — Constitution, Penal Code (PPC), Criminal Procedure Code (CrPC), Civil Procedure Code (CPC), Family Law, Contract Act — using hybrid FAISS + BM25 semantic search
4. Sends the document and retrieved law context to an LLM for analysis
5. Verifies every cited law section against a citation guard, filtering out any reference the model may have hallucinated
6. Returns a structured report: case strength score, key allegations, strong/weak points, and suggested arguments — in Urdu or English

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | FastAPI |
| PDF extraction | PyMuPDF, pdfplumber, EasyOCR (fallback for scanned docs) |
| Retrieval (RAG) | FAISS (vector search) + BM25 (keyword search), `sentence-transformers` multilingual embeddings |
| LLM providers | Groq, OpenAI, Gemini, Anthropic Claude, local OpenClaw — with automatic fallback if one is unavailable |
| Safety | PII masking, citation/hallucination guard |

## Project Structure

```
backend/
├── app.py                  # FastAPI server + web UI
├── analyze.py               # CLI entry point
├── law_indexer.py           # Builds the FAISS index from law book PDFs
├── core/
│   ├── analyzer.py          # LLM prompting, analysis, provider fallback
│   ├── citation_guard.py    # Validates law citations, filters hallucinations
│   ├── router.py            # Multi-provider LLM routing
│   └── security.py          # PII masking (CNIC, phone, email, account no.)
├── data/
│   ├── pdf_engine.py        # PDF text extraction with OCR fallback
│   ├── searcher.py          # Hybrid FAISS + BM25 retrieval
│   ├── law_books/           # Source PDFs (Constitution, PPC, CrPC, etc.)
│   ├── vector_store/        # Generated FAISS index
│   └── indexing/dedup.py    # Chunking + deduplication for indexing
└── tests/
    └── test_system.py
```

## Setup

```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate        # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in `backend/` with your API keys:

```
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

You only need keys for the providers you intend to use — the system will fall back to whichever ones are configured.

### Build the law index (first run only)

```bash
python law_indexer.py
```

### Run the app

```bash
uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

### Run via CLI

```bash
python analyze.py --pdf path/to/document.pdf --provider groq
```

### Run tests

```bash
python -m pytest tests/test_system.py
```

## Key Design Decisions

- **Hallucination guarding**: every law citation returned by the LLM is checked against known section/article formats (including PPC, CrPC, Constitution articles, and named acts like the Family Courts Act) before being shown to the user.
- **Privacy by default**: CNIC numbers, phone numbers, emails, and account numbers are masked before any document text is sent to an external LLM provider.
- **Multi-provider resilience**: if one LLM provider is down or rate-limited, the system automatically falls back to the next configured provider.
- **Bilingual output**: results can be generated in Urdu or English based on user selection.

## Limitations

- This is an informational tool, not a legal advice service — outputs should always be reviewed by a qualified lawyer.
- Currently has no authentication on the API; not intended for public-facing production deployment without adding auth and rate limiting.
- Citation guard checks reference *format* validity, not whether the LLM's reasoning about applicability is correct.

## License

This project is shared for portfolio and educational purposes.
