"""
Command-line interface for analyzing Pakistani legal PDF documents.

Runs the full pipeline: PDF extraction → RAG law search → LLM analysis, with
optional provider fallback and JSON export.

Usage: python analyze.py document.pdf [--provider groq] [--no-fallback] [--save]
"""

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from data.pdf_engine import extract_text_auto
from core.analyzer import analyze_with_fallback, analyze

# RAG
RAG_READY = False
searcher  = None
try:
    from data.searcher import LawSearcher
    searcher  = LawSearcher("data/vector_store")
    RAG_READY = True
except Exception:
    pass

W = 62

def line(ch="─"):
    """Print a horizontal rule of width W."""
    print(ch * W)

def header(t):
    """Print a centred section header bordered by double lines."""
    print(); line("═"); print(f"  {t}"); line("═")

def section(t):
    """Print a single-line section title bordered by single lines."""
    print(); line(); print(f"  {t}"); line()

def wrap(t, indent=4):
    """Word-wrap text to terminal width and print with indentation."""
    for l in textwrap.wrap(str(t), width=W - indent):
        print(" " * indent + l)

def bar(pct):
    """Render a 20-character ASCII progress bar for a 0–100 percentage."""
    filled = round(pct / 5)
    return f"[{'█'*filled}{'░'*(20-filled)}] {pct}%"

def main():
    """Parse CLI arguments and run the three-step analysis pipeline."""
    parser = argparse.ArgumentParser(description="Pakistani Legal AI")
    parser.add_argument("pdf")
    parser.add_argument("--provider", default="groq",
                        choices=["groq", "openai", "gemini", "claude", "openclaw"])
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[ERROR] File nahi mili: {pdf_path}")
        sys.exit(1)

    header("Pakistani Legal AI — Document Analyzer")
    print(f"  File    : {pdf_path.name}")
    print(f"  Provider: {args.provider.upper()}")
    print(f"  RAG     : {'ON' if RAG_READY else 'OFF'}")

    section("Step 1/3 — PDF Reading")
    text = extract_text_auto(str(pdf_path))
    if len(text.strip()) < 50:
        print("[ERROR] Text nahi nikla")
        sys.exit(1)
    print(f"  Text: {len(text):,} characters")

    section("Step 2/3 — Law Search")
    law_context = ""
    if RAG_READY and searcher:
        results = searcher.search(text[:2000], top_k=5)
        law_context = "\n\n".join(
            f"[{r['law']} {r['relevance_pct']}%]\n{r['text']}"
            for r in results)
        print(f"  {len(results)} sections found")
        for r in results:
            print(f"    • {r['law']} {r['relevance_pct']}%")

    section("Step 3/3 — AI Analysis")
    print(f"  {args.provider.upper()} analyzing...")

    try:
        if args.no_fallback:
            # Single provider only — fail immediately if that provider errors.
            result = analyze(text, law_context, provider=args.provider)
            result["fallback_used"] = False
        else:
            result = analyze_with_fallback(
                text, law_context, preferred=args.provider)
    except (RuntimeError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    header("ANALYSIS RESULTS")
    print(f"  Case Type : {result.get('case_type','N/A')}")
    print(f"  Provider  : {result.get('provider','?').upper()}")

    section("Case Strength")
    pct = result.get("case_strength", 0)
    print(f"  {bar(pct)}")
    wrap(result.get("strength_reason",""), 2)

    section("Summary")
    wrap(result.get("summary",""), 2)

    section("Main Allegations")
    for a in result.get("main_allegations",[]):
        wrap(f"• {a}", 2)

    section("Key Evidence")
    for e in result.get("key_evidence",[]):
        wrap(f"• {e}", 2)

    section("Strong Points ✔")
    for p in result.get("strong_points",[]):
        wrap(f"✔  {p}", 2)

    section("Weak Points ✘")
    for p in result.get("weak_points",[]):
        wrap(f"✘  {p}", 2)

    section("Relevant Pakistani Law Sections")
    secs = result.get("relevant_sections",[])
    if secs:
        for s in secs:
            print(f"\n  [{s.get('ref','?')}] {s.get('title','')}")
            wrap(s.get("reason",""), 6)
    else:
        print("  Koi verified section nahi mila")

    if result.get("hallucination_blocked"):
        print(f"\n  [GUARD] Blocked: {result['hallucination_blocked']}")

    section("Suggested Arguments")
    for a in result.get("suggested_arguments",[]):
        wrap(f"→  {a}", 2)

    section("Legal Citations")
    for c in result.get("citations",[]):
        print(f"  §  {c}")

    line("═")

    if args.save:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = pdf_path.stem + f"_analysis_{ts}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n  [SAVED] {out}")

if __name__ == "__main__":
    main()