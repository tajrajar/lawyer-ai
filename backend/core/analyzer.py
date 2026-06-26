# -*- coding: utf-8 -*-
"""
Legal document analysis engine.

Builds LLM prompts from extracted document text and RAG context, parses structured
JSON responses, applies citation validation, and supports multi-provider fallback.
"""

import json, os, asyncio, logging
from core.router import call_llm_async
from core.security import mask_sensitive_info
from core.citation_guard import guard

logger = logging.getLogger(__name__)

SYSTEM = """You are a senior Pakistani Legal Expert with 20 years court experience. Specialize in PPC 1860, CrPC 1898, Constitution 1973, MFLO 1961, Contract Act 1872, CPC 1908. MANDATORY: Return ONLY valid JSON starting with {. relevant_sections MUST have 4-5 entries. For fraud cases MUST include PPC 420 and PPC 406. For forgery MUST include PPC 468 and PPC 471. For any FIR MUST include CrPC 154 and CrPC 156. For contract breach MUST include Contract Act 73. case_strength must be honest 0-100."""

PROMPT = """You are a senior Pakistani Legal Expert analyzing a legal document.
Read the document carefully and extract ALL relevant information.

<context>
{ctx}
</context>

<document>
{doc}
</document>

Based on what you READ in the document above, return this JSON:
{{
  "case_type": "identify from document: FIR/Contract Dispute/Family Case/Property Case/Criminal etc",
  "summary": "3-4 sentences summarizing what actually happened in this specific document",
  "main_allegations": [
    "exact allegation 1 from this document",
    "exact allegation 2 from this document",
    "exact allegation 3 from this document"
  ],
  "key_evidence": [
    "exact evidence mentioned in document 1",
    "exact evidence mentioned in document 2",
    "exact evidence mentioned in document 3"
  ],
  "strong_points": [
    "strong point based on actual document content",
    "strong point based on actual document content",
    "strong point based on actual document content"
  ],
  "weak_points": [
    "weak point based on actual document content",
    "weak point based on actual document content"
  ],
  "relevant_sections": [
    {{
      "ref": "actual applicable law section",
      "title": "section title",
      "reason": "why this section applies to THIS specific case"
    }}
  ],
  "case_strength": 0,
  "strength_reason": "honest assessment based on evidence in this document",
  "suggested_arguments": [
    "specific legal argument for this case",
    "specific legal argument for this case",
    "specific legal argument for this case"
  ],
  "citations": [
    "Law Section - Law Name Year"
  ]
}}

RULES:
- Read document carefully and extract REAL information
- For fraud/cheating cases: consider PPC 420, 406, 468, 471
- For murder/injury: consider PPC 302, 324, 325
- For family cases: consider MFLO, Family Courts Act
- For property: consider Transfer of Property Act, Contract Act
- For FIR: always include CrPC 154, 156
- For bail: consider CrPC 497, 498
- ONLY cite sections that actually apply to THIS document
- case_strength should be honest 0-100 based on evidence
{lang_instruction}"""

LANG_INSTRUCTIONS = {
    "Urdu": (
        "- LANGUAGE: Write summary, main_allegations, key_evidence, strong_points, "
        "weak_points, strength_reason, and suggested_arguments in Urdu. "
        "Keep JSON keys in English; law section refs may use English (e.g. PPC 420)."
    ),
    "English": (
        "- LANGUAGE: Write summary, main_allegations, key_evidence, strong_points, "
        "weak_points, strength_reason, and suggested_arguments in clear English."
    ),
}


def normalize_lang(target_lang: str) -> str:
    """
    Map user-facing language labels to a supported output language.

    Args:
        target_lang: Raw language string from CLI or API (e.g. "Urdu", "en").

    Returns:
        Either "Urdu" or "English". Defaults to "Urdu" for unknown values.
    """
    lang = (target_lang or "Urdu").strip().lower()
    if lang in ("urdu", "ur", "اردو"):
        return "Urdu"
    if lang in ("english", "en"):
        return "English"
    return "Urdu"


def parse_json(raw: str) -> dict:
    """
    Parse JSON from an LLM response, tolerating markdown fences and extra prose.

    Args:
        raw: Raw text returned by the LLM.

    Returns:
        Parsed dict from the model response.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        logger.debug("Full JSON parse failed: %s", e)
    # Models sometimes wrap JSON in explanatory text; extract the outermost object.
    s = clean.find("{")
    e = clean.rfind("}") + 1
    if s != -1 and e > s:
        try:
            return json.loads(clean[s:e])
        except json.JSONDecodeError as err:
            logger.warning("Substring JSON parse failed: %s", err)
    raise ValueError(f"JSON fail: {raw[:200]}")


async def analyze_async(
    doc_text,
    law_context="",
    provider="groq",
    model=None,
    mask_pii=True,
    target_lang="Urdu",
):
    """
    Analyze a legal document asynchronously via the selected LLM provider.

    Args:
        doc_text: Full extracted text of the uploaded document.
        law_context: RAG-retrieved statute excerpts to ground the analysis.
        provider: LLM provider key (groq, openai, gemini, claude, openclaw).
        model: Optional model override; uses provider default when None.
        mask_pii: Whether to mask CNIC, phone, email, and account numbers first.
        target_lang: Output language for narrative fields ("Urdu" or "English").

    Returns:
        Structured analysis dict including case_type, summary, relevant_sections,
        and metadata fields (provider, target_lang, pii_masked, masked_count).
    """
    originals = {}
    if mask_pii:
        doc_text, originals = mask_sensitive_info(doc_text)
    lang = normalize_lang(target_lang)
    prompt = PROMPT.format(
        ctx=law_context or "No context",
        doc=doc_text[:8000],
        lang_instruction=LANG_INSTRUCTIONS[lang],
    )
    raw    = await call_llm_async(SYSTEM, prompt, provider=provider,
                                   model=model, temperature=0.05)
    result = parse_json(raw)
    result = guard(result)
    result["provider"]     = provider
    result["target_lang"]  = lang
    result["pii_masked"]   = mask_pii
    result["masked_count"] = len(originals)
    return result


def analyze(doc_text, law_context="", provider="groq", model=None, mask_pii=True, target_lang="Urdu"):
    """
    Synchronous wrapper around analyze_async for CLI and scripting use.

    Args:
        doc_text: Full extracted text of the uploaded document.
        law_context: RAG-retrieved statute excerpts.
        provider: LLM provider key.
        model: Optional model override.
        mask_pii: Whether to mask sensitive identifiers before sending to the LLM.
        target_lang: Output language for narrative fields.

    Returns:
        Structured analysis dict (same shape as analyze_async).
    """
    return asyncio.run(
        analyze_async(doc_text, law_context, provider, model, mask_pii, target_lang)
    )


def analyze_with_fallback(doc_text, law_context="", preferred="groq", mask_pii=True, target_lang="Urdu"):
    """
    Try the preferred LLM provider first, then fall through to other configured providers.

    Args:
        doc_text: Full extracted text of the uploaded document.
        law_context: RAG-retrieved statute excerpts.
        preferred: Provider to attempt first.
        mask_pii: Whether to mask sensitive identifiers.
        target_lang: Output language for narrative fields.

    Returns:
        Structured analysis dict from the first provider that succeeds.

    Raises:
        RuntimeError: If every provider fails or lacks required configuration.
    """
    keys = {
        "groq":     "GROQ_API_KEY",
        "openai":   "OPENAI_API_KEY",
        "gemini":   "GEMINI_API_KEY",
        "claude":   "ANTHROPIC_API_KEY",
        "openclaw": "OPENCLAW_BASE_URL",
    }
    # Preferred provider first; remaining providers follow in dict insertion order.
    order = [preferred] + [p for p in keys if p != preferred]
    last  = None
    for p in order:
        # OpenClaw is local and needs no API key; cloud providers require env credentials.
        if p != "openclaw" and not os.getenv(keys.get(p, "")):
            continue
        try:
            print(f"  [AI] {p}...")
            r = analyze(doc_text, law_context, provider=p, mask_pii=mask_pii, target_lang=target_lang)
            r["fallback_used"] = (p != preferred)
            return r
        except Exception as e:
            print(f"  [FAIL] {p}: {e}")
            last = e
    raise RuntimeError(f"All failed: {last}")
