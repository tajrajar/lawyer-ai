"""
Citation validation for LLM-generated legal references.

Filters hallucinated or malformed statute citations from model output before
results are returned to the user. Supports PPC/CrPC-style sections, Constitution
articles, named acts, and Negotiable Instruments Section 489-F.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Upper bounds for numeric section references per statute (inclusive).
VALID = {
    "PPC": set(range(1, 511)),
    "CrPC": set(range(1, 565)),
    "Constitution": set(range(1, 281)),
    "Contract Act": set(range(1, 238)),
    "MFLO": set(range(1, 14)),
    "CPC": set(range(1, 160)),
}

# Numeric section patterns: (regex, statute key in VALID).
PATTERNS = [
    (r"PPC\s+(\d+)", "PPC"),                    # e.g. "PPC 420"
    (r"CR\.?PC\s+(\d+)", "CrPC"),               # e.g. "CrPC 154" or "Cr.PC 497"
    (r"CONTRACT\s+ACT\s+(\d+)", "Contract Act"), # e.g. "Contract Act 73"
    (r"CONSTITUTION.*?(\d+)", "Constitution"),  # e.g. "Constitution Article 199" (numeric tail)
    (r"MFLO\s+(\d+)", "MFLO"),                  # e.g. "MFLO 7"
    (r"\bCPC\s+(\d+)", "CPC"),                  # e.g. "CPC 115"
]

# Whole-act references without a section number (common in family and property cases).
NAMED_ACT_PATTERNS = [
    r"FAMILY\s+COURTS?\s+ACT",           # Family Courts Act 1964
    r"TRANSFER\s+OF\s+PROPERTY\s+ACT",  # Transfer of Property Act 1882
    r"NEGOTIABLE\s+INSTRUMENTS?\s+ACT", # Negotiable Instruments Act 1881
    r"EVIDENCE\s+ACT",                  # Qanun-e-Shahadat / Evidence Act
    r"SPECIFIC\s+RELIEF\s+ACT",         # Specific Relief Act 1877
    r"MUSLIM\s+FAMILY\s+LAWS?\s+ORDINANCE",  # MFLO by full name
]

# Cheque-dishonour offence under the Negotiable Instruments Act (letter suffix, not plain numeric).
SECTION_489F = re.compile(r"(?:SECTION|S\.?)\s*489\s*[-.]?\s*F\b", re.IGNORECASE)
# Constitutional fundamental-rights articles (Articles 1–280 of the 1973 Constitution).
ARTICLE_REF = re.compile(r"ART(?:ICLE)?\.?\s*(\d+)", re.IGNORECASE)


def is_valid_ref(ref: str) -> bool:
    """
    Determine whether a citation string looks like a real Pakistani legal reference.

    Args:
        ref: Citation text from the LLM (e.g. "PPC 420", "Article 199").

    Returns:
        True if the reference matches a known valid format; False otherwise.
    """
    if not ref or not ref.strip():
        return False

    ref_upper = ref.upper()

    for pat in NAMED_ACT_PATTERNS:
        if re.search(pat, ref_upper):
            return True

    if SECTION_489F.search(ref) or re.search(r"\b489\s*[-.]?\s*F\b", ref_upper):
        return True

    m = ARTICLE_REF.search(ref)
    if m and 1 <= int(m.group(1)) <= 280:
        return True

    for pat, law in PATTERNS:
        m = re.search(pat, ref_upper)
        if m and int(m.group(1)) in VALID.get(law, set()):
            return True

    return False


def guard(result: dict) -> dict:
    """
    Remove invalid citations from an LLM analysis result.

    Args:
        result: Parsed analysis dict containing a relevant_sections list.

    Returns:
        The same dict with invalid sections removed and metadata added:
        hallucination_blocked (list of rejected refs) and verification_score (0–100).
    """
    good, bad = [], []
    for s in result.get("relevant_sections", []):
        ref = s.get("ref", "")
        (good if is_valid_ref(ref) else bad).append(s)

    if bad:
        blocked = [s["ref"] for s in bad]
        logger.warning("[GUARD] Blocked citations: %s", blocked)

    result["relevant_sections"] = good
    result["hallucination_blocked"] = [s["ref"] for s in bad]
    result["verification_score"] = round(
        len(good) / max(len(good) + len(bad), 1) * 100
    )
    return result
