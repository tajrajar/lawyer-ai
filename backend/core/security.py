"""
PII masking before document text is sent to external LLM providers.

Replaces sensitive identifiers with reversible placeholders so the model can
analyze case content without receiving raw personal or financial data.
"""

import re


def mask_sensitive_info(text: str) -> tuple[str, dict]:
    """
    Replace sensitive identifiers in document text with labeled placeholders.

    Args:
        text: Raw document text extracted from a PDF.

    Returns:
        Tuple of (masked_text, originals) where originals maps each placeholder
        key (e.g. "[CNIC_MASKED_1]") back to the original matched string.
    """
    originals = {}
    counter = 0
    rules = [
        # Pakistani CNIC: 12345-1234567-1 (5-7-1 digit groups with hyphens).
        (r"\b\d{5}-\d{7}-\d\b", "CNIC"),
        # Mobile numbers starting with +92 or 03 followed by 9 more digits.
        (r"(\+92|03)\d{9}", "PHONE"),
        # Standard email addresses (local@domain.tld).
        (r"\b[\w.-]+@[\w.-]+\.\w{2,}\b", "EMAIL"),
        # Bank account numbers: long uninterrupted digit sequences (14–20 digits).
        (r"\b\d{14,20}\b", "ACCOUNT"),
    ]
    for pattern, label in rules:
        def replacer(m, lbl=label):
            nonlocal counter
            counter += 1
            key = f"[{lbl}_MASKED_{counter}]"
            originals[key] = m.group(0)
            return key
        text = re.sub(pattern, replacer, text)
    return text, originals


def unmask(text: str, originals: dict) -> str:
    """
    Restore original sensitive values from placeholders.

    Args:
        text: Text containing [LABEL_MASKED_N] placeholders.
        originals: Mapping from placeholder keys to original values.

    Returns:
        Text with placeholders replaced by their original strings.
    """
    for key, val in originals.items():
        text = text.replace(key, val)
    return text
