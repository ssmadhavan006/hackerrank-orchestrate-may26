from __future__ import annotations

import re
from typing import Dict, List


COMPANIES = ("hackerrank", "claude", "visa")

KEYWORDS: Dict[str, List[str]] = {
    "hackerrank": [
        "hackerrank",
        "assessment",
        "test",
        "coding challenge",
        "recruiter",
        "candidate",
        "proctoring",
        "plagiarism",
        "hiring",
    ],
    "claude": [
        "claude",
        "anthropic",
        "claude.ai",
        "conversation",
        "artifact",
        "project",
        "claude pro",
        "claude team",
    ],
    "visa": [
        "visa",
        "card",
        "transaction",
        "payment",
        "chargeback",
        "merchant",
        "cvv",
        "pin",
        "contactless",
        "dispute",
    ],
}

MALICIOUS_PATTERNS = [
    "ignore previous instructions",
    "you are now",
    "pretend you are",
    "disregard",
    "jailbreak",
]

SENSITIVE_PATTERNS = [
    "fraud",
    "unauthorized",
    "dispute",
    "account hacked",
    "stolen",
    "chargeback",
    "legal",
    "lawsuit",
    "refund",
]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate_words(text: str, max_words: int = 1500) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _detect_company(company_raw: str, text: str) -> str:
    normalized_company = company_raw.strip().lower()
    if normalized_company in COMPANIES:
        return normalized_company

    if normalized_company in {"none", ""}:
        scores = {}
        lower_text = text.lower()
        for company, keywords in KEYWORDS.items():
            scores[company] = sum(lower_text.count(keyword) for keyword in keywords)

        max_score = max(scores.values(), default=0)
        if max_score == 0:
            return "unknown"

        leaders = [company for company, score in scores.items() if score == max_score]
        if len(leaders) != 1:
            return "unknown"
        return leaders[0]

    return "unknown"


def _contains_any(text: str, patterns: List[str]) -> bool:
    lower_text = text.lower()
    return any(pattern in lower_text for pattern in patterns)


def _first_matching_pattern(text: str, patterns: List[str]) -> str | None:
    lower_text = text.lower()
    for pattern in patterns:
        if pattern in lower_text:
            return pattern
    return None


def preprocess(row: dict) -> dict:
    subject = str(row.get("subject", "") or "")
    issue = str(row.get("issue", "") or "")
    company_raw = str(row.get("company", "") or "")

    clean_text = _truncate_words(_normalize_space(f"{subject} {issue}"))
    if not clean_text:
        clean_text = "No subject provided."
    detected_company = _detect_company(company_raw, clean_text)

    malicious_pattern = _first_matching_pattern(clean_text, MALICIOUS_PATTERNS)

    return {
        **row,
        "detected_company": detected_company,
        "clean_text": clean_text,
        "is_potentially_malicious": malicious_pattern is not None,
        "malicious_pattern": malicious_pattern or "",
        "is_sensitive_domain": _contains_any(clean_text, SENSITIVE_PATTERNS),
    }

