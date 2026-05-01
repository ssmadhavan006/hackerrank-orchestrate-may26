from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from llm_client import call_llm
from preprocessor import preprocess
from retriever import retrieve

ALLOWED_STATUS = {"replied", "escalated"}
ALLOWED_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}

MALICIOUS_ESCALATION = {
    "status": "escalated",
    "product_area": "Security / Trust & Safety",
    "response": "This request cannot be processed as it appears to contain potentially manipulative content. Your ticket has been escalated to our trust and safety team.",
    "justification": "Ticket flagged for potential prompt injection or policy manipulation attempt.",
    "request_type": "invalid",
}

SAFE_ESCALATION_FALLBACK = {
    "status": "escalated",
    "product_area": "General Support",
    "response": "Thanks for your request. We need a specialist to review this case and will follow up after manual triage.",
    "justification": "Automatic triage could not produce a valid structured response with sufficient confidence.",
    "request_type": "product_issue",
}

HEDGE_WORDS = ("i'm not sure", "not sure", "may", "might", "possibly", "perhaps")
CONCRETE_STEP_HINTS = ("go to", "click", "contact", "visit", "follow", "submit", "provide", "check")
RUN_LOG_PATH = Path(__file__).resolve().parent / "run_log.jsonl"
SECURITY_LOG_PATH = Path(__file__).resolve().parent / "security_log.txt"
DECISION_CACHE_PATH = Path(__file__).resolve().parent / "decision_cache.json"


def _log_security_event(pattern: str, text: str) -> None:
    ts = datetime.now().isoformat()
    safe_preview = text.replace("\n", " ")[:240]
    with SECURITY_LOG_PATH.open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"[{ts}] malicious_pattern={pattern} preview={safe_preview}\n")


def _load_cache() -> Dict[str, Dict]:
    if not DECISION_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(DECISION_CACHE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Dict]) -> None:
    DECISION_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def _detect_multi_request(text: str) -> bool:
    lower = text.lower()
    question_marks = lower.count("?")
    numbered = bool(re.search(r"(^|\s)\d+[\).\-\:]", lower))
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    long_multi = len(sentences) > 4
    topic_markers = sum(1 for k in ("also", "another", "additionally", "plus", "and") if f" {k} " in f" {lower} ")
    varied_topics = topic_markers >= 2
    return numbered or question_marks >= 2 or (long_multi and varied_topics)


def _quick_product_area_hint(company: str, clean_text: str) -> str:
    text = clean_text.lower()
    if company == "visa":
        if any(k in text for k in ("dispute", "chargeback", "unauthorized", "fraud", "stolen")):
            return "Card Disputes"
        if any(k in text for k in ("traveller", "traveler", "travel", "trip", "vacation")):
            return "Travel Support"
        if any(k in text for k in ("card", "payment", "transaction", "pin", "cvv", "minimum spend", "merchant")):
            return "Card Management"
        if any(k in text for k in ("cheque", "check")):
            return "Traveller's Cheques"
    if company == "hackerrank":
        if any(k in text for k in ("assessment", "test", "candidate", "proctor", "plagiarism", "score")):
            return "Assessment Configuration"
        if any(k in text for k in ("login", "access", "password", "account", "merge")):
            return "Account Access"
        if any(k in text for k in ("interview", "interviewer", "lobby", "zoom")):
            return "Interview Configuration"
        if any(k in text for k in ("mock interview", "resume", "certificate")):
            return "Candidate Experience"
        if any(k in text for k in ("billing", "payment", "subscription", "pause", "cancel")):
            return "Billing & Payments"
        if any(k in text for k in ("submission", "not working", "down", "error")):
            return "Candidate Experience"
    if company == "claude":
        if any(k in text for k in ("billing", "invoice", "price", "plan", "subscription")):
            return "Billing & Payments"
        if any(k in text for k in ("project", "artifact", "conversation", "workspace")):
            return "Projects & Workspace"
        if any(k in text for k in ("claude code", "code", "terminal", "bedrock")):
            return "Claude Code"
        if any(k in text for k in ("lti", "education", "student", "professor", "canvas")):
            return "Claude for Education"
        if any(k in text for k in ("privacy", "data", "crawling", "crawl")):
            return "Privacy and Legal"
        if any(k in text for k in ("vulnerability", "security", "bug bounty", "jailbreak")):
            return "Safeguards"
    return "General Support"


def _escalation_taxonomy(enriched: dict, output: dict, docs: List[Dict]) -> str:
    text = str(enriched.get("clean_text", "")).lower()
    if enriched.get("is_potentially_malicious"):
        return "abusive_or_malicious_input"
    if any(k in text for k in ("fraud", "unauthorized", "chargeback", "dispute", "stolen")):
        return "fraud_or_financial_risk"
    # Only escalate for explicit account compromise / identity verification — not generic "account" mentions
    if any(k in text for k in ("account hacked", "account compromised", "verify identity", "unlock account", "account locked")):
        return "account_access_required"
    if any(k in text for k in ("legal", "lawsuit", "compliance", "regulatory")):
        return "legal_or_compliance"
    if len(docs) < 2:
        return "corpus_insufficient"
    if str(enriched.get("detected_company", "unknown")) == "unknown":
        return "ambiguous_cross_domain"
    if output.get("status") == "escalated":
        return "corpus_insufficient"
    return ""


def _confidence_score(enriched: dict, output: dict, docs: List[Dict]) -> float:
    score = 0.35
    if str(enriched.get("detected_company")) in {"hackerrank", "claude", "visa"}:
        score += 0.2
    score += min(len(docs), 8) * 0.04
    if output.get("status") == "replied":
        score += 0.15
    if enriched.get("is_sensitive_domain"):
        score -= 0.2
    if enriched.get("is_potentially_malicious"):
        score -= 0.3
    if _low_confidence_response(str(output.get("response", ""))):
        score -= 0.15
    return round(max(0.0, min(1.0, score)), 2)


_COMPANY_PRODUCT_AREAS = {
    "hackerrank": [
        "Account Access", "Assessment Configuration", "Interview Configuration",
        "Billing & Payments", "Certifications", "Integrations",
        "Candidate Experience", "Proctoring & Security", "General Support",
    ],
    "claude": [
        "Account Management", "Billing & Payments", "Claude Code",
        "Claude for Education", "Claude for Nonprofits", "Privacy and Legal",
        "Safeguards", "Projects & Workspace", "API & Console",
        "Mobile & Desktop Apps", "General Support",
    ],
    "visa": [
        "Card Disputes", "Billing & Payments", "Fraud & Security",
        "Travel Support", "Card Management", "Merchant Issues",
        "Traveller's Cheques", "General Support",
    ],
}

_REQUEST_TYPE_GUIDE = """
request_type definitions — choose exactly one, DO NOT default to "product_issue" without checking the others first:
- "bug": user reports a technical malfunction, system outage, broken feature, error, crash, or that something is not working (examples: "site is down", "submissions not working", "Claude Code stopped working", "nothing loads", "I keep getting an error"). Use "bug" for ANY report that something is broken or down.
- "feature_request": user explicitly asks for a new capability, enhancement, or something that does not currently exist (example: "can you add a feature for X")
- "invalid": ticket is spam, a social pleasantry ("thank you", "hello", "you're welcome"), completely unrelated to the company, or contains malicious/out-of-scope content (examples: "Who played Iron Man?", "Thank you for helping me", "What is the weather today?")
- "product_issue": all other tickets — how-to questions, billing/access/configuration help, policy questions, refund requests, account setup/management where the platform is working but the user needs guidance
Critical: if the user says anything is "not working", "down", "failing", "broken", "error", or "stopped", classify as "bug" NOT "product_issue"."""


def _build_system_prompt(company: str) -> str:
    areas = _COMPANY_PRODUCT_AREAS.get(company, ["General Support"])
    areas_list = ", ".join(f'"{a}"' for a in areas)
    return f"""You are a support triage agent for {company}. You must answer ONLY using the provided support documentation. Never invent policies, steps, or facts not present in the documentation.
Output format is strict: return exactly one JSON object with no markdown fences and no extra text before or after the JSON object.

Your task: analyze the support ticket and produce a JSON response with exactly these fields:
- status: "replied" or "escalated"
- product_area: choose the single most relevant category from this list for {company}: [{areas_list}]
- response: a helpful, grounded, user-facing reply (2-4 sentences). If escalating, explain why and what the user should expect.
- justification: 1-2 sentences explaining your routing decision and which docs informed it, and include source filenames from the provided docs (for example: "per data/visa/support/consumer/visa-rules.md")
- request_type: one of "product_issue", "feature_request", "bug", "invalid"

{_REQUEST_TYPE_GUIDE}

Escalation rules (always escalate if ANY of these apply):
1. The issue involves fraud, unauthorized transactions, account compromise, identity theft, or financial disputes
2. The issue requires you to DIRECTLY access, modify, or act upon the user's account on their behalf (e.g. restore access, change data for them) — NOT for self-service steps the user can do themselves (e.g. delete their own account following documented steps)
3. The support documentation does not contain enough information to safely answer
4. The ticket is threatening, abusive, or involves legal claims
5. The issue is about a bug with potential data loss or security implications
6. The issue involves merchant disputes or requires action against a third-party merchant

Reply rules:
1. Only include information that is explicitly stated in the provided documentation
2. Do not speculate or fill gaps with general knowledge
3. Keep response concise and actionable (2-4 sentences with concrete steps)
4. If the documentation has clear step-by-step instructions, include them in the response
5. If you cannot provide a helpful answer from the docs, escalate instead of giving a generic response

Critical: For Visa merchant/minimum spend questions, retrieve the merchant rules or card usage documentation and answer from there. Do NOT escalate if the documentation contains relevant information.

Respond with ONLY a valid JSON object. No markdown, no explanation outside the JSON."""


def _build_user_prompt(subject: str, issue: str, company: str, docs: List[Dict]) -> str:
    def truncate_words(text: str, max_words: int = 300) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    doc_blocks = []
    for chunk in docs:
        text = truncate_words(str(chunk.get("text", "")).strip(), max_words=300)
        src = str(chunk.get("source_file", "unknown"))
        if text:
            doc_blocks.append(f"---\n{text} [Source: {src}]")
    docs_text = "\n".join(doc_blocks) if doc_blocks else "---\nNo matching documentation retrieved."

    return f"""Support ticket from {company}:
Subject: {subject}
Issue: {issue}

Relevant support documentation:
{docs_text}
---

Produce the JSON triage output now."""


def _extract_json(raw: str) -> Dict | None:
    """Extract JSON from LLM response with multiple fallback strategies."""
    # Strategy 1: Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks
    cleaned = re.sub(r'```(?:json)?\s*', '', raw).replace('```', '').strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Regex extraction of JSON object
    match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Line-by-line key-value extraction (last resort)
    try:
        result = {}
        for key in ['status', 'product_area', 'response', 'justification', 'request_type']:
            pattern = rf'"{key}"\s*:\s*"([^"]*)"'
            m = re.search(pattern, raw)
            if m:
                result[key] = m.group(1)
        if len(result) >= 3:  # If we got at least 3 fields
            return result
    except Exception:
        pass
    
    return None


def _validate_output(data: Dict) -> Dict:
    out = dict(data)
    for key in ("status", "product_area", "response", "justification", "request_type"):
        out.setdefault(key, "")

    status = str(out["status"]).strip().lower()
    request_type = str(out["request_type"]).strip().lower()

    if status not in ALLOWED_STATUS:
        status = "escalated"
    if request_type not in ALLOWED_REQUEST_TYPE:
        request_type = "product_issue"

    out["status"] = status
    out["request_type"] = request_type
    out["product_area"] = str(out["product_area"]).strip() or "General Support"
    out["response"] = str(out["response"]).strip() or SAFE_ESCALATION_FALLBACK["response"]
    out["justification"] = str(out["justification"]).strip() or SAFE_ESCALATION_FALLBACK["justification"]

    return out


def _check_response_quality(response: str, docs: List[Dict]) -> bool:
    """Check if response is substantive and grounded in corpus."""
    if not response or len(response.strip()) < 30:
        return False
    
    lower = response.lower()
    # Flag responses that admit ignorance without providing help
    weak_phrases = [
        "i'm sorry, i cannot",
        "i am unable to",
        "out of scope from my capabilities",
        "does not contain enough information",
        "i don't have access",
    ]
    has_weak_phrases = any(phrase in lower for phrase in weak_phrases)
    
    # If response has weak phrases but no concrete steps, it's low quality
    has_concrete = any(hint in lower for hint in CONCRETE_STEP_HINTS)
    
    return not (has_weak_phrases and not has_concrete)


def _low_confidence_response(text: str) -> bool:
    lower = text.lower()
    has_hedge = any(word in lower for word in HEDGE_WORDS)
    has_concrete_steps = any(hint in lower for hint in CONCRETE_STEP_HINTS)
    return has_hedge and not has_concrete_steps


def triage_ticket(row: dict, row_id: int | None = None) -> dict:
    enriched = preprocess(row)
    cache_key = f"{str(enriched.get('detected_company','unknown')).lower()}||{str(enriched.get('clean_text','')).strip().lower()}"
    cache = _load_cache()

    if enriched["is_potentially_malicious"]:
        _log_security_event(enriched.get("malicious_pattern", "unknown"), enriched.get("clean_text", ""))
        out = dict(MALICIOUS_ESCALATION)
        out["_meta"] = {
            "row_id": row_id,
            "company": enriched.get("detected_company", "unknown"),
            "confidence_score": 0.05,
            "chunks_used": [],
            "escalation_reason": "abusive_or_malicious_input",
        }
        cache[cache_key] = out
        _save_cache(cache)
        return out

    if cache_key in cache:
        cached = dict(cache[cache_key])
        cached_meta = dict(cached.get("_meta", {}))
        cached_meta["row_id"] = row_id
        cached["_meta"] = cached_meta
        return cached

    detected_company = str(enriched.get("detected_company", "unknown"))
    company_filter = detected_company if detected_company in {"hackerrank", "claude", "visa"} else None
    product_area_hint = _quick_product_area_hint(detected_company, enriched["clean_text"])
    docs = retrieve(
        enriched["clean_text"],
        company_filter=company_filter,
        top_k=8,
        product_area_hint=product_area_hint,
        first_pass_k=12,
    )

    subject = str(enriched.get("subject", ""))
    issue = str(enriched.get("issue", ""))
    company_name = detected_company if detected_company != "unknown" else "unknown company"

    system_prompt = _build_system_prompt(company_name)
    user_prompt = _build_user_prompt(subject, issue, company_name, docs)

    raw = call_llm(system_prompt, user_prompt, temperature=0.0)
    parsed = _extract_json(raw)

    if parsed is None:
        retry_user_prompt = user_prompt + "\n\nYour previous response was not valid JSON. Respond with ONLY a JSON object."
        raw_retry = call_llm(system_prompt, retry_user_prompt, temperature=0.0)
        parsed = _extract_json(raw_retry)
        
        if parsed is None:
            # Third attempt: ultra-simplified prompt
            simple_system = "Return ONLY a JSON object with these fields: status, product_area, response, justification, request_type"
            simple_user = f"Ticket: {issue[:500]}. Company: {company_name}. Return JSON now."
            raw_retry2 = call_llm(simple_system, simple_user, temperature=0.0)
            parsed = _extract_json(raw_retry2)
            
            if parsed is None:
                # Final fallback: try to construct minimal valid response
                fallback = dict(SAFE_ESCALATION_FALLBACK)
                fallback["_meta"] = {
                    "row_id": row_id,
                    "company": detected_company,
                    "confidence_score": 0.1,
                    "chunks_used": [f"{d.get('source_file','unknown')}#{d.get('chunk_id',0)}" for d in docs],
                    "escalation_reason": "corpus_insufficient",
                }
                cache[cache_key] = fallback
                _save_cache(cache)
                return fallback

    validated = _validate_output(parsed)

    # Quality check: if response is weak, escalate instead
    if validated["status"] == "replied" and not _check_response_quality(validated["response"], docs):
        validated["status"] = "escalated"
        validated["justification"] = (
            f"Escalated (low_quality_response): {validated['justification']} "
            f"Response lacked specificity or actionable guidance."
        ).strip()

    if (
        enriched.get("is_sensitive_domain")
        and validated["status"] == "replied"
        and _low_confidence_response(validated["response"])
    ):
        validated["status"] = "escalated"
        validated["justification"] = (
            str(validated["justification"]).strip()
            + " Escalated due to sensitive-domain content with low-confidence response."
        ).strip()

    if _detect_multi_request(enriched.get("clean_text", "")):
        validated["justification"] = (
            f"{validated['justification']} Ticket contains multiple requests; primary issue addressed, secondary requests may require follow-up."
        ).strip()

    escalation_reason = _escalation_taxonomy(enriched, validated, docs)
    if validated["status"] == "escalated" and escalation_reason:
        if not validated["justification"].lower().startswith("escalated"):
            validated["justification"] = f"Escalated ({escalation_reason}): {validated['justification']}"

    validated["_meta"] = {
        "row_id": row_id,
        "company": detected_company,
        "confidence_score": _confidence_score(enriched, validated, docs),
        "chunks_used": [f"{d.get('source_file','unknown')}#{d.get('chunk_id',0)}" for d in docs],
        "escalation_reason": escalation_reason,
    }
    cache[cache_key] = validated
    _save_cache(cache)

    return validated
