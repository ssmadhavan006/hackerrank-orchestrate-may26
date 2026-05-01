# Architecture Decision Record (ADR)

## Why BM25 + Semantic Retrieval Instead of Pure LLM Retrieval

We use hybrid retrieval to improve recall and grounding:
- BM25 is strong for exact terms, product names, and policy keywords.
- Semantic embeddings (MiniLM) capture paraphrases and noisy phrasing.
- Final ranking blends both (`0.4 * BM25 + 0.6 * semantic`) to reduce misses from either method alone.

This is more reliable than pure LLM retrieval because it keeps retrieval deterministic, auditable, and corpus-bounded.

## Why These Escalation Thresholds

The escalation policy prioritizes safety over overconfident answering:
- High-risk patterns (fraud, identity compromise, legal threats, prompt injection) escalate immediately.
- Low retrieval confidence (weak evidence) auto-escalates.
- Low response confidence adds explicit uncertainty language or escalates.

The goal is to avoid unsupported claims and ensure sensitive cases route to humans.

## Failure Modes and How They Are Handled

1. LLM/API unavailable or timeout:
- Fallback response escalates safely with `confidence=0.0`.

2. Insufficient corpus match:
- Triggered by low retrieval score or few useful chunks.
- Escalated with explicit corpus-insufficient reasoning.

3. Adversarial input / prompt injection:
- Pattern-based detector flags manipulative or obfuscated inputs.
- Request marked `invalid`, escalated to security review, trigger pattern logged.

4. Ambiguous or cross-domain tickets:
- Interactive mode asks one clarification turn when confidence is low/ambiguous.
- Cross-domain routing splits sub-issues by domain and builds composite response.

5. Source freshness risk:
- Startup HEAD checks test source support domains.
- If unreachable, responses include a corpus-freshness warning.

## What We Would Build Next With More Time

- Calibrate confidence thresholds with held-out validation data.
- Add stronger retrieval evals (nDCG/Recall@k) and gold trace checks.
- Add a lightweight classifier model for request_type/domain before LLM.
- Improve multilingual adversarial detection with normalization + language-specific regex packs.
- Add a replayable audit viewer CLI for `decision_audit.jsonl`.
- Add automated regression suite for known edge cases and red-team prompts.
