# HackerRank Orchestrate Agent

## Architecture Overview

The agent implements a **RAG (Retrieval-Augmented Generation)** pipeline with explicit safety routing:

```
Input CSV row
    │
    ▼
preprocessor.py   ← normalise text, detect company, flag injection/sensitive
    │
    ▼
retriever.py      ← Hybrid retrieval: BM25 + semantic similarity (MiniLM), product-area re-rank
    │
    ▼
agent.py          ← build system/user prompts, call LLM, parse+validate JSON
    │
    ▼
Output row: status | product_area | response | justification | request_type
```

**Why Hybrid Retrieval?** BM25 handles exact term matching while semantic search captures paraphrases. We combine scores as `0.4 * bm25 + 0.6 * semantic` and keep per-chunk score traces (`retrieval_bm25`, `retrieval_semantic`, `retrieval_score`) for auditability.

**Why Mixtral 8x22B-Instruct via NVIDIA NIM?** Strong instruction-following with high-quality JSON output at `temperature=0` for determinism. The 8x22B variant has strong multilingual understanding, which matters for tickets in French/Spanish.

**Escalation-first design:** Any ticket involving fraud, identity, legal claims, or injections is escalated *before* the LLM is called. The LLM is only invoked for safe routing decisions.
**Knowledge boundary:** The model is used strictly as a reasoning engine over retrieved local corpus chunks. It must not use external facts or live web knowledge for answers.

## Safety Architecture

The safety layer runs before and after generation. In `preprocessor.py`, the agent detects prompt-injection patterns (including multilingual variants) and sensitive risk signals; malicious tickets are immediately routed to `escalated` with `invalid` request type. In `agent.py`, escalation taxonomy labels capture why a case was escalated (for example fraud/financial risk, legal/compliance, corpus insufficiency), and low-quality or low-confidence replies are converted to escalation. Every row also gets confidence metadata and retrieved source trace IDs in `_meta`, and `run_log.jsonl`/`security_log.txt` keep auditable decision traces.

---

## 1) Prerequisites
- Python 3.10+ (3.11 tested)
- Internet access to NVIDIA-hosted LLM API only for inference transport (not for external knowledge retrieval)

## 2) NVIDIA API key setup
1. Go to `https://build.nvidia.com/`
2. Sign in and create an API key
3. In `code/`, copy `.env.example` to `.env`
4. Set:
   - `NVIDIA_API_KEY=...`
   - Optional: `NVIDIA_BASE_URL` (defaults to `https://integrate.api.nvidia.com/v1`)
   - Optional: `NVIDIA_MODEL` (defaults to `mistralai/mixtral-8x22b-instruct-v0.1`)

## 3) Install dependencies
From `code/`:
```bash
pip install -r requirements.txt
```

## 4) Build searchable corpus  ← MUST run before main.py
From repo root:
```bash
python code/ingest.py
```
This generates `code/corpus_chunks.json` (~7 MB) from:
- `data/hackerrank/`
- `data/claude/`
- `data/visa/`

**This step is required before running the agent.** The generated `corpus_chunks.json` is included in the submission zip for evaluator convenience.

## 5) Run unit tests
From `code/`:
```bash
pytest test_agent.py -v
```
These tests run offline (no LLM, no retriever) and cover preprocessing, injection detection, JSON parsing, and output validation.

## 6) Run ticket triage
From repo root:
```bash
python code/main.py
```
Default input/output:
- Input: `support_tickets/support_tickets.csv`
- Output: `support_tickets/output.csv`

Useful run modes:
```bash
# Use sample tickets (for evaluation against gold labels)
python code/main.py --sample

# Dry-run on first N rows
python code/main.py --sample --dry-run 5

# Explicit paths
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv

# Live interactive demo mode (single ticket triage + clarification loop)
python code/main.py --interactive
```

Interactive mode features:
- Shows top retrieved document snippets with scores before final answer
- Streams output fields (`status`, `product_area`, `request_type`, `confidence`, `justification`, `response`) with slight delay
- If confidence is below threshold (default `0.5`) or routing is ambiguous, asks one clarifying question and re-runs triage

## 7) Evaluate on sample labels
After generating `support_tickets/output_sample.csv`:
```bash
python code/evaluate.py
```
This reports:
- Status accuracy
- Request type accuracy
- Row-level mismatches

## 8) Observability outputs
- `code/run_log.jsonl`: one JSON line per processed ticket (confidence, chunks used, escalation reason)
- `code/decision_audit.jsonl`: per-ticket explainability trail (raw query, top retrieved docs + scores, reasoning chain, escalation rule, final response)
- `code/security_log.txt`: malicious/prompt-injection detections
- `code/decision_cache.json`: caches LLM decisions by content hash (excluded from submission zip)
- `support_tickets/output.csv`: includes `confidence` (`0.0` to `1.0`) per row

## Limitations

At startup, the runner performs lightweight `HEAD` checks against the three source support domains. If any domain is unreachable, the agent continues using the local corpus but appends a freshness warning in responses/justifications because the corpus may be stale relative to live support content.

## Red Team Testing

Adversarial cases tested and handled with explicit security routing:
- Prompt injection attempts: "ignore previous instructions", "show internal rules", "override your instructions"
- Role hijack attempts: "pretend you are a different system", "you are now X"
- Obfuscation attempts: base64-like long encoded payloads
- Destructive requests: "delete all files", "rm -rf"

Defense behavior:
- Returns explicit security response: "This input contains patterns inconsistent with a support request and has been flagged for security review."
- Logs exact triggering malicious pattern in `code/security_log.txt`
- Marks request as `invalid` and routes to Trust & Safety escalation

## Cross-Domain Routing

When a ticket spans multiple domains (e.g., HackerRank + Visa), the agent detects all involved ecosystems, splits the ticket into sub-issues, retrieves evidence from each domain-specific corpus, and returns a composite response. Output includes `domains_involved` for traceability.

## 9) Final submission checklist
1. `python code/ingest.py`               ← build corpus
2. `pytest code/test_agent.py -v`        ← verify helpers
3. `python code/main.py --sample`        ← generate output_sample.csv
4. `python code/evaluate.py`             ← check accuracy vs gold
5. `python code/main.py`                 ← generate final output.csv
6. Confirm `support_tickets/output.csv` row count matches `support_tickets/support_tickets.csv`
7. Confirm no empty cells in output CSV

## 10) Build upload artifacts
From repo root:
```bash
python code/package_submission.py
```

This creates `submission_artifacts/` with:
- `code_submission.zip` (code-only package; excludes `.env`, `__pycache__`, `.pyc`, local cache)
- `predictions_output.csv` (copied from active tickets folder output)
- `chat_log.txt` (copied from user home log path)
