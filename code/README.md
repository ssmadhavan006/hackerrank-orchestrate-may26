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
retriever.py      ← BM25 full-corpus + per-company index, product-area re-rank
    │
    ▼
agent.py          ← build system/user prompts, call LLM, parse+validate JSON
    │
    ▼
Output row: status | product_area | response | justification | request_type
```

**Why BM25?** No embedding API needed, fully deterministic, fast for this corpus size (~7 MB), and interpretable — we can trace exactly which document tokens drove retrieval.

**Why Mixtral 8x22B-Instruct via NVIDIA NIM?** Strong instruction-following with high-quality JSON output at `temperature=0` for determinism. The 8x22B variant has strong multilingual understanding, which matters for tickets in French/Spanish.

**Escalation-first design:** Any ticket involving fraud, identity, legal claims, or injections is escalated *before* the LLM is called. The LLM is only invoked for safe routing decisions.

---

## 1) Prerequisites
- Python 3.10+ (3.11 tested)
- Internet access for NVIDIA-hosted LLM API

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
```

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
- `code/security_log.txt`: malicious/prompt-injection detections
- `code/decision_cache.json`: caches LLM decisions by content hash (excluded from submission zip)

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
