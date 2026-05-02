# Submission Runbook

## 1. Project Purpose
This project is a terminal-based support triage agent for HackerRank Orchestrate.  
It reads support tickets and produces structured outputs:

- `status` (`replied` / `escalated`)
- `product_area`
- `response`
- `justification`
- `request_type` (`product_issue` / `feature_request` / `bug` / `invalid`)

Additional operational fields included in output:

- `confidence`
- `domains_involved`

---

## 2. Step-by-Step Execution

### 2.1 Prerequisites
- Python 3.10+ (3.11 recommended)
- Internet access to NVIDIA endpoint only for model inference transport (not for external knowledge retrieval)
- NVIDIA API key available via env var (not committed)

### 2.2 Install dependencies
From repo root:

```powershell
pip install -r code/requirements.txt
```

### 2.3 Configure API key (local only)
Option A (session env var):

```powershell
$env:NVIDIA_API_KEY="your_key_here"
```

Option B (`code/.env`, local only, never commit):

```env
NVIDIA_API_KEY=your_key_here
```

### 2.4 Build the local corpus index

```powershell
python code/ingest.py
```

This generates `code/corpus_chunks.json`.

### 2.5 Run sample tickets

```powershell
python code/main.py --sample
python code/evaluate.py
```

### 2.6 Run full ticket set

```powershell
python code/main.py
```

Generates `support_tickets/output.csv`.

### 2.7 Strict LLM validation (no silent fallback)

```powershell
python code/main.py --sample --require-llm
```

- Pass: true LLM path is active.
- Fail: fallback was used (key/connectivity/provider issue).

---

## 3. Architecture

### 3.1 Pipeline
1. `preprocessor.py`
- Normalize ticket text
- Detect domain/company
- Detect malicious/injection patterns
- Flag sensitive content
- Detect cross-domain tickets

2. `retriever.py`
- Hybrid retrieval using lexical + semantic ranking
- Returns top evidence chunks with scores

3. `agent.py`
- Builds strict JSON prompt for LLM
- Parses/validates structured output
- Applies escalation taxonomy and confidence scoring
- Uses heuristic fallback when LLM unavailable
- Emits `_meta` audit context

4. `main.py`
- Batch orchestration and CSV I/O
- Runtime + decision logs
- Dashboard metrics
- `--require-llm` fail-fast mode

---

## 4. Techniques Used

### 4.1 Retrieval-Augmented Generation (RAG)
LLM responses are grounded in retrieved local corpus chunks.
The LLM is used as a reasoning engine only; it must not introduce outside facts beyond provided corpus evidence.

### 4.2 Hybrid retrieval
Combines lexical relevance (BM25-style) and semantic similarity for better coverage.

### 4.3 Safety routing
Pre-LLM checks detect injection/malicious patterns and high-risk intents, escalating early when needed.

### 4.4 Structured JSON enforcement
Prompt + parser + validator ensure output schema consistency.

### 4.5 Determinism-oriented design
Low-temperature generation and bounded fallback behavior for stable outputs.

---

## 5. Core Requirement Coverage

Implemented to satisfy challenge requirements:

1. Terminal-runner agent.
2. Input from `support_tickets/support_tickets.csv`.
3. Output to `support_tickets/output.csv`.
4. Grounding from provided local corpus under `data/`.
   The model receives only retrieved corpus chunks as support context.
5. Required triage fields generated.
6. Explicit escalation for unsafe/insufficient cases.
7. Environment-variable based secret handling.
8. `code/README.md` included with run instructions.

---

## 6. Extra Enhancements Beyond Minimum

1. Confidence score per decision.
2. `domains_involved` for traceability.
3. Cross-domain ticket decomposition/routing.
4. Multilingual injection pattern detection.
5. Security log (`code/security_log.txt`).
6. Run log (`code/run_log.jsonl`).
7. Decision audit trail (`code/decision_audit.jsonl`).
8. Decision cache (`code/decision_cache.json`).
9. Interactive mode with clarification loop.
10. Batch dashboard summary after runs.
11. Optional corpus freshness check toggle.
12. Strict LLM enforcement mode (`--require-llm`).

---

## 7. Session Fixes Applied

### 7.1 LLM key + diagnostics hardening
- Runtime env-key resolution in `code/llm_client.py`
- Quote-safe key parsing
- Clearer connection failure details

### 7.2 Over-escalation tuning
- Reduced aggressive fallback escalation thresholds in `code/agent.py`
- Narrowed sensitive trigger scope in `code/preprocessor.py`

### 7.3 Warning noise reduction
- Corpus freshness check now opt-in via `ENABLE_CORPUS_FRESHNESS_CHECK=1` in `code/main.py`

### 7.4 Fallback observability
- Added `llm_unavailable` metadata/logging
- Added `--require-llm` hard-fail mode to prevent silent fallback in validation

---

## 8. Operational Reality

- Code is submission-ready without committing `.env`.
- If evaluator provides `NVIDIA_API_KEY` + outbound HTTPS access to NVIDIA endpoint, live LLM mode should run.
- This does not change the knowledge boundary: responses remain corpus-grounded.
- If not, fallback mode may run unless `--require-llm` is used.

---

## 9. Submission Checklist

1. Ensure `.env` is not committed.
2. Confirm `code/README.md` exists and is accurate.
3. Run:
   - `python code/ingest.py`
   - `python code/main.py --sample`
   - `python code/evaluate.py`
   - `python code/main.py`
4. Verify output row counts match input.
5. Package artifacts:

```powershell
python code/package_submission.py
```

6. Upload code zip, predictions CSV, and transcript log.

---

## 10. Recommended Final Verification

```powershell
python code/main.py --sample --require-llm
python code/evaluate.py
python code/main.py --require-llm
```

If these succeed on your target machine, you have confirmed true end-to-end LLM operation.
