# HackerRank Orchestrate Agent

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

## 4) Build searchable corpus
From repo root:
```bash
python code/ingest.py
```
This generates `code/corpus_chunks.json` from:
- `data/hackerrank/`
- `data/claude/`
- `data/visa/`

## 5) Run ticket triage
From repo root:
```bash
python code/main.py
```
Default input/output:
- Input: `support_tickets/support_tickets.csv`
- Output: `support_tickets/output.csv`

Useful run modes:
```bash
python code/main.py --sample
python code/main.py --sample --dry-run 5
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv
```

## 6) Evaluate on sample labels
After generating `support_tickets/output_sample.csv`:
```bash
python code/evaluate.py
```
This reports:
- Status accuracy
- Request type accuracy
- Row-level mismatches

## 7) Observability outputs
- `code/run_log.jsonl`: one JSON line per processed ticket (confidence, chunks used, escalation reason)
- `code/security_log.txt`: malicious/prompt-injection detections

## 8) Final submission checklist
1. `python code/ingest.py`
2. `python code/main.py --sample --dry-run 5`
3. `python code/evaluate.py`
4. `python code/main.py`
5. Confirm `support_tickets/output.csv` row count matches `support_tickets/support_tickets.csv`
6. Confirm no empty cells in output CSV
