# Phase 1 Setup

## Get NVIDIA API key
1. Go to `https://build.nvidia.com/`.
2. Sign in and create an API key.
3. Copy `code/.env.example` to `code/.env` and set `NVIDIA_API_KEY`.

## Install dependencies
```bash
pip install -r requirements.txt
```

## Build corpus chunks
```bash
python ingest.py
```

This creates `code/corpus_chunks.json` from `data/hackerrank`, `data/claude`, and `data/visa`.

## Run the agent
```bash
python main.py
```
