from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from agent import triage_ticket
from retriever import retrieve

OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]
RUN_LOG_PATH = Path(__file__).resolve().parent / "run_log.jsonl"


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    default_input = root / "support_tickets" / "support_tickets.csv"
    default_output = root / "support_tickets" / "output.csv"
    sample_input = root / "support_tickets" / "sample_support_tickets.csv"

    parser = argparse.ArgumentParser(description="HackerRank Orchestrate support agent runner")
    parser.add_argument("--input", default=str(default_input))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--sample", action="store_true", help="Use sample_support_tickets.csv as input")
    parser.add_argument("--dry-run", type=int, default=None, help="Process only first N rows")
    args = parser.parse_args()

    if args.sample:
        args.input = str(sample_input)
    return args


def _normalize_row(row: pd.Series) -> dict:
    return {
        "subject": row.get("subject", row.get("Subject", "")),
        "issue": row.get("issue", row.get("Issue", "")),
        "company": row.get("company", row.get("Company", "")),
    }


def main() -> None:
    args = _parse_args()
    start = time.time()
    RUN_LOG_PATH.write_text("", encoding="utf-8", newline="\n")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if args.dry_run is not None and args.dry_run >= 0:
        df = df.head(args.dry_run)

    total = len(df)
    print(f"HackerRank Orchestrate Support Agent | Tickets to process: {total}")

    # Warm retriever and ensure corpus has been loaded before loop.
    _ = retrieve("warmup", None, top_k=1)

    results = []
    for idx, row in tqdm(df.iterrows(), total=total, desc="Processing tickets"):
        row_dict = _normalize_row(row)
        try:
            out = triage_ticket(row_dict, row_id=idx + 1)
        except Exception as exc:
            out = {
                "status": "escalated",
                "product_area": "General Support",
                "response": "Your request has been escalated for manual review.",
                "justification": f"Automatic processing failed for this row: {type(exc).__name__}",
                "request_type": "product_issue",
                "_meta": {
                    "row_id": idx + 1,
                    "company": str(row_dict.get("company", "unknown")),
                    "confidence_score": 0.0,
                    "chunks_used": [],
                    "escalation_reason": "corpus_insufficient",
                },
            }

        meta = out.get("_meta", {})
        log_row = {
            "row_id": meta.get("row_id", idx + 1),
            "company": meta.get("company", str(row_dict.get("company", "unknown")).lower()),
            "status": out.get("status", "escalated"),
            "confidence": meta.get("confidence_score", 0.0),
            "chunks_used": meta.get("chunks_used", []),
            "escalation_reason": meta.get("escalation_reason", ""),
        }
        with RUN_LOG_PATH.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")

        results.append({k: out.get(k, "") for k in OUTPUT_COLUMNS})
        company = str(row_dict.get("company", "") or "unknown")
        print(f"Row {len(results)}/{total} | {company} | {out.get('status')} | {out.get('request_type')}")

    out_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    out_df.to_csv(output_path, index=False)

    elapsed = time.time() - start
    status_counts = out_df["status"].value_counts().to_dict()
    req_counts = out_df["request_type"].value_counts().to_dict()
    replied = status_counts.get("replied", 0)
    escalated = status_counts.get("escalated", 0)

    print("\nFinal stats")
    print(f"- Total tickets processed: {len(out_df)}")
    print(f"- replied: {replied}, escalated: {escalated}")
    print(f"- request_type breakdown: {req_counts}")
    print(f"- Time taken: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
