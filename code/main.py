from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm
try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None  # type: ignore
    Table = None  # type: ignore

from agent import SECURITY_LOG_PATH, triage_ticket
from preprocessor import preprocess
from retriever import retrieve
from runtime_state import set_stale_domains

OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type", "confidence", "domains_involved"]
RUN_LOG_PATH = Path(__file__).resolve().parent / "run_log.jsonl"
AUDIT_LOG_PATH = Path(__file__).resolve().parent / "decision_audit.jsonl"
SOURCE_URLS = {
    "hackerrank": "https://support.hackerrank.com/",
    "claude": "https://support.claude.com/en/",
    "visa": "https://www.visa.co.in/support.html",
}
console = Console() if Console is not None else None


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    tickets_dir = root / "support_tickets"
    issues_dir = root / "support_issues"
    active_dir = tickets_dir if tickets_dir.exists() else issues_dir
    default_input = active_dir / "support_tickets.csv"
    default_output = active_dir / "output.csv"
    sample_input = active_dir / "sample_support_tickets.csv"
    sample_output = active_dir / "output_sample.csv"

    parser = argparse.ArgumentParser(description="HackerRank Orchestrate support agent runner")
    parser.add_argument("--input", default=str(default_input))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--sample", action="store_true", help="Use sample_support_tickets.csv as input, output to output_sample.csv")
    parser.add_argument("--dry-run", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--interactive", action="store_true", help="Run live interactive support triage mode")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold to ask one clarifying question in interactive mode")
    parser.add_argument("--require-llm", action="store_true", help="Fail fast if heuristic fallback is used due to LLM unavailability")
    parser.add_argument("--fallback-only", action="store_true", help="Disable LLM calls and run deterministic heuristic fallback for all rows")
    args = parser.parse_args()

    if args.sample:
        args.input = str(sample_input)
        # Only auto-set output if user hasn't explicitly provided one
        if args.output == str(default_output):
            args.output = str(sample_output)
    return args


def _normalize_row(row: pd.Series) -> dict:
    return {
        "subject": row.get("subject", row.get("Subject", "")),
        "issue": row.get("issue", row.get("Issue", "")),
        "company": row.get("company", row.get("Company", "")),
    }


def _check_corpus_freshness() -> list[str]:
    if os.getenv("ENABLE_CORPUS_FRESHNESS_CHECK", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        set_stale_domains([])
        return []
    unreachable: list[str] = []
    for name, url in SOURCE_URLS.items():
        try:
            resp = httpx.head(url, timeout=5.0, follow_redirects=True)
            if resp.status_code >= 400:
                unreachable.append(name)
        except Exception:
            unreachable.append(name)
    set_stale_domains(unreachable)
    if unreachable:
        print(f"[WARN] Corpus freshness check: unreachable domains at startup: {', '.join(unreachable)}")
    return unreachable


def _append_audit_trail(row_dict: dict, out: dict) -> None:
    meta = out.get("_meta", {}) or {}
    retrieved_docs = meta.get("retrieved_docs", []) or []
    audit = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "raw_query": {
            "company": row_dict.get("company", ""),
            "subject": row_dict.get("subject", ""),
            "issue": row_dict.get("issue", ""),
        },
        "top_retrieved_documents": [
            {
                "source_file": d.get("source_file", "unknown"),
                "chunk_id": d.get("chunk_id", 0),
                "retrieval_score": d.get("retrieval_score", 0.0),
                "snippet": str(d.get("snippet", ""))[:240],
            }
            for d in retrieved_docs[:3]
        ],
        "classification_reasoning": meta.get("reasoning_chain", []),
        "escalation_decision": {
            "status": out.get("status", "escalated"),
            "triggering_rule": meta.get("escalation_reason", ""),
            "confidence": out.get("confidence", meta.get("confidence_score", 0.0)),
        },
        "final_response": {
            "product_area": out.get("product_area", ""),
            "request_type": out.get("request_type", ""),
            "justification": out.get("justification", ""),
            "response": out.get("response", ""),
        },
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")


def _print_batch_dashboard(out_df: pd.DataFrame, elapsed: float) -> None:
    status_counts = out_df["status"].value_counts().to_dict() if "status" in out_df.columns else {}
    req_counts = out_df["request_type"].value_counts().to_dict() if "request_type" in out_df.columns else {}
    area_counts = out_df["product_area"].value_counts().to_dict() if "product_area" in out_df.columns else {}
    avg_conf = float(out_df["confidence"].mean()) if "confidence" in out_df.columns and len(out_df) else 0.0

    adversarial = 0
    if SECURITY_LOG_PATH.exists():
        try:
            adversarial = len([ln for ln in SECURITY_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()])
        except Exception:
            adversarial = 0

    if Table is None or console is None:
        print("\nBatch Dashboard")
        print(f"Total tickets processed: {len(out_df)}")
        print(f"Status replied/escalated: {status_counts.get('replied',0)}/{status_counts.get('escalated',0)}")
        print(f"Average confidence: {avg_conf:.2f}")
        print(f"Adversarial inputs detected: {adversarial}")
        print(f"Processing time (s): {elapsed:.2f}")
        print(f"Request type breakdown: {req_counts}")
        print(f"Product area breakdown: {area_counts}")
        return

    summary = Table(title="Support Triage Batch Dashboard")
    summary.add_column("Metric", style="cyan", no_wrap=True)
    summary.add_column("Value", style="green")
    summary.add_row("Total tickets processed", str(len(out_df)))
    summary.add_row("Status: replied", str(status_counts.get("replied", 0)))
    summary.add_row("Status: escalated", str(status_counts.get("escalated", 0)))
    summary.add_row("Average confidence", f"{avg_conf:.2f}")
    summary.add_row("Adversarial inputs detected", str(adversarial))
    summary.add_row("Processing time (s)", f"{elapsed:.2f}")
    console.print(summary)

    req_table = Table(title="Request Type Breakdown")
    req_table.add_column("request_type", style="magenta")
    req_table.add_column("count", justify="right")
    for k, v in sorted(req_counts.items(), key=lambda x: x[1], reverse=True):
        req_table.add_row(str(k), str(v))
    console.print(req_table)

    area_table = Table(title="Product Area Breakdown (Top 12)")
    area_table.add_column("product_area", style="yellow")
    area_table.add_column("count", justify="right")
    for k, v in sorted(area_counts.items(), key=lambda x: x[1], reverse=True)[:12]:
        area_table.add_row(str(k), str(v))
    console.print(area_table)


def _stream_fields(out: dict, delay_s: float = 0.2) -> None:
    print("\nTriage Result")
    print("-" * 52)
    for key in ("status", "product_area", "request_type", "confidence", "justification", "response"):
        value = out.get(key, "")
        if key == "confidence":
            try:
                value = f"{float(value):.2f}"
            except Exception:
                pass
        print(f"{key}: {value}")
        time.sleep(delay_s)


def _show_retrieval_chain(out: dict, top_n: int = 4) -> None:
    docs = out.get("_meta", {}).get("retrieved_docs", []) or []
    if not docs:
        return
    print("\nRetrieved Document Snippets")
    print("-" * 52)
    for i, d in enumerate(docs[:top_n], start=1):
        src = d.get("source_file", "unknown")
        score = float(d.get("retrieval_score", 0.0))
        snippet = str(d.get("snippet", "")).replace("\n", " ").strip()[:220]
        print(f"[{i}] score={score:.3f} | {src}")
        print(f"    {snippet}")


def _clarifying_question(row: dict) -> str:
    enriched = preprocess(row)
    text = str(enriched.get("clean_text", "")).lower()
    if any(k in text for k in ("login", "password", "access", "locked")) and any(k in text for k in ("billing", "payment", "invoice", "refund", "charge")):
        return "Are you asking about account access or billing/payment?"
    if any(k in text for k in ("visa", "card", "transaction", "charge")):
        return "Is this about an unauthorized/fraud transaction or a regular merchant/payment issue?"
    if any(k in text for k in ("hackerrank", "assessment", "test", "submission")):
        return "Should I route this as assessment configuration, candidate-side issue, or account access?"
    if any(k in text for k in ("claude", "api", "bedrock", "workspace", "project")):
        return "Is this primarily an API/Bedrock issue or an account/workspace management issue?"
    return "Can you clarify the exact product area you need help with?"


def _interactive_mode(conf_threshold: float) -> None:
    _check_corpus_freshness()
    print("Interactive Support Triage Mode")
    print("Enter 'exit' in any field to quit.\n")
    while True:
        company = input("Company (HackerRank/Claude/Visa/None): ").strip()
        if company.lower() == "exit":
            break
        subject = input("Subject: ").strip()
        if subject.lower() == "exit":
            break
        issue = input("Issue: ").strip()
        if issue.lower() == "exit":
            break

        row = {"company": company, "subject": subject, "issue": issue}
        try:
            out = triage_ticket(row, row_id=None)
        except Exception as exc:
            out = {
                "status": "escalated",
                "product_area": "General Support",
                "response": "Your request has been escalated for manual review.",
                "justification": f"Automatic processing failed in interactive mode: {type(exc).__name__}",
                "request_type": "product_issue",
                "confidence": 0.0,
                "_meta": {"retrieved_docs": []},
            }
        _show_retrieval_chain(out)
        _stream_fields(out)
        _append_audit_trail(row, out)

        enriched = preprocess(row)
        confidence = float(out.get("confidence", 0.0) or 0.0)
        ambiguous = str(enriched.get("detected_company", "unknown")) == "unknown"
        if confidence < conf_threshold or ambiguous:
            q = _clarifying_question(row)
            print("\nClarification Needed")
            print(f"Question: {q}")
            answer = input("Your answer: ").strip()
            if answer:
                clarified = {
                    "company": company,
                    "subject": subject,
                    "issue": f"{issue}\nClarification from user: {answer}",
                }
                try:
                    out2 = triage_ticket(clarified, row_id=None)
                except Exception as exc:
                    out2 = {
                        "status": "escalated",
                        "product_area": "General Support",
                        "response": "Your request has been escalated for manual review.",
                        "justification": f"Automatic processing failed in clarification turn: {type(exc).__name__}",
                        "request_type": "product_issue",
                        "confidence": 0.0,
                        "_meta": {"retrieved_docs": []},
                    }
                _show_retrieval_chain(out2)
                _stream_fields(out2)
                _append_audit_trail(clarified, out2)

        print("\n" + "=" * 60 + "\n")


def main() -> None:
    args = _parse_args()
    if args.interactive:
        threshold = max(0.0, min(1.0, float(args.confidence_threshold)))
        _interactive_mode(threshold)
        return

    start = time.time()
    _check_corpus_freshness()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if args.dry_run is not None and args.dry_run >= 0:
        df = df.head(args.dry_run)

    total = len(df)
    print(f"HackerRank Orchestrate Support Agent | Tickets to process: {total}")
    run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    with RUN_LOG_PATH.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"event": "run_start", "started_at": run_started_at, "total_tickets": total}) + "\n")

    # Warm retriever and ensure corpus has been loaded before loop.
    _ = retrieve("warmup", None, top_k=1)

    results = []
    llm_fallback_rows: list[int] = []
    for idx, row in tqdm(df.iterrows(), total=total, desc="Processing tickets"):
        row_dict = _normalize_row(row)
        try:
            out = triage_ticket(row_dict, row_id=idx + 1, force_fallback=bool(args.fallback_only))
        except Exception as exc:
            out = {
                "status": "escalated",
                "product_area": "General Support",
                "response": "Your request has been escalated for manual review.",
                "justification": f"Automatic processing failed for this row: {type(exc).__name__}",
                "request_type": "product_issue",
                "confidence": 0.0,
                "domains_involved": [],
                "_meta": {
                    "row_id": idx + 1,
                    "company": str(row_dict.get("company", "unknown")),
                    "confidence_score": 0.0,
                    "chunks_used": [],
                    "escalation_reason": "corpus_insufficient",
                },
            }

        meta = out.get("_meta", {})
        if bool(meta.get("llm_unavailable", False)):
            llm_fallback_rows.append(idx + 1)
        log_row = {
            "row_id": meta.get("row_id", idx + 1),
            "company": meta.get("company", str(row_dict.get("company", "unknown")).lower()),
            "status": out.get("status", "escalated"),
            "confidence": meta.get("confidence_score", 0.0),
            "chunks_used": meta.get("chunks_used", []),
            "escalation_reason": meta.get("escalation_reason", ""),
            "llm_unavailable": bool(meta.get("llm_unavailable", False)),
        }
        with RUN_LOG_PATH.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
        _append_audit_trail(row_dict, out)

        results.append({k: out.get(k, "") for k in OUTPUT_COLUMNS})
        company = str(row_dict.get("company", "") or "unknown")
        print(f"Row {len(results)}/{total} | {company} | {out.get('status')} | {out.get('request_type')}")

    if args.require_llm and llm_fallback_rows:
        preview = ",".join(str(x) for x in llm_fallback_rows[:10])
        raise RuntimeError(
            f"LLM-required mode failed: fallback used for {len(llm_fallback_rows)} rows "
            f"(example rows: {preview}). NVIDIA endpoint is unreachable or key is invalid."
        )

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
    _print_batch_dashboard(out_df, elapsed)


if __name__ == "__main__":
    main()
