from __future__ import annotations

from pathlib import Path

import pandas as pd


def _norm(s: object) -> str:
    return str(s or "").strip().lower()


def _find_col(df: pd.DataFrame, *candidates: str) -> str:
    lowered = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    raise KeyError(f"Could not find any of columns: {candidates}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    tickets_dir = root / "support_tickets"
    issues_dir = root / "support_issues"
    active_dir = tickets_dir if tickets_dir.exists() else issues_dir
    sample_path = active_dir / "sample_support_tickets.csv"
    pred_path = active_dir / "output_sample.csv"

    gold = pd.read_csv(sample_path)
    pred = pd.read_csv(pred_path)

    n = min(len(gold), len(pred))
    gold = gold.head(n).copy()
    pred = pred.head(n).copy()

    gold_status_col = _find_col(gold, "Status")
    gold_req_col = _find_col(gold, "Request Type", "Request_Type", "request_type")
    pred_status_col = _find_col(pred, "status")
    pred_req_col = _find_col(pred, "request_type")

    status_matches = []
    req_matches = []
    mismatches = []

    for i in range(n):
        gs = _norm(gold.iloc[i][gold_status_col])
        ps = _norm(pred.iloc[i][pred_status_col])
        gr = _norm(gold.iloc[i][gold_req_col])
        pr = _norm(pred.iloc[i][pred_req_col])

        status_ok = gs == ps
        req_ok = gr == pr
        status_matches.append(status_ok)
        req_matches.append(req_ok)

        if not (status_ok and req_ok):
            mismatches.append(
                {
                    "row_id": i + 1,
                    "expected_status": gs,
                    "pred_status": ps,
                    "expected_request_type": gr,
                    "pred_request_type": pr,
                }
            )

    status_acc = (sum(status_matches) / n * 100.0) if n else 0.0
    req_acc = (sum(req_matches) / n * 100.0) if n else 0.0

    print("Evaluation Report")
    print(f"- Rows compared: {n}")
    print(f"- Status accuracy: {status_acc:.2f}%")
    print(f"- Request type accuracy: {req_acc:.2f}%")

    if mismatches:
        print("\nMismatches:")
        for m in mismatches:
            print(
                f"  Row {m['row_id']}: status expected={m['expected_status']} pred={m['pred_status']} | "
                f"request_type expected={m['expected_request_type']} pred={m['pred_request_type']}"
            )
    else:
        print("\nNo mismatches.")


if __name__ == "__main__":
    main()
