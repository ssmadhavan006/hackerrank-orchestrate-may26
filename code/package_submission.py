from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def _active_ticket_dir(root: Path) -> Path:
    tickets = root / "support_tickets"
    issues = root / "support_issues"
    return tickets if tickets.exists() else issues


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    code_dir = root / "code"
    ticket_dir = _active_ticket_dir(root)
    artifacts_dir = root / "submission_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    code_zip = artifacts_dir / "code_submission.zip"
    temp_zip = artifacts_dir / "code_submission.tmp.zip"
    output_csv = ticket_dir / "output.csv"
    pred_copy = artifacts_dir / "predictions_output.csv"
    chat_log = Path.home() / "hackerrank_orchestrate" / "log.txt"
    chat_copy = artifacts_dir / "chat_log.txt"

    excluded_names = {
        ".env",
        "__pycache__",
        "decision_cache.json",
    }
    excluded_suffixes = {".pyc"}

    if temp_zip.exists():
        temp_zip.unlink()
    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in code_dir.rglob("*"):
            rel = p.relative_to(code_dir)
            if any(part in excluded_names for part in rel.parts):
                continue
            if p.suffix in excluded_suffixes:
                continue
            if p.is_file():
                zf.write(p, arcname=str(rel))
    temp_zip.replace(code_zip)

    if output_csv.exists():
        shutil.copy2(output_csv, pred_copy)
    if chat_log.exists():
        shutil.copy2(chat_log, chat_copy)

    print(f"Created: {code_zip}")
    print(f"Copied predictions: {pred_copy if pred_copy.exists() else 'missing output.csv'}")
    print(f"Copied chat log: {chat_copy if chat_copy.exists() else 'missing log.txt'}")


if __name__ == "__main__":
    main()
