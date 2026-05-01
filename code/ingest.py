from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".json", ".csv"}
TARGET_WORDS = 400
OVERLAP_WORDS = 50
COMPANIES = ("hackerrank", "claude", "visa")


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return "\n".join(self.parts)


def _read_file(path: Path) -> str:
    content = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".html":
        parser = _HTMLTextExtractor()
        parser.feed(content)
        return parser.text()
    return content


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(text: str, target_words: int = TARGET_WORDS, overlap_words: int = OVERLAP_WORDS) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: List[str] = []

    for paragraph in paragraphs:
        words = _tokenize_words(paragraph)
        if len(words) <= target_words:
            units.append(paragraph)
        else:
            units.extend(_split_sentences(paragraph))

    chunks: List[str] = []
    i = 0
    while i < len(units):
        merged: List[str] = []
        word_count = 0
        j = i
        while j < len(units):
            candidate = units[j]
            c_words = len(_tokenize_words(candidate))
            if merged and word_count + c_words > target_words:
                break
            merged.append(candidate)
            word_count += c_words
            j += 1

        if not merged:
            merged = [units[i]]
            j = i + 1

        chunk_text = "\n\n".join(merged).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if j >= len(units):
            break

        overlap = overlap_words
        k = j - 1
        while k >= i and overlap > 0:
            overlap -= len(_tokenize_words(units[k]))
            k -= 1
        i = max(i + 1, k + 1)

    return chunks


def build_corpus() -> None:
    root = Path(__file__).resolve().parent.parent
    data_root = root / "data"
    output_path = root / "code" / "corpus_chunks.json"

    all_files: List[Path] = []
    for company in COMPANIES:
        company_root = data_root / company
        if not company_root.exists():
            continue
        all_files.extend([p for p in company_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])

    corpus: List[Dict[str, object]] = []
    file_count = 0
    company_file_count = {c: 0 for c in COMPANIES}
    company_chunk_count = {c: 0 for c in COMPANIES}

    for file_path in tqdm(all_files, desc="Processing files"):
        rel = file_path.relative_to(root).as_posix()
        company = rel.split("/")[1] if rel.startswith("data/") else "unknown"
        if company not in COMPANIES:
            continue

        text = _read_file(file_path).strip()
        if not text:
            continue

        chunks = _chunk_text(text)
        if not chunks:
            continue

        company_file_count[company] += 1
        file_count += 1

        for chunk_id, chunk_text in enumerate(chunks):
            corpus.append(
                {
                    "source_file": rel,
                    "company": company,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
            )
            company_chunk_count[company] += 1

    output_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(f"Files processed: {file_count}")
    print(f"Chunks created: {len(corpus)}")
    print("Per company:")
    for company in COMPANIES:
        print(f"  - {company}: files={company_file_count[company]}, chunks={company_chunk_count[company]}")


if __name__ == "__main__":
    build_corpus()

