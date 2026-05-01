from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi


# Query expansion synonyms for better recall
QUERY_EXPANSIONS = {
    "minimum spend": ["minimum purchase", "minimum transaction", "minimum amount", "minimum charge"],
    "merchant": ["seller", "vendor", "store", "business"],
    "refund": ["money back", "return payment", "reversal"],
    "not working": ["broken", "failing", "error", "down", "issue", "problem"],
    "delete": ["remove", "cancel", "close"],
    "blocked": ["frozen", "locked", "disabled"],
    "lost access": ["cannot login", "locked out", "removed", "no access"],
    "submission": ["submit", "code submission", "test submission"],
    "assessment": ["test", "exam", "challenge", "coding test"],
    "workspace": ["team", "project", "account"],
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _expand_query(query: str) -> str:
    """Expand query with synonyms for better retrieval recall."""
    expanded = query.lower()
    for term, synonyms in QUERY_EXPANSIONS.items():
        if term in expanded:
            for synonym in synonyms:
                if synonym not in expanded:
                    expanded += " " + synonym
    return expanded


class _CorpusRetriever:
    def __init__(self) -> None:
        corpus_path = Path(__file__).resolve().parent / "corpus_chunks.json"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}. Run `python ingest.py` first.")

        self.chunks: List[Dict] = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.all_docs = [c.get("text", "") for c in self.chunks]
        self.all_tokens = [_tokenize(doc) for doc in self.all_docs]
        self.all_bm25 = BM25Okapi(self.all_tokens) if self.all_tokens else None

        self.company_indexes: Dict[str, List[int]] = {}
        self.company_bm25: Dict[str, BM25Okapi] = {}
        for idx, chunk in enumerate(self.chunks):
            company = str(chunk.get("company", ""))
            self.company_indexes.setdefault(company, []).append(idx)

        for company, indexes in self.company_indexes.items():
            tokens = [self.all_tokens[i] for i in indexes]
            if tokens:
                self.company_bm25[company] = BM25Okapi(tokens)

    def _keyword_overlap_score(self, product_area_hint: str, chunk_text: str) -> int:
        if not product_area_hint:
            return 0
        hint_tokens = set(_tokenize(product_area_hint))
        chunk_tokens = set(_tokenize(chunk_text))
        return len(hint_tokens.intersection(chunk_tokens))

    def retrieve(
        self,
        query: str,
        company_filter: Optional[str],
        top_k: int,
        product_area_hint: str | None = None,
        first_pass_k: int = 12,
    ) -> List[Dict]:
        # Expand query with synonyms for better recall
        expanded_query = _expand_query(query)
        query_tokens = _tokenize(expanded_query)
        if not query_tokens:
            return []

        first_k = max(top_k, first_pass_k)
        hint = (product_area_hint or "").strip()

        if company_filter:
            company_filter = company_filter.strip().lower()
            indexes = self.company_indexes.get(company_filter, [])
            bm25 = self.company_bm25.get(company_filter)
            if not indexes or bm25 is None:
                return []
            scores = bm25.get_scores(query_tokens)
            ranked: List[Tuple[int, float]] = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            top = ranked[:first_k]
            candidates: List[Tuple[int, float, int]] = []
            for local_i, bm25_score in top:
                chunk = self.chunks[indexes[local_i]]
                overlap = self._keyword_overlap_score(hint, str(chunk.get("text", "")))
                candidates.append((local_i, bm25_score, overlap))
            reranked = sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True)[:top_k]
            results = [self.chunks[indexes[local_i]] for local_i, _, _ in reranked]
            
            # If we got few results, try without company filter
            if len(results) < top_k // 2:
                all_results = self._retrieve_all(query, top_k - len(results), product_area_hint, first_pass_k)
                results.extend([r for r in all_results if r not in results])
            
            return results[:top_k]

        if self.all_bm25 is None:
            return []
        scores = self.all_bm25.get_scores(query_tokens)
        ranked_all: List[Tuple[int, float]] = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_all = ranked_all[:first_k]
        candidates_all: List[Tuple[int, float, int]] = []
        for i, bm25_score in top_all:
            chunk = self.chunks[i]
            overlap = self._keyword_overlap_score(hint, str(chunk.get("text", "")))
            candidates_all.append((i, bm25_score, overlap))
        reranked_all = sorted(candidates_all, key=lambda x: (x[2], x[1]), reverse=True)[:top_k]
        return [self.chunks[i] for i, _, _ in reranked_all]
    
    def _retrieve_all(
        self,
        query: str,
        top_k: int,
        product_area_hint: str | None = None,
        first_pass_k: int = 12,
    ) -> List[Dict]:
        """Retrieve from entire corpus without company filter."""
        return self.retrieve(query, company_filter=None, top_k=top_k, 
                           product_area_hint=product_area_hint, first_pass_k=first_pass_k)


_RETRIEVER = _CorpusRetriever()


def retrieve(
    query: str,
    company_filter: str | None,
    top_k: int = 8,
    product_area_hint: str | None = None,
    first_pass_k: int = 12,
) -> List[Dict]:
    return _RETRIEVER.retrieve(
        query=query,
        company_filter=company_filter,
        top_k=top_k,
        product_area_hint=product_area_hint,
        first_pass_k=first_pass_k,
    )

