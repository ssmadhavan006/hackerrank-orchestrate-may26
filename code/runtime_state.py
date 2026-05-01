from __future__ import annotations

from typing import List

_stale_domains: List[str] = []


def set_stale_domains(domains: List[str]) -> None:
    global _stale_domains
    _stale_domains = list(domains)


def get_stale_domains() -> List[str]:
    return list(_stale_domains)

