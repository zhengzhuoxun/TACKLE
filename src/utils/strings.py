"""Lightweight string utilities for semantic name/description matching.

These are deterministic, offline helpers used by the KGC pipeline to compare
column/property names and descriptions without any LLM or embedding calls.
"""

from __future__ import annotations

import re

# Moderate abbreviation expansion (kept literal for descriptions).
ABBREVIATIONS: dict[str, str] = {
    "dept": "department",
    "dpt": "department",
    "stu": "student",
    "stud": "student",
    "addr": "address",
    "loc": "location",
    "desc": "description",
    "dob": "birth",
    "yob": "birth",
    "tel": "telephone",
    "phone": "telephone",
    "mob": "mobile",
    "amt": "amount",
    "yrs": "years",
}

# Tokens that all denote an identifier-like concept. During *name* matching
# these collapse to one canonical token so "stu_nr" and "studentID" compare
# equal.
IDENTIFIER_TOKENS: frozenset[str] = frozenset({
    "id", "ids", "identifier", "nr", "no", "num", "number", "nbr",
    "key", "code", "ref", "reference",
})
CANONICAL_IDENTIFIER = "identifier"

# Words that carry little semantic weight when comparing names/descriptions.
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "this", "that", "these",
    "those", "each", "every", "per", "with", "which", "what", "who",
    "has", "have", "had", "by", "as", "from", "its", "it",
})


def _split_camel(text: str) -> str:
    """Insert spaces at camelCase boundaries: 'studentID' -> 'student ID'."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return text


def _raw_tokens(text: str) -> list[str]:
    """Lower-case, split, and strip stopwords — no synonym expansion."""
    if not text:
        return []
    s = _split_camel(str(text))
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s).lower()
    return [tok for tok in s.split() if tok not in STOPWORDS]


def tokenize(text: str) -> list[str]:
    """Normalise text into canonical tokens (abbreviation expansion only)."""
    return [ABBREVIATIONS.get(tok, tok) for tok in _raw_tokens(text)]


def tokenize_name(text: str) -> list[str]:
    """Normalise a column/property *name*, collapsing identifier tokens."""
    tokens: list[str] = []
    for tok in _raw_tokens(text):
        tok = ABBREVIATIONS.get(tok, tok)
        if tok in IDENTIFIER_TOKENS:
            tok = CANONICAL_IDENTIFIER
        tokens.append(tok)
    return tokens


def snake_case(text: str) -> str:
    """Convert any identifier-ish text into snake_case."""
    if not text:
        return ""
    s = _split_camel(str(text))
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).lower().strip("_")
    return s


def _jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _token_match(a: str, b: str) -> bool:
    """Exact token match, or prefix match for tokens of length >= 3."""
    if a == b:
        return True
    if len(a) >= 3 and len(b) >= 3:
        return a.startswith(b) or b.startswith(a)
    return False


def name_similarity(left: str, right: str) -> float:
    """Score how similar two column/property *names* are (0..1)."""
    a = tokenize_name(left)
    b = tokenize_name(right)
    if not a or not b:
        return 0.0

    matched = 0
    used: set[int] = set()
    for ta in a:
        for j, tb in enumerate(b):
            if j in used:
                continue
            if _token_match(ta, tb):
                matched += 1
                used.add(j)
                break

    score = matched / min(len(a), len(b))
    return round(min(1.0, score), 4)


def description_similarity(left: str, right: str) -> float:
    """Score how similar two *descriptions* are (0..1)."""
    return round(_jaccard(tokenize(left), tokenize(right)), 4)


def strip_table_prefixes(name: str, table_names: list[str]) -> str:
    """Remove leading table-name tokens from a canonical property name.

    e.g. 'dept_department_id' with table 'dept_registry' -> 'department_id'.
    """
    tokens = tokenize(name)
    table_tokens: set[str] = set()
    for table_name in table_names:
        table_tokens.update(tokenize(table_name))
    while tokens and tokens[0] in table_tokens:
        tokens.pop(0)
    return "_".join(tokens) if tokens else snake_case(name)
