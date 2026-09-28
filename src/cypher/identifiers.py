"""Kuzu identifier rendering and validation helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable


_PLAIN_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Exact symbolic-name denylist from Kuzu v0.11.3's Cypher grammar: these are
# the keyword tokens that are not accepted by kU_NonReservedKeywords.
_KUZU_V0_11_3_RESERVED_SYMBOLIC_NAMES = frozenset(
    """
    ALL AND ASC ASCENDING CASE CAST COLUMN CREATE DBTYPE DEFAULT DESC DESCENDING
    DISTINCT ELSE END ENDS EXISTS FALSE GLOB GROUP HEADERS IN INSTALL MACRO NOT
    NULL ON ONLY OPTIONAL OR ORDER PRIMARY PROFILE SHORTEST STARTS TABLE THEN TRUE
    UNION UNWIND WHEN WHERE WITH XOR
    """.split()
)


def is_plain_identifier(value: str) -> bool:
    """Return whether a value has the lexical shape of a plain identifier."""
    return _PLAIN_IDENTIFIER_RE.fullmatch(str(value)) is not None


def quote_identifier(value: str) -> str:
    """Backtick-quote identifiers that are invalid or reserved in Kuzu 0.11.3."""
    text = str(value)
    if (
        is_plain_identifier(text)
        and text.upper() not in _KUZU_V0_11_3_RESERVED_SYMBOLIC_NAMES
    ):
        return text
    return "`" + text.replace("`", "``") + "`"


def identifiers_missing_required_quotes(
    query: str,
    identifiers: Iterable[str | None],
) -> list[str]:
    """Return used identifiers that an LLM query failed to backtick-quote.

    ``identifiers`` should contain schema names required by the grounded plan,
    rather than every word in the query. This keeps Cypher keywords such as the
    ``ORDER`` in ``ORDER BY`` from being mistaken for schema identifiers.
    """
    missing: list[str] = []
    seen: set[str] = set()
    for value in identifiers:
        if not value:
            continue
        name = str(value)
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)

        quoted = quote_identifier(name)
        if quoted == name:
            continue

        # A schema label/property is used after ':' or '.'. The delimiter check
        # avoids treating a longer identifier with the same prefix as a match.
        unquoted_pattern = re.compile(
            rf"[:.]\s*{re.escape(name)}(?=$|[^A-Za-z0-9_])",
            re.IGNORECASE,
        )
        quoted_pattern = re.compile(re.escape(quoted), re.IGNORECASE)
        if unquoted_pattern.search(query) or not quoted_pattern.search(query):
            missing.append(name)

    return missing
