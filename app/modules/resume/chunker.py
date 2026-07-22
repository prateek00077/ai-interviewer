"""Split parsed resume into embeddable chunks.

Chunks are cut along the section boundaries the parser already found, not on a
fixed character stride. A resume is a list of discrete items -- one job, one
degree, one project -- and a blind sliding window reliably splits a single role
across two chunks, so retrieving either gives the interviewer half a job.

Each chunk carries its section name as a prefix. The embedding then encodes
"this is work experience" rather than leaving the model to infer it from a bare
fragment, which is what makes an "experience" chunk rank above a "skills" chunk
for a question about past work.

Deterministic: the same text always yields the same chunks, so a re-run after a
failed embedding produces rows that line up with the ones already stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Sized for nv-embedqa-e5-v5's 512-token window, at a conservative ~3.5 chars per
# token. Overshooting silently truncates at the model, losing the tail of a role.
MAX_CHUNK_CHARS = 1400
# Below this a chunk is a heading fragment or a stray line: too little context to
# embed usefully, so it is merged into its neighbour instead.
MIN_CHUNK_CHARS = 80
# One resume should not produce an unbounded number of vectors. ~40 is the
# expected shape; this is the guard against a pathological document.
MAX_CHUNKS = 120

# Blank line, or a line starting a new dated entry ("2021 - 2023 Senior Engineer").
_ENTRY_BREAK = re.compile(r"\n\s*\n|\n(?=\s*(?:\d{4}|[A-Z][^\n]{0,60}\s+[-–—|]\s))")


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int
    section: str
    content: str


def _split_entries(body: str) -> list[str]:
    """One span per resume entry, before any size limits are applied."""
    return [part.strip() for part in _ENTRY_BREAK.split(body) if part and part.strip()]


def _split_oversized(text: str, limit: int) -> list[str]:
    """Break a too-long span on line boundaries, never mid-line.

    A single line longer than the limit is passed through whole rather than cut
    mid-sentence: the embedding model truncating it is a better outcome than
    storing half a sentence as if it were complete.
    """
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        # +1 for the newline that will rejoin them.
        if current and size + len(line) + 1 > limit:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        parts.append("\n".join(current))
    return parts


def _merge_small(spans: list[str], *, limit: int, floor: int) -> list[str]:
    """Fold undersized spans forward into the previous one where it fits."""
    merged: list[str] = []
    for span in spans:
        if merged and len(span) < floor and len(merged[-1]) + len(span) + 2 <= limit:
            merged[-1] = f"{merged[-1]}\n\n{span}"
        else:
            merged.append(span)
    return merged


def chunk_sections(
    sections: dict[str, str],
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chunks: int = MAX_CHUNKS,
) -> list[Chunk]:
    """Section map -> ordered, embeddable chunks."""
    chunks: list[Chunk] = []
    ordinal = 0

    for section, body in sections.items():
        if not body.strip():
            continue

        spans = _split_entries(body)
        sized = [part for span in spans for part in _split_oversized(span, max_chars)]
        # Merge after splitting: splitting can itself produce a small tail.
        for span in _merge_small(sized, limit=max_chars, floor=min_chars):
            if ordinal >= max_chunks:
                return chunks
            # The prefix is part of the embedded text, so retrieval can tell a
            # role apart from a skills list that mentions the same technology.
            chunks.append(
                Chunk(ordinal=ordinal, section=section, content=f"[{section}] {span}")
            )
            ordinal += 1

    return chunks
