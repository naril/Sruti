from __future__ import annotations


def chunk_text(text: str, *, max_chars: int) -> list[str]:
    """
    Deterministic paragraph-aware chunking by character budget.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for para in paragraphs:
        size = len(para) + (2 if current else 0)
        if current and current_size + size > max_chars:
            chunks.append("\n\n".join(current))
            current = [para]
            current_size = len(para)
            continue
        current.append(para)
        current_size += size
    if current:
        chunks.append("\n\n".join(current))
    return chunks
