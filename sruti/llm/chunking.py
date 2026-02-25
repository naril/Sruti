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
        if not para:
            continue
        for part in _split_oversize_paragraph(para, max_chars=max_chars):
            size = len(part) + (2 if current else 0)
            if current and current_size + size > max_chars:
                chunks.append("\n\n".join(current))
                current = [part]
                current_size = len(part)
                continue
            current.append(part)
            current_size += size
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_oversize_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    pieces: list[str] = []
    current = ""
    for word in paragraph.split(" "):
        if not current:
            if len(word) <= max_chars:
                current = word
                continue
            pieces.extend(_split_hard(word, max_chars=max_chars))
            continue

        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        pieces.append(current)
        if len(word) <= max_chars:
            current = word
        else:
            pieces.extend(_split_hard(word, max_chars=max_chars))
            current = ""
    if current:
        pieces.append(current)
    return pieces


def _split_hard(value: str, *, max_chars: int) -> list[str]:
    return [value[idx : idx + max_chars] for idx in range(0, len(value), max_chars)]
