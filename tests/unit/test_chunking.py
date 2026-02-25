from __future__ import annotations

from sruti.llm.chunking import chunk_text


def test_chunk_text_splits_by_budget_and_preserves_text() -> None:
    text = "A\n\nB paragraph\n\nC longer paragraph"
    chunks = chunk_text(text, max_chars=12)
    assert len(chunks) == 4
    assert all(len(chunk) <= 12 for chunk in chunks)
    joined = " ".join(" ".join(chunks).split())
    baseline = " ".join(text.replace("\n\n", " ").split())
    assert joined == baseline


def test_chunk_text_splits_oversized_paragraph() -> None:
    text = "word " * 80
    chunks = chunk_text(text.strip(), max_chars=50)
    assert chunks
    assert all(len(chunk) <= 50 for chunk in chunks)
