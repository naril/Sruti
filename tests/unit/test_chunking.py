from __future__ import annotations

from sruti.llm.chunking import chunk_text


def test_chunk_text_splits_by_budget_and_preserves_text() -> None:
    text = "A\n\nB paragraph\n\nC longer paragraph"
    chunks = chunk_text(text, max_chars=12)
    assert len(chunks) == 3
    assert "\n\n".join(chunks) == text
