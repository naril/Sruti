from __future__ import annotations


def s05_cleanup_prompt(text: str) -> str:
    return (
        "You are cleaning ASR output from an English lecture.\n"
        "Rules:\n"
        "- Fix obvious ASR mistakes only.\n"
        "- Keep meaning, facts, and terminology unchanged.\n"
        "- Remove accidental duplicate fragments.\n"
        "- Improve punctuation and sentence boundaries.\n"
        "- Do NOT summarize and do NOT add new facts.\n"
        "Return only cleaned text.\n\n"
        f"Input:\n{text}"
    )


def s06_classification_prompt(span_lines: str) -> str:
    return (
        "Classify each span as KEEP or REMOVE.\n"
        "REMOVE only non-lecture material like audience chatter, technical glitches, or logistics.\n"
        "Do not rewrite any text.\n"
        "Return strict JSON array with objects:\n"
        '{"span_id": <int>, "action": "KEEP|REMOVE", "label": "<category>", "reason": "<short>"}\n'
        "No markdown.\n\n"
        f"Spans:\n{span_lines}"
    )


def s06_repair_json_prompt(original_prompt: str, bad_response: str) -> str:
    return (
        "Fix the following response so it is valid JSON matching the required schema.\n"
        "Return JSON only.\n\n"
        f"Original prompt:\n{original_prompt}\n\n"
        f"Bad response:\n{bad_response}"
    )


def s07_editorial_prompt(text: str) -> str:
    return (
        "Edit the following English lecture text for publication quality.\n"
        "Rules:\n"
        "- Keep all meaning and terminology.\n"
        "- No summarization.\n"
        "- No compression.\n"
        "- Remove spoken redundancies only.\n"
        "- Do not add facts.\n"
        "Return only edited text.\n\n"
        f"Input:\n{text}"
    )


def s08_translate_prompt(text: str) -> str:
    return (
        "Translate English to Czech faithfully.\n"
        "Rules:\n"
        "- Preserve structure and terminology.\n"
        "- No added information.\n"
        "- No summarization.\n"
        "Return Czech text only.\n\n"
        f"Input:\n{text}"
    )


def s09_czech_editorial_prompt(text: str) -> str:
    return (
        "Improve Czech readability and style without changing meaning.\n"
        "Rules:\n"
        "- Preserve terminology.\n"
        "- No added facts.\n"
        "- No summarization.\n"
        "Return edited Czech text only.\n\n"
        f"Input:\n{text}"
    )
