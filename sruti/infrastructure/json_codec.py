from __future__ import annotations

import json
from typing import Any

try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    orjson = None


def dumps(data: Any, *, indent: int | None = None) -> str:
    if orjson is not None:
        option = 0
        if indent:
            option |= orjson.OPT_INDENT_2
        return orjson.dumps(data, option=option).decode("utf-8")
    return json.dumps(data, ensure_ascii=False, indent=indent)


def loads(value: str) -> Any:
    if orjson is not None:
        return orjson.loads(value)
    return json.loads(value)
