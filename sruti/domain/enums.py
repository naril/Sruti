from __future__ import annotations

from enum import Enum


class StageId(str, Enum):
    S01 = "s01"
    S02 = "s02"
    S03 = "s03"
    S04 = "s04"
    S05 = "s05"
    S06 = "s06"
    S07 = "s07"
    S08 = "s08"
    S09 = "s09"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    DRY_RUN = "dry_run"


class OnExistsMode(str, Enum):
    ASK = "ask"
    SKIP = "skip"
    OVERWRITE = "overwrite"
    FAIL = "fail"
