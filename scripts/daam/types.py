from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedDaamFlags:
    enable_daam: bool
    enable_time_focus: bool
    time_focus: str
    enable_diagnostics: bool
    influence_mode: str
    remaining_args: list[Any]

    def to_tuple(self):
        return (
            self.enable_daam,
            self.enable_time_focus,
            self.time_focus,
            self.enable_diagnostics,
            self.influence_mode,
            self.remaining_args,
        )
