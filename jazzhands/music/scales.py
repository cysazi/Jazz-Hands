"""Built-in and editable scale definitions."""

from __future__ import annotations

import json
from pathlib import Path


NOTE_TO_SEMITONE = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}
SEMITONE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")

BUILTIN_SCALE_INTERVALS = {
    "chromatic": tuple(range(12)),
    "major": (0, 2, 4, 5, 7, 9, 11),
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "natural-minor": (0, 2, 3, 5, 7, 8, 10),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "harmonic-minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic-minor": (0, 2, 3, 5, 7, 9, 11),
    "harmonic-major": (0, 2, 4, 5, 7, 8, 11),
    "double-harmonic": (0, 1, 4, 5, 7, 8, 11),
    "hungarian-minor": (0, 2, 3, 6, 7, 8, 11),
    "ukrainian-dorian": (0, 2, 3, 6, 7, 9, 10),
    "neapolitan-minor": (0, 1, 3, 5, 7, 8, 11),
    "neapolitan-major": (0, 1, 3, 5, 7, 9, 11),
    "enigmatic": (0, 1, 4, 6, 8, 10, 11),
    "persian": (0, 1, 4, 5, 6, 8, 11),
    "romanian-minor": (0, 2, 3, 6, 7, 9, 10),
    "spanish-gypsy": (0, 1, 4, 5, 7, 8, 10),
    "altered": (0, 1, 3, 4, 6, 8, 10),
    "whole-tone-plus": (0, 2, 4, 6, 8, 10, 11),
}

PRESET_PATH = Path(__file__).with_name("scale_presets.json")


def normalize_scale_key(name: str) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def _coerce_intervals(name: str, intervals: object) -> tuple[int, ...]:
    if not isinstance(intervals, (list, tuple)):
        raise ValueError(f"{name}: intervals must be a list")
    values = tuple(int(value) for value in intervals)
    if not values:
        raise ValueError(f"{name}: intervals cannot be empty")
    if any(value < -48 or value > 48 for value in values):
        raise ValueError(f"{name}: intervals must be between -48 and 48")
    return values


def load_scale_presets(path: Path = PRESET_PATH) -> dict[str, tuple[int, ...]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"[Jazz Hands scales] Could not read {path}: {error}")
        return {}
    if not isinstance(payload, dict):
        print(f"[Jazz Hands scales] Ignoring {path}: top-level value must be an object")
        return {}

    presets: dict[str, tuple[int, ...]] = {}
    for raw_name, raw_definition in payload.items():
        name = normalize_scale_key(raw_name)
        try:
            if isinstance(raw_definition, dict):
                intervals = _coerce_intervals(name, raw_definition.get("intervals"))
            else:
                intervals = _coerce_intervals(name, raw_definition)
        except Exception as error:
            print(f"[Jazz Hands scales] Ignoring preset {raw_name!r}: {error}")
            continue
        presets[name] = intervals
    return presets


SCALE_INTERVALS = {
    **BUILTIN_SCALE_INTERVALS,
    **load_scale_presets(),
}
