"""Load feature tables from CSV or JSON Lines and type each column.

A column is numeric when every present value parses as a finite number —
one "N/A" string among a million floats does not demote the column, it
counts as missing. Anything else is categorical. The rule is deliberately
that simple, and the report says which kind each feature was read as, so a
surprising typing is visible rather than silent.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .errors import DataFormatError
from .types import FeatureColumn, FeatureKind

#: Strings treated as missing, matched case-insensitively after stripping.
MISSING = frozenset({"", "na", "n/a", "nan", "null", "none"})


def load(path: str | Path) -> dict[str, FeatureColumn]:
    """Columns by name, from a .csv or .jsonl/.ndjson file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _csv_rows(path)
    elif suffix in (".jsonl", ".ndjson"):
        rows = _jsonl_rows(path)
    else:
        raise DataFormatError(
            f"unsupported input {path.name!r}: expected .csv, .jsonl or .ndjson"
        )
    if not rows:
        raise DataFormatError(f"{path.name}: no data rows")
    return columns_from_rows(rows)


def columns_from_rows(rows: list[dict[str, str]]) -> dict[str, FeatureColumn]:
    names: dict[str, None] = {}
    for row in rows:
        for name in row:
            names.setdefault(name, None)

    out: dict[str, FeatureColumn] = {}
    for name in names:
        raw = [row.get(name, "") for row in rows]
        present = [value.strip() for value in raw]
        missing = sum(1 for value in present if value.casefold() in MISSING)
        values = [value for value in present if value.casefold() not in MISSING]

        numbers = _as_numbers(values)
        if numbers is not None:
            out[name] = FeatureColumn(
                name=name, kind=FeatureKind.NUMERIC,
                numbers=tuple(numbers), missing=missing,
            )
        else:
            out[name] = FeatureColumn(
                name=name, kind=FeatureKind.CATEGORICAL,
                categories=tuple(values), missing=missing,
            )
    return out


def _as_numbers(values: list[str]) -> list[float] | None:
    if not values:
        return None
    numbers = []
    for value in values:
        try:
            number = float(value)
        except ValueError:
            return None
        if not math.isfinite(number):
            return None
        numbers.append(number)
    return numbers


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DataFormatError(f"{path.name}: file is empty")
            return [
                {k: (v if v is not None else "") for k, v in row.items() if k}
                for row in reader
            ]
    except OSError as err:
        raise DataFormatError(f"cannot read {path}: {err.strerror or err}") from err


def _jsonl_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as err:
                    raise DataFormatError(
                        f"{path.name}:{number}: invalid JSON: {err.msg}"
                    ) from err
                if not isinstance(obj, dict):
                    raise DataFormatError(
                        f"{path.name}:{number}: expected an object per line"
                    )
                rows.append({str(k): _scalar(v) for k, v in obj.items()})
    except OSError as err:
        raise DataFormatError(f"cannot read {path}: {err.strerror or err}") from err
    return rows


def _scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    # Containers are not feature values; stringifying them would invent
    # categories the data does not contain.
    raise DataFormatError("nested values are not supported as features")
