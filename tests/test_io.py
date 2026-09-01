from __future__ import annotations

import json
from pathlib import Path

import pytest

from feature_drift import io
from feature_drift.errors import DataFormatError
from feature_drift.types import FeatureKind


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestTyping:
    def test_all_numeric_column_is_numeric(self, tmp_path: Path) -> None:
        cols = io.load(write(tmp_path / "d.csv", "x\n1\n2.5\n-3e2\n"))
        assert cols["x"].kind is FeatureKind.NUMERIC
        assert cols["x"].numbers == (1.0, 2.5, -300.0)

    def test_one_word_demotes_nothing_it_is_categorical(self, tmp_path: Path) -> None:
        cols = io.load(write(tmp_path / "d.csv", "x\n1\ntwo\n3\n"))
        assert cols["x"].kind is FeatureKind.CATEGORICAL
        assert cols["x"].categories == ("1", "two", "3")

    def test_missing_markers_count_as_missing_not_values(self, tmp_path: Path) -> None:
        # A quoted empty field is a real row; a fully blank line is not a
        # CSV record at all and the csv module rightly skips it.
        cols = io.load(write(tmp_path / "d.csv", 'x\n1\nNA\n""\nnull\n4\n'))
        column = cols["x"]
        assert column.kind is FeatureKind.NUMERIC
        assert column.numbers == (1.0, 4.0)
        assert column.missing == 3
        assert column.missing_rate == pytest.approx(0.6)

    def test_all_missing_column_is_categorical_and_empty(self, tmp_path: Path) -> None:
        cols = io.load(write(tmp_path / "d.csv", "x\nNA\n\n"))
        assert cols["x"].kind is FeatureKind.CATEGORICAL
        assert cols["x"].present == 0

    def test_infinity_is_not_a_number_here(self, tmp_path: Path) -> None:
        cols = io.load(write(tmp_path / "d.csv", "x\n1\ninf\n"))
        assert cols["x"].kind is FeatureKind.CATEGORICAL


class TestJsonl:
    def test_loads_scalars_and_nulls(self, tmp_path: Path) -> None:
        lines = [
            json.dumps({"n": 1, "flag": True, "s": "web", "gone": None}),
            json.dumps({"n": 2.5, "flag": False, "s": "app", "gone": None}),
        ]
        cols = io.load(write(tmp_path / "d.jsonl", "\n".join(lines) + "\n"))
        assert cols["n"].kind is FeatureKind.NUMERIC
        assert cols["flag"].categories == ("true", "false")
        assert cols["gone"].missing == 2

    def test_ragged_objects_fill_missing(self, tmp_path: Path) -> None:
        lines = [json.dumps({"a": 1}), json.dumps({"a": 2, "b": "x"})]
        cols = io.load(write(tmp_path / "d.jsonl", "\n".join(lines) + "\n"))
        assert cols["b"].missing == 1
        assert cols["b"].categories == ("x",)

    def test_nested_values_are_refused(self, tmp_path: Path) -> None:
        line = json.dumps({"a": {"nested": 1}})
        with pytest.raises(DataFormatError, match="nested"):
            io.load(write(tmp_path / "d.jsonl", line + "\n"))

    def test_invalid_json_names_the_line(self, tmp_path: Path) -> None:
        with pytest.raises(DataFormatError, match=r"d\.jsonl:2"):
            io.load(write(tmp_path / "d.jsonl", '{"a": 1}\n{oops\n'))


class TestErrors:
    def test_unknown_extension_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DataFormatError, match="unsupported input"):
            io.load(write(tmp_path / "d.parquet", "x"))

    def test_empty_csv_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(DataFormatError, match="empty"):
            io.load(write(tmp_path / "d.csv", ""))

    def test_header_only_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(DataFormatError, match="no data rows"):
            io.load(write(tmp_path / "d.csv", "x\n"))

    def test_missing_file_reports_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(DataFormatError, match="cannot read"):
            io.load(tmp_path / "absent.csv")
