from __future__ import annotations

import json
from pathlib import Path

from feature_drift import report
from feature_drift.types import Verdict


def write(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def stable_pair(tmp_path: Path) -> tuple[str, str]:
    body = "x,c\n" + "".join(f"{i % 7},{'a' if i % 3 else 'b'}\n" for i in range(300))
    return (
        write(tmp_path / "train.csv", body),
        write(tmp_path / "serve.csv", body),
    )


def drifted_pair(tmp_path: Path) -> tuple[str, str]:
    train = "x,c\n" + "".join(f"{i % 7},{'a' if i % 3 else 'b'}\n" for i in range(300))
    serve = "x,c\n" + "".join(f"{i % 7 + 40},NEW\n" for i in range(300))
    return (
        write(tmp_path / "train.csv", train),
        write(tmp_path / "serve.csv", serve),
    )


class TestRun:
    def test_stable_pair_has_stable_worst_verdict(self, tmp_path: Path) -> None:
        result = report.run(*stable_pair(tmp_path))
        assert result.worst_verdict() is Verdict.STABLE
        assert result.only_train == [] and result.only_serve == []

    def test_drifted_pair_is_severe(self, tmp_path: Path) -> None:
        result = report.run(*drifted_pair(tmp_path))
        assert result.worst_verdict() is Verdict.SEVERE

    def test_schema_change_alone_is_severe(self, tmp_path: Path) -> None:
        train = write(tmp_path / "t.csv", "a,b\n1,2\n")
        serve = write(tmp_path / "s.csv", "a\n1\n")
        assert report.run(train, serve).worst_verdict() is Verdict.SEVERE


class TestRenderJson:
    def test_documented_top_level_keys_and_summary(self, tmp_path: Path) -> None:
        payload = json.loads(report.render_json(report.run(*drifted_pair(tmp_path))))
        assert set(payload) == {
            "train", "serve", "features", "only_in_training",
            "only_in_serving", "summary",
        }
        assert payload["summary"]["severe"] >= 1

    def test_features_carry_metrics_by_name(self, tmp_path: Path) -> None:
        payload = json.loads(report.render_json(report.run(*drifted_pair(tmp_path))))
        by_name = {f["feature"]: f for f in payload["features"]}
        assert by_name["c"]["unseen_categories"] == ["NEW"]
        assert by_name["x"]["psi"] > 0.25


class TestRenderText:
    def test_stable_run_says_so(self, tmp_path: Path) -> None:
        text = report.render_text(report.run(*stable_pair(tmp_path)))
        assert "no drift" in text

    def test_drifted_run_lists_metrics_and_notes(self, tmp_path: Path) -> None:
        text = report.render_text(report.run(*drifted_pair(tmp_path)))
        assert "SEVERE" in text
        assert "PSI=" in text
        assert "never seen in training" in text

    def test_schema_changes_render_before_features(self, tmp_path: Path) -> None:
        train = write(tmp_path / "t.csv", "a,b\n1,2\n")
        serve = write(tmp_path / "s.csv", "a,c\n1,3\n")
        text = report.render_text(report.run(train, serve))
        assert "never arrive in serving: b" in text
        assert "never trained on: c" in text
