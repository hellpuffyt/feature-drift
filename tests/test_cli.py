from __future__ import annotations

import json
from pathlib import Path

import pytest

from feature_drift import __version__, cli


def write(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def stable(tmp_path: Path) -> tuple[str, str]:
    body = "x\n" + "".join(f"{i % 5}\n" for i in range(100))
    return write(tmp_path / "t.csv", body), write(tmp_path / "s.csv", body)


def severe(tmp_path: Path) -> tuple[str, str]:
    train = "x\n" + "".join(f"{i % 5}\n" for i in range(100))
    serve = "x\n" + "".join(f"{i % 5 + 50}\n" for i in range(100))
    return write(tmp_path / "t.csv", train), write(tmp_path / "s.csv", serve)


class TestExitCodes:
    def test_stable_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main([*stable(tmp_path)]) == cli.EXIT_OK

    def test_severe_drift_exits_three(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main([*severe(tmp_path)]) == cli.EXIT_DRIFT

    def test_fail_on_never_reports_but_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli.main([*severe(tmp_path), "--fail-on", "never"])
        assert code == cli.EXIT_OK
        assert "SEVERE" in capsys.readouterr().out

    def test_fail_on_drifting_is_stricter(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        train = "x\n" + "".join(f"{i % 100}\n" for i in range(2000))
        # Shifted 8% of the range: PSI in the drifting band, under severe.
        serve = "x\n" + "".join(f"{i % 100 + 8}\n" for i in range(2000))
        t = write(tmp_path / "t.csv", train)
        s = write(tmp_path / "s.csv", serve)
        assert cli.main([t, s]) == cli.EXIT_OK
        assert cli.main([t, s, "--fail-on", "drifting"]) == cli.EXIT_DRIFT

    def test_bad_input_exits_one_on_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        good = write(tmp_path / "t.csv", "x\n1\n")
        code = cli.main([good, str(tmp_path / "absent.csv")])
        captured = capsys.readouterr()
        assert code == cli.EXIT_USAGE
        assert captured.err.startswith("error:")


class TestOutput:
    def test_json_format(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main([*severe(tmp_path), "--format", "json", "--fail-on", "never"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["severe"] == 1

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--version"])
        assert excinfo.value.code == 0
        assert __version__ in capsys.readouterr().out


class TestModuleEntryPoint:
    def test_python_dash_m_invocation(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        done = subprocess.run(
            [sys.executable, "-m", "feature_drift", *stable(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0
        assert "no drift" in done.stdout
