"""Console entry point. Exit codes are the CI/monitoring contract:

* 0 — compared, nothing at or above ``--fail-on``
* 1 — usage or input error
* 3 — drift at or above ``--fail-on``
"""

from __future__ import annotations

import argparse
import sys

from . import __version__, report
from .errors import FeatureDriftError
from .types import Verdict

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_DRIFT = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feature-drift",
        description=(
            "Detect feature drift between training and serving data and "
            "report which features moved and by how much."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("train", help="training/reference table (.csv, .jsonl)")
    parser.add_argument("serve", help="serving/current table (.csv, .jsonl)")
    parser.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="output format (default: text)",
    )
    parser.add_argument(
        "--fail-on", choices=("severe", "drifting", "never"), default="severe",
        help="exit 3 when any feature reaches this verdict (default: severe)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = report.run(args.train, args.serve)
    except FeatureDriftError as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_USAGE

    print(
        report.render_json(result)
        if args.format == "json"
        else report.render_text(result)
    )

    if args.fail_on == "never":
        return EXIT_OK
    threshold = Verdict.SEVERE if args.fail_on == "severe" else Verdict.DRIFTING
    if result.worst_verdict().rank >= threshold.rank:
        return EXIT_DRIFT
    return EXIT_OK
