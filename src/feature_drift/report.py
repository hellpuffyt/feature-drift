"""Assemble and render the drift report."""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import drift, io
from .types import FeatureDrift, Verdict


@dataclass(frozen=True)
class Report:
    train_path: str
    serve_path: str
    features: list[FeatureDrift]
    only_train: list[str]
    only_serve: list[str]

    def worst_verdict(self) -> Verdict:
        worst = Verdict.STABLE
        for feature in self.features:
            if feature.verdict.rank > worst.rank:
                worst = feature.verdict
        if self.only_train or self.only_serve:
            # A schema change outranks a stable distribution: the model is
            # being fed different columns than it was trained on.
            worst = max(worst, Verdict.SEVERE, key=lambda v: v.rank)
        return worst

    def to_dict(self) -> dict[str, object]:
        return {
            "train": self.train_path,
            "serve": self.serve_path,
            "features": [f.to_dict() for f in self.features],
            "only_in_training": self.only_train,
            "only_in_serving": self.only_serve,
            "summary": {
                verdict.value: sum(
                    1 for f in self.features if f.verdict is verdict
                )
                for verdict in Verdict
            },
        }


def run(train_path: str, serve_path: str) -> Report:
    train = io.load(train_path)
    serve = io.load(serve_path)
    features, only_train, only_serve = drift.compare(train, serve)
    return Report(
        train_path=train_path,
        serve_path=serve_path,
        features=features,
        only_train=only_train,
        only_serve=only_serve,
    )


def render_json(report: Report) -> str:
    return json.dumps(report.to_dict(), indent=2)


_MARK = {Verdict.SEVERE: "SEVERE ", Verdict.DRIFTING: "drifting", Verdict.STABLE: "stable "}


def render_text(report: Report) -> str:
    lines = [f"training: {report.train_path}", f"serving:  {report.serve_path}", ""]

    if report.only_train:
        lines.append(
            f"SEVERE  schema: {len(report.only_train)} feature(s) in training "
            f"never arrive in serving: {', '.join(report.only_train)}"
        )
    if report.only_serve:
        lines.append(
            f"SEVERE  schema: {len(report.only_serve)} serving feature(s) the "
            f"model never trained on: {', '.join(report.only_serve)}"
        )
    if report.only_train or report.only_serve:
        lines.append("")

    for feature in report.features:
        lines.append(
            f"{_MARK[feature.verdict]} {feature.name}  "
            f"PSI={feature.psi:.3f}  distance={feature.distance:.3f}"
        )
        for note in feature.notes:
            lines.append(f"          {note}")

    moved = [f for f in report.features if f.verdict is not Verdict.STABLE]
    lines.append("")
    if not moved and not report.only_train and not report.only_serve:
        lines.append("no drift: every shared feature is stable at the PSI thresholds")
    else:
        lines.append(
            f"{len(moved)} of {len(report.features)} shared feature(s) moved"
        )
    return "\n".join(lines)
