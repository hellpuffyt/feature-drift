"""Errors the CLI reports as exit code 1."""

from __future__ import annotations


class FeatureDriftError(Exception):
    """Base class for user-facing input/usage failures."""


class DataFormatError(FeatureDriftError):
    """An input file could not be understood as a feature table."""
