"""Evaluation tasks. A task turns a SystemUnderTest plus a Dataset into numbers."""

from .text_retrieval import run as run_text_retrieval

__all__ = ["run_text_retrieval"]
