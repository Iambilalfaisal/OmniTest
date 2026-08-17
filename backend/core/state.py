"""QAState: the shared LangGraph state. `test_results` uses `operator.add` so every
parallel `worker_node` branch appends its own TestResult instead of overwriting
the others.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from .models import TestCase, TestResult


class QAState(TypedDict):
    target_url: str
    instruction: str
    test_cases: list[TestCase]
    test_results: Annotated[list[TestResult], operator.add]
    summary: dict


class WorkerState(TypedDict):
    """State handed to a single `worker_node` branch via `Send` — not the full QAState."""

    target_url: str
    test_case: TestCase
