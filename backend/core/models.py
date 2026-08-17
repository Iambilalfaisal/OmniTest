"""Pydantic schemas for the planner's structured output and each worker's result."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    test_id: str
    goal: str
    steps: list[str] = Field(description="Ordered, plain-language instructions for the worker to execute")


class TestPlan(BaseModel):
    """Structured-output contract the planner LLM call is constrained to."""

    test_cases: list[TestCase]


class TestResult(BaseModel):
    test_id: str
    status: str = Field(description="Strictly 'Pass' or 'Fail'")
    screenshot_path: str
    trace_path: str | None
    video_path: str | None
    reason: str
