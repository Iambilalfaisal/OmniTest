"""Parses a user's own plain-English test-case description(s) into this tool's
structured TestCase format for "My Own Plan" mode. Unlike planner_node/discovery_agent_node,
the tester has already decided what to test — this only reformats it, so it deliberately
skips both the site crawl those do and TEST_CASE_AUTHORING_GUIDELINES' coverage checklist
(core/models.py): that checklist exists to help a planner CHOOSE what to test, the opposite
of what this mode is for, and a crawl would burn exactly the extra call "My Own Plan" exists
to save over Explore/Quick.
"""
from __future__ import annotations

from ..core.llm import ModelRole, with_fallback
from ..core.models import TestPlan

CUSTOM_PLAN_PARSE_PROMPT = """You are converting a QA tester's own plain-English test-case \
descriptions into this tool's structured test-case format. These are cases the tester has \
ALREADY decided on — you are formatting them, not designing a plan. Never add a test case \
the tester didn't describe, and never invent element labels, URLs, or error messages the \
text doesn't mention or clearly imply.

The tester's target site: {url}

Their description (may describe one test case or several — split it into one structured \
test case per distinct test they described):
{raw_text}

For each test case, produce:
- test_id: short, stable, lowercase-with-hyphens, derived from the behavior under test.
- feature_id: name the small number of high-level features these cases cover (e.g. "login", \
"checkout") and list them in your separate features output; every test case's feature_id \
must be one of those ids.
- goal: one sentence, "<action> should <expected outcome>".
- category: happy_path, edge_case, negative, error_handling, security, or state_interaction \
— pick the one that matches what the tester described; default to happy_path if genuinely \
ambiguous.
- priority: high, medium, or low.
- expected_result: the one observable outcome that decides Pass/Fail. If the tester stated \
one, use it (quoted where they gave exact text). If they only implied one, write the most \
concrete observable version of what they implied — never a vague oracle like "it works".
- steps: ordered, one action per step, with every value to type written literally in quotes. \
Preserve the tester's own steps/values where they gave them; only add an obvious missing \
setup step (e.g. "Navigate to {url}") if their description clearly starts mid-flow.
- preconditions: DISPLAY-ONLY summary of the starting state.
- requires_auth: true only if the tester's own description says this case needs an \
already-logged-in session and is not itself testing login/signup.
"""


def _custom_plan_llm():
    # Same PLANNER_MODEL/PLANNER_FALLBACK_MODEL role as planner_node/discovery_agent_node
    # (core/llm.py's with_fallback) — this is a lighter-weight variant of the same
    # structured-output job, not a different capability tier.
    return with_fallback(ModelRole.PLANNER, lambda m: m.with_structured_output(TestPlan), temperature=0)


async def parse_custom_plan(raw_text: str, target_url: str, *, config: dict | None = None) -> TestPlan:
    return await _custom_plan_llm().ainvoke(
        CUSTOM_PLAN_PARSE_PROMPT.format(url=target_url, raw_text=raw_text), config=config
    )
