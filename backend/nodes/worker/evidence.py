"""Evidence capture for one worker's browser session: tracing is started once, right
after the Playwright MCP session connects (`start_capture`, called from session.py's
`get_session`), and captured once in verdict_node right before the browser closes
(`stop_and_capture`). Screenshot is likewise captured once, in verdict_node
(`capture_screenshot`).

Video is deliberately NOT session-length — see `capture_mutation_clip` below for why a
single recording spanning the whole test case was replaced with one short clip per
mutating action.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Awaitable, Callable

from ...mcp.client import invoke_tool

EVIDENCE_DIR = Path(__file__).resolve().parent.parent.parent / "evidence"

# Session keys for which browser_video_show_actions has already been turned on —
# tracked here (not in session.py's _SESSIONS tuple) so enabling it stays a small,
# self-contained add-on instead of changing that tuple's shape everywhere it's
# unpacked. See ensure_action_overlay() for why this can't just happen once in
# start_capture().
_action_overlay_enabled: set[str] = set()

# How much of the page's STATIC state (nothing happening yet / already settled)
# brackets each mutation clip — enough for a human reviewer to orient on the
# before/after state without padding every clip into a multi-second wait of its own.
# Env-tunable since it's pure wall-clock cost (no LLM/API quota spent): every mutating
# action in a test case adds roughly PRE + action-time + POST to that test case's
# real duration, so a plan with several mutations per case can add up.
ACTION_CLIP_PRE_SECONDS = float(os.getenv("ACTION_CLIP_PRE_SECONDS", "3"))
ACTION_CLIP_POST_SECONDS = float(os.getenv("ACTION_CLIP_POST_SECONDS", "3"))

# Tool calls worth their own clip — genuine page/state-changing interactions a human
# reviewer actually wants to SEE happen. Deliberately excludes reads (browser_snapshot,
# browser_console_messages, the cookie/storage *_get/*_list tools, browser_find,
# browser_take_screenshot, browser_network_request*), waits (browser_wait_for), and
# debug/dashboard tools (browser_annotate, browser_highlight) — none of those is a
# "mutation" a test case's video needs to show, and giving every read its own clip
# would reproduce the exact noise problem capture_mutation_clip exists to remove.
# Confirmed against the full installed @playwright/mcp tool list (52 tools).
MUTATING_TOOL_NAMES = frozenset(
    {
        "browser_click",
        "browser_type",
        "browser_fill_form",
        "browser_select_option",
        "browser_press_key",
        "browser_navigate",
        "browser_navigate_back",
        "browser_drag",
        "browser_drop",
        "browser_file_upload",
        "browser_handle_dialog",
    }
)


def run_dir_for(session_key: str) -> Path:
    # session_key contains ":" (invalid in a Windows path component) — sanitize for the dir name.
    return EVIDENCE_DIR / session_key.replace(":", "_")


async def start_capture(tool_map: dict, session_key: str) -> None:
    """Started once per session (--caps=devtools tools, gated in mcp/client.py) so it
    covers the whole test case; stopped + captured in verdict_node, just before
    browser_close. Guarded with .get() in case --caps=devtools isn't set.

    Video is deliberately NOT started here — see capture_mutation_clip below.
    """
    if tool_map.get("browser_start_tracing") is not None:
        # Confirmed against the tool's real input schema: browser_start_tracing and
        # browser_stop_tracing both take NO arguments at all — there is no way to
        # steer the destination, it always lands under the MCP server's own default
        # working-directory location. stop_and_capture() below reflects this honestly
        # (returns None) rather than claiming a file exists where it can't.
        await invoke_tool(tool_map["browser_start_tracing"], {})


async def ensure_action_overlay(tool_map: dict, session_key: str) -> None:
    """Best-effort, idempotent: turns on @playwright/mcp's built-in
    browser_video_show_actions (--caps=devtools) the first time it can succeed for this
    session, then never calls it again. It bakes a callout naming each subsequent
    action, a highlight box around the action's target element, and an animated
    pointer moving between action points directly into the recorded video — so a human
    watching a clip afterward can see exactly what the agent clicked or typed on, not
    just the end result.

    Can't just be called once from start_capture() alongside a session-length
    recording (there no longer is one): this tool operates on "the current tab" and
    throws ("No open pages available.") until the session's first page exists — which
    happens lazily, on the first navigate/click/etc., not at session-open time. So
    this is called again before every mutation clip; it's a no-op past the first
    success (tracked in _action_overlay_enabled), and harmlessly retries on failure (no
    tab yet) until one exists.
    """
    if session_key in _action_overlay_enabled:
        return
    tool = tool_map.get("browser_video_show_actions")
    if tool is None:
        return
    try:
        await invoke_tool(tool, {})
        _action_overlay_enabled.add(session_key)
    except Exception:
        pass  # no open tab yet — the next mutation's attempt will retry


def discard_action_overlay(session_key: str) -> None:
    _action_overlay_enabled.discard(session_key)


async def capture_screenshot(tool_map: dict, run_dir: Path) -> str:
    path = run_dir / "final.png"
    await invoke_tool(tool_map["browser_take_screenshot"], {"filename": str(path)})
    return str(path.relative_to(EVIDENCE_DIR.parent))


async def stop_and_capture(tool_map: dict, tool_name: str, path: Path) -> str | None:
    # Confirmed against the real input schema: browser_stop_tracing takes no argument
    # at all — passing one here used to be silently ignored, and tracing has no
    # destination control anyway, so `path` never exists — report that honestly
    # instead of returning a path to a file that isn't there. Tool absent entirely if
    # --caps=devtools isn't set. (Video no longer goes through this path — see
    # capture_mutation_clip, which starts and stops its own clip inline.)
    tool = tool_map.get(tool_name)
    if tool is None:
        return None
    await invoke_tool(tool, {})
    if not path.exists():
        return None
    return str(path.relative_to(EVIDENCE_DIR.parent))


async def capture_mutation_clip(
    tool_map: dict, run_dir: Path, clip_index: int, do_action: Callable[[], Awaitable]
) -> tuple[object, str | None]:
    """Records ONE short clip bracketing a single mutating tool call, instead of one
    recording spanning the whole test case.

    Confirmed live against the installed @playwright/mcp: browser_start_video /
    browser_stop_video can be called repeatedly within a single session, each cycle
    producing its own independent, valid clip file — that repeatability is what makes
    per-action clips possible without restarting the browser context.

    Why not one continuous recording: video used to start the moment a session opened
    (nodes/worker/session.py's get_session, before agent_node's first LLM call) and
    stop only in verdict_node, after the LAST turn — capturing the full wall-clock
    duration of the test case, including every turn spent waiting on an LLM decision
    between actions, which under real API latency/retries dominates it. Confirmed
    directly from a real run's evidence: a test case with well under a minute of actual
    browser interaction produced a 4-5 minute video where only the last few seconds
    showed anything happening. Bracketing just the mutation removes that dead air
    entirely — no recording runs while agent_node is deciding what to do next, or while
    tool_node is executing a read (browser_snapshot, etc.) that isn't in
    MUTATING_TOOL_NAMES.

    Multiple clips are not concatenated server-side into one file — WebM containers
    can't be safely joined by concatenating bytes, and a correct muxer is real,
    fiddly container-format work that's excess risk for something whose only job is to
    be trustworthy evidence. Instead each clip is kept as its own file and played back
    to back client-side (frontend/src/components/WorkerCard.tsx's ClipSequencePlayer),
    which needs no new dependency and reads as one continuous video regardless.

    Returns (action_result, clip_path | None) — clip_path is relative to
    EVIDENCE_DIR.parent, matching capture_screenshot's convention (the frontend
    resolves it against apiBase), or None if devtools capture isn't available
    (--caps) or the clip file never materialized.
    """
    start_tool = tool_map.get("browser_start_video")
    stop_tool = tool_map.get("browser_stop_video")
    if start_tool is None or stop_tool is None:
        return await do_action(), None

    run_dir.mkdir(parents=True, exist_ok=True)
    clip_path = run_dir / f"clip_{clip_index:03d}.webm"
    # Bounded and non-raising, unlike do_action() below: this runs directly inside
    # tool_node with no enclosing try/except, and a raise here would crash the WHOLE
    # run (see invoke_tool's docstring) over a video-recording failure, not even the
    # actual test action. Degrades to "no clip for this action" instead — the same
    # fallback this function already uses when the video tools aren't offered at all.
    try:
        await invoke_tool(start_tool, {"filename": str(clip_path)})
    except Exception:
        logging.exception("capture_mutation_clip: browser_start_video failed/timed out — proceeding without a clip")
        return await do_action(), None

    await asyncio.sleep(ACTION_CLIP_PRE_SECONDS)
    try:
        result = await do_action()
    finally:
        # In `finally` so a raised action still yields a clip showing what led to it —
        # and so a stray in-progress recording never lingers past this function ready
        # to collide with the NEXT mutation's browser_start_video call. Same
        # non-raising reasoning as browser_start_video above.
        await asyncio.sleep(ACTION_CLIP_POST_SECONDS)
        try:
            await invoke_tool(stop_tool, {})
        except Exception:
            logging.exception("capture_mutation_clip: browser_stop_video failed/timed out")

    if not clip_path.exists():
        return result, None
    return result, str(clip_path.relative_to(EVIDENCE_DIR.parent))
