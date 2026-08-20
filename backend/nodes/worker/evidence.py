"""Evidence capture for one worker's browser session: tracing/video are started once,
right after the Playwright MCP session connects (`start_capture`, called from
session.py's `get_session`); screenshot/trace/video are captured once, in
verdict_node, right before the browser closes (`capture_screenshot`/
`stop_and_capture`). See each function's docstring for the tool-schema quirks that
make the start and stop/capture halves asymmetric.
"""
from __future__ import annotations

from pathlib import Path

EVIDENCE_DIR = Path(__file__).resolve().parent.parent.parent / "evidence"

# Session keys for which browser_video_show_actions has already been turned on —
# tracked here (not in session.py's _SESSIONS tuple) so enabling it stays a small,
# self-contained add-on instead of changing that tuple's shape everywhere it's
# unpacked. See ensure_action_overlay() for why this can't just happen once in
# start_capture().
_action_overlay_enabled: set[str] = set()


def run_dir_for(session_key: str) -> Path:
    # session_key contains ":" (invalid in a Windows path component) — sanitize for the dir name.
    return EVIDENCE_DIR / session_key.replace(":", "_")


async def start_capture(tool_map: dict, session_key: str) -> None:
    """Started once per session (--caps=devtools tools, gated in mcp/client.py) so they
    cover the whole test case; stopped + captured in verdict_node, just before
    browser_close. Guarded with .get() in case --caps=devtools isn't set.
    """
    if tool_map.get("browser_start_tracing") is not None:
        # Confirmed against the tool's real input schema: browser_start_tracing and
        # browser_stop_tracing both take NO arguments at all — there is no way to
        # steer the destination, it always lands under the MCP server's own default
        # working-directory location. stop_and_capture() below reflects this honestly
        # (returns None) rather than claiming a file exists where it can't.
        await tool_map["browser_start_tracing"].ainvoke({})
    if tool_map.get("browser_start_video") is not None:
        # Unlike tracing, browser_start_video's `filename` is the ONLY place the
        # destination can be set — browser_stop_video takes no arguments — so it
        # must be fixed here, up front, using the same run_dir stop_and_capture()
        # will look for it in.
        run_dir = run_dir_for(session_key)
        run_dir.mkdir(parents=True, exist_ok=True)
        await tool_map["browser_start_video"].ainvoke({"filename": str(run_dir / "video.webm")})


async def ensure_action_overlay(tool_map: dict, session_key: str) -> None:
    """Best-effort, idempotent: turns on @playwright/mcp's built-in
    browser_video_show_actions (--caps=devtools) the first time it can succeed for this
    session, then never calls it again. It bakes a callout naming each subsequent
    action, a highlight box around the action's target element, and an animated
    pointer moving between action points directly into the recorded video — so a human
    watching the recording afterward can see exactly what the agent clicked or typed
    on, not just the end result.

    Can't just be called once from start_capture() alongside browser_start_video:
    unlike video/tracing start (context-level, no page needed), this tool operates on
    "the current tab" and throws ("No open pages available.") until the session's
    first page exists — which happens lazily, on the first navigate/click/etc., not at
    session-open time. So this is called again after every real tool_node action;
    it's a no-op past the first success (tracked in _action_overlay_enabled), and
    harmlessly retries on failure (no tab yet) until one exists.
    """
    if session_key in _action_overlay_enabled:
        return
    tool = tool_map.get("browser_video_show_actions")
    if tool is None:
        return
    try:
        await tool.ainvoke({})
        _action_overlay_enabled.add(session_key)
    except Exception:
        pass  # no open tab yet — the next action's attempt will retry


def discard_action_overlay(session_key: str) -> None:
    _action_overlay_enabled.discard(session_key)


async def capture_screenshot(tool_map: dict, run_dir: Path) -> str:
    path = run_dir / "final.png"
    await tool_map["browser_take_screenshot"].ainvoke({"filename": str(path)})
    return str(path.relative_to(EVIDENCE_DIR.parent))


async def stop_and_capture(tool_map: dict, tool_name: str, path: Path) -> str | None:
    # Confirmed against the real input schemas: neither browser_stop_tracing nor
    # browser_stop_video takes a filename (or any argument at all) — passing one here
    # used to be silently ignored. video's destination is fixed at browser_start_video
    # time (see start_capture), so `path` correctly exists once this returns; tracing has
    # no destination control at all, so `path` never exists — report that honestly
    # instead of returning a path to a file that isn't there. Tool absent entirely if
    # --caps=devtools isn't set.
    tool = tool_map.get(tool_name)
    if tool is None:
        return None
    await tool.ainvoke({})
    if not path.exists():
        return None
    return str(path.relative_to(EVIDENCE_DIR.parent))
