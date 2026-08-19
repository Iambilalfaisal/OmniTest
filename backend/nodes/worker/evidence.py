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
