"""Bounded, read-only, same-origin site crawl used to build a lightweight site map for
the planning prompt (one-shot planner and chat-first discovery alike). A plain function
over an already-open MCP tools list — no LangGraph/state dependency — so it's callable
both automatically (upfront, from planner_node/discovery_agent_node) and on-demand (a
narrower start_url mid-discovery-chat), using whichever session the caller already has
open via mcp.client.open_playwright_session.
"""
from __future__ import annotations

from collections import deque
from urllib.parse import urlparse

from ..core.models import PageSummary, SiteMap
from ..mcp.client import get_page_title, list_page_links, navigate, shallow_snapshot

DEFAULT_MAX_PAGES = 8
DEFAULT_MAX_DEPTH = 2
DEFAULT_SNAPSHOT_DEPTH = 4
DEFAULT_DIGEST_CHAR_LIMIT = 1500

# Link text/href substrings that should never be followed during a read-only crawl, even
# though they're same-origin — these are one-click state-changing actions, not content.
CRAWL_SKIP_LINK_KEYWORDS = (
    "logout",
    "log-out",
    "log_out",
    "signout",
    "sign-out",
    "sign_out",
    "delete",
    "remove",
    "deactivate",
    "unsubscribe",
    "cancel",
)


def _normalize(url: str) -> str:
    """Dedup key: drop query/fragment and trailing slash so pagination/filter links and
    /about vs /about/ don't inflate the frontier."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"


def _is_explorable(href: str, text: str, origin_netloc: str) -> bool:
    parsed = urlparse(href)
    if parsed.scheme not in ("http", "https"):
        return False  # skip mailto:, tel:, javascript:, etc.
    if parsed.netloc != origin_netloc:
        return False  # same-origin only
    haystack = f"{href} {text}".lower()
    return not any(keyword in haystack for keyword in CRAWL_SKIP_LINK_KEYWORDS)


async def crawl_site(
    tools: list,
    start_url: str,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    snapshot_depth: int = DEFAULT_SNAPSHOT_DEPTH,
    digest_char_limit: int = DEFAULT_DIGEST_CHAR_LIMIT,
    start_already_loaded: bool = False,
    already_visited: set[str] | None = None,
) -> SiteMap:
    """BFS same-origin crawl from start_url, depth-bounded and page-count-bounded.
    Never calls a form-filling, clicking, or other state-changing tool — only
    navigate/shallow_snapshot (read) and list_page_links (a fixed, read-only DOM query).

    start_already_loaded: True when the caller's `tools` browser is already sitting on
    start_url (planner_node calls get_accessibility_snapshot(tools, target_url) right
    before this, which already navigated there) — skips a redundant re-navigate for the
    depth-0 page only.

    already_visited: URLs (post-_normalize) to treat as already known and skip entirely —
    lets an on-demand call (e.g. discovery chat's "look deeper at /checkout") avoid
    re-walking pages a prior crawl already covered in this conversation.
    """
    visited: set[str] = set(already_visited or ())
    origin_netloc = urlparse(start_url).netloc
    frontier: deque[tuple[str, int]] = deque([(start_url, 0)])
    pages: list[PageSummary] = []

    while frontier and len(pages) < max_pages:
        url, depth = frontier.popleft()
        key = _normalize(url)
        if key in visited:
            continue
        visited.add(key)

        if not (depth == 0 and start_already_loaded):
            await navigate(tools, url)

        title = await get_page_title(tools)
        digest = (await shallow_snapshot(tools, depth=snapshot_depth))[:digest_char_limit]
        pages.append(PageSummary(url=url, title=title, depth=depth, snapshot_digest=digest))

        if depth < max_depth:
            for link in await list_page_links(tools):
                if len(pages) + len(frontier) >= max_pages:
                    break
                href = link.get("href", "")
                text = link.get("text", "")
                if not href or not _is_explorable(href, text, origin_netloc):
                    continue
                if _normalize(href) not in visited:
                    frontier.append((href, depth + 1))

    return SiteMap(pages=pages, truncated=bool(frontier))


def format_site_map_for_prompt(site_map: SiteMap, *, exclude_url: str | None = None) -> str:
    """Render a SiteMap into a prompt's {site_map} block. Excludes exclude_url (the
    target page) since the caller already feeds that page's FULL tree separately — this
    only covers the other discovered pages, at shallow/truncated fidelity.
    """
    pages = [p for p in site_map.pages if p.url != exclude_url]
    if not pages:
        return "No additional pages discovered.\n"
    blocks = [f'- {p.url} — "{p.title}" (depth {p.depth})\n  {p.snapshot_digest}' for p in pages]
    suffix = "\n(Crawl stopped early — more linked pages exist than were explored.)\n" if site_map.truncated else "\n"
    return "\n".join(blocks) + suffix
