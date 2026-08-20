"""Root logger configuration — there is none today. Confirmed: no `basicConfig`/
`dictConfig`/`getLogger` call anywhere in backend/ before this module; the handful of
existing `logging.exception`/`logging.info` call sites (api.py, nodes/memory.py,
nodes/worker/session.py) all log through the root logger, which — since uvicorn only
configures its OWN `uvicorn.*` loggers, not root — sits on logging's `lastResort`
handler. That handler prints ERROR+ with no formatting, so `logging.exception` calls
were visible but bare (no timestamp, no run context) and `logging.info` calls were
silently dropped (lastResort's level floor is WARNING).

Call `configure_logging()` once, immediately after `load_dotenv()` in api.py — before
any node/graph import — so LOG_LEVEL is honored from the start and nothing logs before
the real handler is installed.
"""
from __future__ import annotations

import logging.config
import os

from .run_context import run_id_var, test_id_var


class _RunContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = run_id_var.get()
        record.test_id = test_id_var.get()
        return True


def configure_logging() -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            # False, not True: True would silence uvicorn's own already-configured
            # loggers (uvicorn.error, uvicorn.access) the moment this runs.
            "disable_existing_loggers": False,
            "filters": {"run_context": {"()": _RunContextFilter}},
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s [run=%(run_id)s test=%(test_id)s] %(name)s: %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["run_context"],
                }
            },
            "root": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "handlers": ["console"],
            },
            "loggers": {
                # Surfaces the SDK's own retry backoff (google-genai's internal
                # tenacity `before_sleep_log`) — the only visibility into the cheap,
                # SDK-level retry layer core/llm.py's node_retry_on now defers to
                # instead of retrying at node level. Format-coupled to the installed
                # tenacity version (it logs via an f-string, so `record.args` is empty
                # — only the rendered message is usable) and blind to the final,
                # terminally-failed attempt (no log line fires for that one).
                "google_genai._api_client": {"level": "INFO"},
            },
        }
    )
