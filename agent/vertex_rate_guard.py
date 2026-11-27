"""Cross-process Vertex capacity cooldown, shared by Hermes profiles using one endpoint.

Only Vertex 429s change this state. SQLite transactions serialize recovery probes
across gateway workers and independent Hermes processes. Database failures fail open.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager

from hermes_constants import get_default_hermes_root

logger = logging.getLogger(__name__)
_PROBE_LEASE_SECONDS = 120


def _key(provider: str, model: str, base_url: str) -> str:
    if str(provider).lower() != "vertex" or not model or not base_url:
        return ""
    return f"vertex\0{base_url.rstrip('/').lower()}\0{model}"


@contextmanager
def _store():
    root = get_default_hermes_root() / "state"
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(root / "vertex_rate_guard.sqlite3"), timeout=1.5)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS routes (route TEXT PRIMARY KEY, "
                     "failures INTEGER NOT NULL, next_probe REAL NOT NULL, lease REAL NOT NULL)")
        yield conn
    finally:
        conn.close()


def seconds_blocked(provider: str, model: str, base_url: str) -> float:
    """Remaining shared cooldown (or another worker's probe lease). Read-only gate."""
    key = _key(provider, model, base_url)
    if not key:
        return 0
    try:
        with _store() as db:
            row = db.execute("SELECT next_probe, lease FROM routes WHERE route=?", (key,)).fetchone()
        return max(0.0, max(row) - time.time()) if row else 0
    except (OSError, sqlite3.Error):
        logger.debug("Vertex capacity guard unavailable", exc_info=True)
        return 0


def admit_probe(provider: str, model: str, base_url: str) -> bool:
    """Allow normal traffic when healthy; admit only one post-cooldown probe."""
    key = _key(provider, model, base_url)
    if not key:
        return True
    try:
        with _store() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT failures, next_probe, lease FROM routes WHERE route=?", (key,)).fetchone()
            now = time.time()
            if row is None or row[0] == 0:
                db.commit()
                return True
            if max(row[1], row[2]) > now:
                db.commit()
                return False
            db.execute("UPDATE routes SET lease=? WHERE route=?", (now + _PROBE_LEASE_SECONDS, key))
            db.commit()
            return True
    except (OSError, sqlite3.Error):
        logger.debug("Vertex capacity probe gate unavailable", exc_info=True)
        return True


def record_failure(provider: str, model: str, base_url: str) -> float:
    """Publish an exponentially increasing shared 429 cooldown; return seconds remaining."""
    key = _key(provider, model, base_url)
    if not key:
        return 0
    try:
        with _store() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT failures, next_probe, lease FROM routes WHERE route=?", (key,)).fetchone()
            now = time.time()
            # In-flight requests can return 429 after the first rejection.
            # They are not recovery probes and must not drive the global delay to 4h.
            if row and row[1] > now and row[2] <= now:
                db.commit()
                return row[1] - now
            failures = min((row[0] if row else 0) + 1, 9)
            delay = min(60 * 2 ** (failures - 1), 14400)
            next_probe = max(now + delay, row[1] if row else 0)
            db.execute("INSERT INTO routes(route, failures, next_probe, lease) VALUES(?,?,?,0) "
                       "ON CONFLICT(route) DO UPDATE SET failures=excluded.failures, "
                       "next_probe=excluded.next_probe, lease=0", (key, failures, next_probe))
            db.commit()
            return max(0, next_probe - now)
    except (OSError, sqlite3.Error):
        logger.debug("Could not publish Vertex rate limit", exc_info=True)
        return 0


def record_success(provider: str, model: str, base_url: str) -> None:
    """Clear the shared cooldown after a validated response from the same route."""
    key = _key(provider, model, base_url)
    if not key:
        return
    try:
        with _store() as db:
            # A request already in flight before the 429 is not a recovery probe.
            # Only the leased probe can clear an active outage marker.
            db.execute("DELETE FROM routes WHERE route=? AND lease > ?", (key, time.time()))
            db.commit()
    except (OSError, sqlite3.Error):
        logger.debug("Could not clear Vertex rate limit", exc_info=True)
