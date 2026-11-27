"""Cross-process Vertex 429 recovery gate and per-route isolation."""

import multiprocessing
import os
import time
from unittest.mock import patch

from agent import vertex_rate_guard as guard


URL = "https://aiplatform.googleapis.com/v1beta1/projects/labwired-ai/locations/global/endpoints/openapi"
MODEL = "google/gemini-4-neon"


def _attempt_from_child(home, queue):
    os.environ["HERMES_HOME"] = home
    from agent.vertex_rate_guard import admit_probe
    queue.put(admit_probe("vertex", MODEL, URL))


def test_shared_cooldown_and_single_probe_across_processes(tmp_path):
    with patch.object(guard, "get_default_hermes_root", return_value=tmp_path):
        assert guard.admit_probe("vertex", MODEL, URL)
        assert 0 < guard.record_failure("vertex", MODEL, URL) <= 61
        assert guard.seconds_blocked("vertex", MODEL, URL) > 0
        assert not guard.admit_probe("vertex", MODEL, URL)
        # An unrelated response before a probe cannot clear a fresh 429.
        guard.record_success("vertex", MODEL, URL)
        assert guard.seconds_blocked("vertex", MODEL, URL) > 0
        with guard._store() as db:
            db.execute("UPDATE routes SET next_probe=0, lease=0")
            db.commit()
        assert guard.admit_probe("vertex", MODEL, URL)
        assert not guard.admit_probe("vertex", MODEL, URL)
        guard.record_success("vertex", MODEL, URL)
        assert guard.seconds_blocked("vertex", MODEL, URL) == 0
        assert guard.admit_probe("vertex", MODEL, URL)


def test_429_escalation_is_route_scoped(tmp_path):
    with patch.object(guard, "get_default_hermes_root", return_value=tmp_path):
        a = guard.record_failure("vertex", MODEL, URL)
        # A concurrent in-flight 429 must not double the window.
        duplicate = guard.record_failure("vertex", MODEL, URL)
        assert 0 < a <= 61
        assert 0 < duplicate <= 61
        with guard._store() as db:
            db.execute("UPDATE routes SET next_probe=0, lease=0")
            db.commit()
        assert guard.admit_probe("vertex", MODEL, URL)
        b = guard.record_failure("vertex", MODEL, URL)
        assert 110 < b <= 121
        assert guard.admit_probe("openai-codex", MODEL, URL)
        assert guard.seconds_blocked("vertex", MODEL, URL + "/another") == 0
        guard.record_success("vertex", MODEL, URL + "/another")
        assert guard.seconds_blocked("vertex", MODEL, URL) > 0


def test_existing_sqlite_gate_blocks_another_process(tmp_path):
    # The second process reads an existing on-disk cooldown without sharing Python state.
    with patch.object(guard, "get_default_hermes_root", return_value=tmp_path):
        guard.record_failure("vertex", MODEL, URL)
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_attempt_from_child, args=(str(tmp_path), q))
    proc.start()
    proc.join(timeout=10)
    assert proc.exitcode == 0
    assert q.get(timeout=2) is False


def test_ongoing_outage_probes_no_less_than_every_ten_minutes(tmp_path):
    with patch.object(guard, "get_default_hermes_root", return_value=tmp_path):
        delay = guard.record_failure("vertex", MODEL, URL)
        for _ in range(9):
            with guard._store() as db:
                db.execute("UPDATE routes SET next_probe=0, lease=0")
                db.commit()
            assert guard.admit_probe("vertex", MODEL, URL)
            delay = guard.record_failure("vertex", MODEL, URL)
        assert 590 <= delay <= 601
