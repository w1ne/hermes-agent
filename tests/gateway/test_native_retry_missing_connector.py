"""A temporarily missing messaging connector must not strand a durable native FIFO."""
from types import SimpleNamespace

import pytest

from hermes_state_runtime import RuntimeStoreError
from gateway.session_authority import SessionAuthority
from gateway.session_ingress import pause_notice


@pytest.mark.asyncio
@pytest.mark.parametrize('reason,native,retries', [
    ('not_found', True, 3),
    ('not_found', False, 1),
    ('permission_denied', True, 1),
    ('profile_mismatch', True, 1),
])
async def test_missing_native_connector_retries_with_fresh_authorization(
    monkeypatch, reason, native, retries,
):
    import gateway.session_authority as mod

    pending = [{'status': 'queued', 'payload': {'native_text_v1': {}} if native else {'text': 'local'}}]
    monkeypatch.setattr(mod, 'list_session_admissions', lambda db, *, session_id: pending)
    monkeypatch.setattr(mod, '_NATIVE_REAUTH_RETRY_MIN_S', 0.001)
    monkeypatch.setattr(mod, '_NATIVE_REAUTH_RETRY_MAX_S', 0.003)
    calls = []
    pauses = []
    async def claim(ref, live):
        calls.append(1)
        if len(calls) < retries or retries == 1:
            raise RuntimeStoreError(reason)
        return ('claimed', pending[0])

    authority = SimpleNamespace(
        db=object(), _claim_next=claim,
        _pause=lambda ref, cause: pauses.append(cause),
        _drain_stopping=lambda ref, live: False,
    )
    ref = SimpleNamespace(session_id='session1')
    got = await SessionAuthority._claim_with_retry(authority, ref, object())
    assert len(calls) == retries
    if reason == 'not_found' and native:
        assert got == ('claimed', pending[0])
        assert pauses == ['reauthorization_retry'] * (retries - 1)
    else:
        assert got is None
        assert pauses == [reason]


def test_transient_notice_says_retry_without_resetting_history():
    live = SimpleNamespace(pause_notified=False)
    authority = SimpleNamespace(sessions={'s': live})
    ref = SimpleNamespace(session_id='s')
    notice = pause_notice(authority, ref, 'reauthorization_retry')
    assert 'retry automatically' in notice
    assert 'only after authorization succeeds' in notice
    assert 'will not be answered here' not in notice
    assert pause_notice(authority, ref, 'reauthorization_retry') is None
