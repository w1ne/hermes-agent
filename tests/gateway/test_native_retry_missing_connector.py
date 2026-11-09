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


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy,provenance,pinned,allowed', [
    (True, True, False, True),  # recorded connector, but pre-column transport owner
    (False, True, False, False),  # a modern binding must not be substituted
    (True, False, False, False),  # legacy row without committed connector proof
    (True, True, True, False),  # known adapter went away: never fall back to another
])
async def test_legacy_native_route_uses_only_authenticated_recorded_connector(
    monkeypatch, legacy, provenance, pinned, allowed,
):
    import gateway.session_envelope as envelope
    import gateway.session_identity as identity
    import gateway.session_ingress_context as ingress_context

    adapter = object()
    source = SimpleNamespace(platform=SimpleNamespace(value='telegram'))
    available = SimpleNamespace()
    event = SimpleNamespace(source=source)
    entry = SimpleNamespace(session_id='s', transport_profile=None if legacy else 'primary')
    store = SimpleNamespace(_generate_session_key=lambda src: 'r', lookup_by_session_key=lambda route: entry)
    runner = SimpleNamespace(session_store=store, _adapter_for_source=lambda src: adapter)
    payload = {'native_text_v1': {'route': 'r'}}
    if provenance:
        payload['native_text_v1']['provenance'] = {'authenticated-current-connector': True}

    monkeypatch.setattr(envelope, 'restore_native', lambda *args: event)
    monkeypatch.setattr(envelope, '_validate_native', lambda *args: None)
    async def roles(*args):
        return False
    monkeypatch.setattr(ingress_context, 'reauthorize_roles', roles)
    monkeypatch.setattr(identity, 'identity_of', lambda src: object() if pinned else None)
    if allowed:
        result = await envelope.check_native_route(runner, payload, 's', available, None)
        assert result == (source, 'r')
    else:
        with pytest.raises(RuntimeStoreError, match='not_found'):
            await envelope.check_native_route(runner, payload, 's', available, None)
