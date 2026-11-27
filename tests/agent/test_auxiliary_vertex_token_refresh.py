"""A cached Vertex auxiliary client must use fresh OAuth tokens on every request."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent import auxiliary_client as aux
from agent import vertex_adapter


@pytest.fixture
def vertex_endpoint(monkeypatch):
    """Exercise real SDK requests against a local endpoint and a rotating fake token mint."""
    state = {"token": "vertex-test-first", "headers": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            authorization = self.headers.get("Authorization")
            state["headers"].append(authorization)
            valid = authorization == f"Bearer {state['token']}"
            body = (
                {"id": "local-completion", "object": "chat.completion", "created": 1,
                 "model": "google/gemini-3-flash-preview",
                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                              "finish_reason": "stop"}]}
                if valid else
                {"error": {"message": "ACCESS_TOKEN_EXPIRED", "type": "authentication_error"}}
            )
            encoded = json.dumps(body).encode()
            self.send_response(200 if valid else 401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setattr(vertex_adapter, "has_vertex_credentials", lambda: True)
    monkeypatch.setattr(vertex_adapter, "get_vertex_credentials", lambda *args, **kwargs: (state["token"], "test-project"))
    monkeypatch.setattr(vertex_adapter, "build_vertex_base_url", lambda *args, **kwargs: base_url)
    # SDK defaults suffice for loopback; avoid unrelated process-wide keepalive bootstrap.
    monkeypatch.setattr(aux, "_openai_http_client_kwargs", lambda *args, **kwargs: {})
    monkeypatch.setattr(aux, "_client_cache", {})
    monkeypatch.setattr(aux, "_aux_unhealthy_until", {})
    monkeypatch.setattr(aux, "_aux_unhealthy_logged_at", {})
    yield state, base_url
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_cached_vertex_aux_client_refreshes_expired_token(vertex_endpoint, async_mode):
    state, base_url = vertex_endpoint
    runtime = {"provider": "vertex", "model": "google/gemini-3-flash-preview", "base_url": base_url}

    def resolve():
        return aux._get_cached_client(
            "auto", async_mode=async_mode, main_runtime=runtime, task="kanban_decomposer",
            model=runtime["model"],
        )

    kwargs = {"model": runtime["model"], "messages": [{"role": "user", "content": "test"}]}

    async def async_requests():
        client, _model = resolve()
        assert client is not None
        try:
            assert (await client.chat.completions.create(**kwargs)).choices[0].message.content == "ok"
            state["token"] = "vertex-test-renewed"
            assert resolve()[0] is client
            assert (await client.chat.completions.create(**kwargs)).choices[0].message.content == "ok"
        finally:
            await client.close()

    if async_mode:
        asyncio.run(async_requests())
    else:
        client, _model = resolve()
        assert client is not None
        try:
            assert client.chat.completions.create(**kwargs).choices[0].message.content == "ok"
            state["token"] = "vertex-test-renewed"
            assert resolve()[0] is client
            assert client.chat.completions.create(**kwargs).choices[0].message.content == "ok"
        finally:
            client.close()
    assert state["headers"] == ["Bearer vertex-test-first", "Bearer vertex-test-renewed"]


@pytest.mark.parametrize("host", ["aiplatform.googleapis.com", "us-central1-aiplatform.googleapis.com"])
def test_auto_vertex_auth_refresh_uses_selected_provider(host):
    """Auto routing retains Vertex identity for both global and regional endpoints."""
    base_url = f"https://{host}/v1beta1/projects/test-project/locations/global/endpoints/openapi"
    assert aux._auth_refresh_provider_for_route("auto", base_url, "vertex") == "vertex"
