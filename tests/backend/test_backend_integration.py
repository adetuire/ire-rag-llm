from __future__ import annotations

import json
import os
from http import HTTPStatus
from pathlib import Path
from typing import Dict
from unittest import mock

import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Configure runtime before importing the application so that settings resolve to
# predictable values for the tests.
os.environ.setdefault("BACKEND_USERNAME", "admin")
os.environ.setdefault("BACKEND_PASSWORD", "s3cret")
os.environ.setdefault("RAG_SERVICE_URL", "http://rag-service.local/chat")
os.environ.setdefault("RAG_SERVICE_API_KEY", "token-123")

from backend.core import settings
from backend.core.app import app
from backend.core.http import BackendApp


class BackendClient:
    def __init__(self, backend: BackendApp):
        self.backend = backend
        self.cookies: Dict[str, str] = {}

    def post(self, path: str, json_body: Dict[str, str], headers: Dict[str, str] | None = None):
        response = self.backend.handle(path, json_body=json_body, headers=headers, cookies=self.cookies)
        for name, meta in response.cookies.items():
            value = meta.get("value", "")
            if not value:
                self.cookies.pop(name, None)
            else:
                self.cookies[name] = value
        return response


def _client() -> BackendClient:
    return BackendClient(app)


def test_login_sets_session_and_csrf_cookie():
    client = _client()

    response = client.post("/auth/login", json_body={"username": "admin", "password": "s3cret"})

    assert response.status_code == HTTPStatus.OK
    assert response.json()["status"] == "ok"

    session_cookie = response.cookies[settings.SESSION_COOKIE_NAME]["value"]
    csrf_cookie = response.cookies[settings.CSRF_COOKIE_NAME]["value"]

    assert session_cookie
    assert csrf_cookie


def test_proxy_requires_login():
    client = _client()
    response = client.post("/rag/proxy", json_body={"message": "hi"})
    assert response.status_code == HTTPStatus.FORBIDDEN


def test_proxy_with_session_and_csrf():
    client = _client()

    login_response = client.post("/auth/login", json_body={"username": "admin", "password": "s3cret"})
    csrf_cookie = login_response.cookies[settings.CSRF_COOKIE_NAME]["value"]

    fake_response = mock.MagicMock()
    fake_response.read.return_value = json.dumps({"reply": "hello"}).encode("utf-8")
    fake_response.status = HTTPStatus.OK
    fake_response.headers = {"Content-Type": "application/json"}

    class _Context:
        def __enter__(self):
            return fake_response

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request_obj):
        assert request_obj.full_url == os.environ["RAG_SERVICE_URL"]
        assert json.loads(request_obj.data.decode("utf-8")) == {"message": "ping"}
        assert request_obj.get_header("Authorization") == "Bearer token-123"
        return _Context()

    with mock.patch("backend.core.views.rag_proxy.urllib.request.urlopen", fake_urlopen):
        response = client.post(
            "/rag/proxy",
            json_body={"message": "ping"},
            headers={settings.CSRF_HEADER_NAME: csrf_cookie},
        )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"reply": "hello"}


def test_logout_clears_session():
    client = _client()
    login_response = client.post("/auth/login", json_body={"username": "admin", "password": "s3cret"})
    csrf_cookie = login_response.cookies[settings.CSRF_COOKIE_NAME]["value"]

    response = client.post(
        "/auth/logout",
        json_body={},
        headers={settings.CSRF_HEADER_NAME: csrf_cookie},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.cookies[settings.SESSION_COOKIE_NAME]["value"] == ""
