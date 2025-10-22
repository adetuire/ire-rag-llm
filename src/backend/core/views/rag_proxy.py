"""Endpoints for authenticating users and proxying chat requests."""

from __future__ import annotations

import json
import urllib.request
from http import HTTPStatus
from typing import Any, Dict

from .. import auth, settings
from ..session import ensure_csrf_token, login_user, logout_user, validate_csrf
from ..http import RequestContext, Response


def login_endpoint(request: RequestContext) -> Response:
    payload = request.json()
    username = payload.get("username")
    password = payload.get("password")
    if not username or not password:
        return Response(status_code=HTTPStatus.BAD_REQUEST, body={"detail": "Missing credentials"})

    user = auth.authenticate(username, password)
    if not user:
        return Response(status_code=HTTPStatus.UNAUTHORIZED, body={"detail": "Invalid credentials"})

    session = request.session
    login_user(session, user.username)
    csrf_token = ensure_csrf_token(session)

    response = Response(status_code=HTTPStatus.OK, body={"status": "ok", "user": user.username})
    response.cookies[settings.CSRF_COOKIE_NAME] = {
        "value": csrf_token,
        "secure": settings.CSRF_COOKIE_SECURE,
        "httponly": False,
        "samesite": settings.CSRF_COOKIE_SAMESITE,
        "path": "/",
    }
    return response


def logout_endpoint(request: RequestContext) -> Response:
    session = request.session
    if session.is_authenticated:
        logout_user(session)
    response = Response(status_code=HTTPStatus.OK, body={"status": "ok"})
    response.cookies[settings.CSRF_COOKIE_NAME] = {
        "value": "",
        "secure": settings.CSRF_COOKIE_SECURE,
        "samesite": settings.CSRF_COOKIE_SAMESITE,
        "path": "/",
        "expires": 0,
    }
    return response


def rag_proxy_endpoint(request: RequestContext) -> Response:
    session = request.session
    if not session.is_authenticated:
        return Response(status_code=HTTPStatus.FORBIDDEN, body={"detail": "Login required"})

    csrf_header = request.headers.get(settings.CSRF_HEADER_NAME)
    try:
        validate_csrf(session, csrf_header)
    except ValueError as exc:
        return Response(status_code=HTTPStatus.FORBIDDEN, body={"detail": str(exc)})

    payload = request.json()
    headers = {"Content-Type": "application/json"}
    if settings.RAG_SERVICE_API_KEY:
        headers["Authorization"] = f"Bearer {settings.RAG_SERVICE_API_KEY}"
    proxy_request = urllib.request.Request(
        settings.RAG_SERVICE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(proxy_request) as result:
            body = result.read().decode("utf-8")
            status_code = result.status
            content_type = result.headers.get("Content-Type", "application/json")
    except Exception as exc:  # pragma: no cover
        return Response(status_code=HTTPStatus.BAD_GATEWAY, body={"detail": str(exc)})

    data = json.loads(body) if body else {}
    response = Response(status_code=status_code, body=data)
    response.headers["Content-Type"] = content_type
    return response
