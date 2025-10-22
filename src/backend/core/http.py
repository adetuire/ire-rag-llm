"""Utilities for handling lightweight HTTP-style requests and responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from . import settings
from . import session as session_backend


@dataclass
class Response:
    status_code: int
    body: Dict[str, Any]
    cookies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)

    def json(self) -> Dict[str, Any]:
        return self.body


class RequestContext:
    def __init__(
        self,
        json_body: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
        cookies: Optional[Dict[str, str]],
        session: session_backend.SessionData,
    ) -> None:
        self._json_body = json_body or {}
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.session = session

    def json(self) -> Dict[str, Any]:
        return dict(self._json_body)


class BackendApp:
    def __init__(self) -> None:
        self._routes: Dict[str, Callable[[RequestContext], Response]] = {}

    def route(self, path: str) -> Callable[[Callable[[RequestContext], Response]], Callable[[RequestContext], Response]]:
        def decorator(func: Callable[[RequestContext], Response]) -> Callable[[RequestContext], Response]:
            self._routes[path] = func
            return func

        return decorator

    def handle(
        self,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
    ) -> Response:
        handler = self._routes.get(path)
        if not handler:
            raise ValueError(f"Unknown path: {path}")

        cookie_value = (cookies or {}).get(settings.SESSION_COOKIE_NAME)
        session = session_backend.load_from_cookie(cookie_value)
        request = RequestContext(json_body, headers, cookies, session)

        response = handler(request)

        cookie_payload = session_backend.persist_session(request.session)
        if cookie_payload == "":
            response.cookies[settings.SESSION_COOKIE_NAME] = {
                "value": "",
                "path": "/",
                "secure": settings.SESSION_COOKIE_SECURE,
                "samesite": settings.SESSION_COOKIE_SAMESITE,
                "expires": 0,
            }
        elif cookie_payload:
            response.cookies[settings.SESSION_COOKIE_NAME] = {
                "value": cookie_payload,
                "path": "/",
                "secure": settings.SESSION_COOKIE_SECURE,
                "httponly": settings.SESSION_COOKIE_HTTPONLY,
                "samesite": settings.SESSION_COOKIE_SAMESITE,
            }

        return response


__all__ = ["BackendApp", "RequestContext", "Response"]
