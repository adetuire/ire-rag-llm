"""Lightweight signed cookie session implementation."""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Dict, Optional

from . import settings


@dataclass
class SessionData:
    """Represents the mutable data stored for a session."""

    session_id: Optional[str]
    payload: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    dirty: bool = False
    should_clear: bool = False

    def mark_dirty(self) -> None:
        self.dirty = True

    @property
    def is_authenticated(self) -> bool:
        return "user" in self.payload

    def require_authentication(self) -> None:
        if not self.is_authenticated:
            raise PermissionError("Authentication required")


_SESSIONS: Dict[str, SessionData] = {}
_LOCK = threading.Lock()


def _sign(value: str) -> str:
    signature = hmac.new(settings.SECRET_KEY.encode("utf-8"), value.encode("utf-8"), sha256)
    return signature.hexdigest()


def _serialize_cookie(session_id: str) -> str:
    return f"{session_id}:{_sign(session_id)}"


def _deserialize_cookie(cookie_value: str) -> Optional[str]:
    if not cookie_value:
        return None
    try:
        session_id, signature = cookie_value.split(":", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(signature, _sign(session_id)):
        return None
    return session_id


def create_session(payload: Optional[Dict[str, str]] = None) -> SessionData:
    session_id = secrets.token_hex(16)
    data = SessionData(session_id=session_id, payload=payload or {}, created_at=time.time())
    with _LOCK:
        _SESSIONS[session_id] = data
    data.mark_dirty()
    return data


def get_session(session_id: str) -> Optional[SessionData]:
    with _LOCK:
        data = _SESSIONS.get(session_id)
    if not data:
        return None
    lifetime = settings.SESSION_MAX_AGE.total_seconds()
    if data.created_at + lifetime < time.time():
        destroy_session(session_id)
        return None
    return data


def destroy_session(session_id: Optional[str]) -> None:
    if not session_id:
        return
    with _LOCK:
        _SESSIONS.pop(session_id, None)


def load_from_cookie(cookie_value: Optional[str]) -> SessionData:
    session_id = _deserialize_cookie(cookie_value or "")
    if not session_id:
        return SessionData(session_id=None)
    data = get_session(session_id)
    if not data:
        return SessionData(session_id=None)
    return SessionData(session_id=session_id, payload=dict(data.payload), created_at=data.created_at)


def persist_session(session: SessionData) -> Optional[str]:
    """Persist the session if it is marked as dirty.

    Returns the cookie value that should be sent to the client when the session
    has been updated.
    """

    if session.should_clear:
        destroy_session(session.session_id)
        return ""

    if not session.dirty:
        return None

    if not session.session_id:
        new_session = create_session(session.payload)
        session.session_id = new_session.session_id
        session.payload = dict(new_session.payload)
        session.created_at = new_session.created_at
        session.dirty = False
        return _serialize_cookie(session.session_id)

    with _LOCK:
        _SESSIONS[session.session_id] = SessionData(
            session_id=session.session_id,
            payload=dict(session.payload),
            created_at=time.time(),
        )
    session.dirty = False
    return _serialize_cookie(session.session_id)


def rotate_session(session: SessionData) -> SessionData:
    destroy_session(session.session_id)
    new_session = create_session(dict(session.payload))
    return new_session


def ensure_csrf_token(session: SessionData) -> str:
    token = session.payload.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session.payload["csrf_token"] = token
        session.mark_dirty()
    return token


def validate_csrf(session: SessionData, token: Optional[str]) -> None:
    expected = session.payload.get("csrf_token")
    if not expected or not token or not hmac.compare_digest(expected, token):
        raise ValueError("Invalid CSRF token")


def login_user(session: SessionData, username: str) -> None:
    session.payload["user"] = username
    session.mark_dirty()


def logout_user(session: SessionData) -> None:
    session.payload.clear()
    session.should_clear = True

