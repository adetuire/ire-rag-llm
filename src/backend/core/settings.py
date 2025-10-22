"""Runtime settings for the backend proxy service."""

from __future__ import annotations

import os
from datetime import timedelta

SECRET_KEY = os.environ.get("BACKEND_SECRET_KEY", "development-secret-key")
SESSION_COOKIE_NAME = "sessionid"
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "strict"
SESSION_MAX_AGE = timedelta(hours=1)

CSRF_COOKIE_NAME = "csrftoken"
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_SAMESITE = "strict"
CSRF_HEADER_NAME = "X-CSRFToken"

DEFAULT_USERNAME = os.environ.get("BACKEND_USERNAME", "admin")
DEFAULT_PASSWORD = os.environ.get("BACKEND_PASSWORD", "change-me")

RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:9000/chat")
RAG_SERVICE_API_KEY = os.environ.get("RAG_SERVICE_API_KEY", "test-api-key")
