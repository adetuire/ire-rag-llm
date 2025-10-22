"""Simple credential store backed by environment defaults."""

from __future__ import annotations

from dataclasses import dataclass

from . import settings


@dataclass
class User:
    username: str


def authenticate(username: str, password: str) -> User | None:
    expected_user = settings.DEFAULT_USERNAME
    expected_password = settings.DEFAULT_PASSWORD
    if username == expected_user and password == expected_password:
        return User(username=username)
    return None
