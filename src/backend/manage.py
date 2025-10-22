#!/usr/bin/env python3
"""Minimal management entry point for the backend stack.

This stub mirrors the layout produced by ``django-admin startproject`` so that
other tooling can discover the backend package.  In this offline environment we
bootstrap the FastAPI application that powers the proxy and authentication
logic instead of invoking Django's management commands.
"""

from __future__ import annotations

import os
import sys

DEFAULT_SETTINGS_MODULE = "core.settings"


def main() -> None:
    os.environ.setdefault("BACKEND_SETTINGS_MODULE", DEFAULT_SETTINGS_MODULE)
    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
    else:
        command = "(none)"
    message = (
        "Management command support is not available in this environment. "
        "Requested command: %s."
    )
    print(message % command)


if __name__ == "__main__":
    main()
