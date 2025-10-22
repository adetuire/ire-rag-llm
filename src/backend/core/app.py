"""Application wiring for the backend proxy service."""

from __future__ import annotations

from .http import BackendApp
from .views import rag_proxy

app = BackendApp()


@app.route("/auth/login")
def login_endpoint(request):
    return rag_proxy.login_endpoint(request)


@app.route("/auth/logout")
def logout_endpoint(request):
    return rag_proxy.logout_endpoint(request)


@app.route("/rag/proxy")
def rag_proxy_endpoint(request):
    return rag_proxy.rag_proxy_endpoint(request)


__all__ = ["app"]
