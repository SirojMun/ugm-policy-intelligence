"""ASGI entrypoint for `uvicorn app.server:app`."""

from __future__ import annotations

from app.factory import QueryRequest, create_app

app = create_app()

__all__ = ["app", "create_app", "QueryRequest"]
