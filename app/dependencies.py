"""
Shared FastAPI dependencies.

The predictor is loaded once at application startup and injected into routes
via this dependency, so it is never reloaded per request.
"""

from fastapi import Request

from app.predictor import load_predictor


def get_predictor(request: Request):
    """Return the predictor stored on app.state (set during startup)."""
    return request.app.state.predictor