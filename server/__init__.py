"""
Server package exposing a FastAPI app to interact with agents.

Usage (run server):
  uvicorn server.api:create_app --host 0.0.0.0 --port 9000

The app factory pattern keeps initialization modular and testable.
"""

from .api import create_app  # re-export for convenience

__all__ = ["create_app"]


