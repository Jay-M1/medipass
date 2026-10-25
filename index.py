"""Vercel entrypoint. Vercel finds `app` here, and everything else lives in the agent package."""

from agent.server import app

__all__ = ["app"]
