"""ASGI entrypoint for the Neuroinformatik API."""

from src.api.app import create_app

app = create_app()
