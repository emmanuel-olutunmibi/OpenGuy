"""Vercel serverless entrypoint for the Flask app."""

from app import app

# Exported as the WSGI application for @vercel/python
handler = app
