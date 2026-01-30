# WSGI entrypoint for Gunicorn on Render
# Exposes `app` for the default start command `gunicorn app:app`

from jarvis_assistant import app as app
