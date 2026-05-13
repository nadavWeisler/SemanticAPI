#!/usr/bin/env python3
"""Development entrypoint — use Gunicorn for production."""
import os

from semanticapi import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
