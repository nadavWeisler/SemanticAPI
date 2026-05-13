"""Structured JSON logging configuration."""
from __future__ import annotations

import logging
import time

from flask import Flask, g, request


def configure_logging(app: Flask) -> None:
    """Attach structured JSON logging to *app*."""
    try:
        from pythonjsonlogger import jsonlogger

        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        handler.setFormatter(formatter)

        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        app.logger.handlers = []
    except ImportError:
        # Fall back to default logging if python-json-logger is not installed
        pass

    @app.before_request
    def _start_timer():
        g.start_time = time.perf_counter()

    @app.after_request
    def _log_request(response):
        start = g.get("start_time")
        latency_ms = round((time.perf_counter() - start) * 1000, 2) if start is not None else None
        app.logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
                "lang": request.args.get("lang", ""),
            },
        )
        return response
