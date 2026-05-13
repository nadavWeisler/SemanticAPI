"""SemanticAPI Flask application factory."""
from __future__ import annotations

from flask import Flask

from semanticapi.config import RATE_LIMIT_DEFAULT
from semanticapi.logging_config import configure_logging
from semanticapi.models import preload_languages


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="../templates")

    # ── Structured JSON logging ────────────────────────────────────────────────
    configure_logging(app)

    # ── Flasgger (OpenAPI / Swagger UI) ───────────────────────────────────────
    try:
        from flasgger import Swagger
        Swagger(app, template={
            "info": {
                "title": "SemanticAPI",
                "description": (
                    "A REST API for semantic word similarity, analogies, "
                    "clustering and more — powered by FastText embeddings."
                ),
                "version": "2.0.0",
            },
            "basePath": "/",
        })
    except ImportError:
        pass

    # ── Rate limiting ──────────────────────────────────────────────────────────
    # The Limiter instance is stored on app.extensions["limiter"] by Flask-Limiter
    # itself; the local variable is not needed after initialization.
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        Limiter(
            get_remote_address,
            app=app,
            default_limits=[RATE_LIMIT_DEFAULT],
            storage_uri="memory://",
        )
    except ImportError:
        pass

    # ── Prometheus metrics ─────────────────────────────────────────────────────
    try:
        from prometheus_flask_exporter import PrometheusMetrics
        PrometheusMetrics(app)
    except ImportError:
        pass

    # ── Blueprints ─────────────────────────────────────────────────────────────
    from semanticapi.routes.health import bp as health_bp
    from semanticapi.routes.languages import bp as languages_bp
    from semanticapi.routes.similarity import bp as similarity_bp
    from semanticapi.routes.most_similar import bp as most_similar_bp
    from semanticapi.routes.analogy import bp as analogy_bp
    from semanticapi.routes.odd_one_out import bp as odd_one_out_bp
    from semanticapi.routes.word_vector import bp as word_vector_bp
    from semanticapi.routes.sentence_similarity import bp as sentence_similarity_bp
    from semanticapi.routes.cluster import bp as cluster_bp
    from semanticapi.routes.examples import bp as examples_bp

    for blueprint in (
        health_bp,
        languages_bp,
        similarity_bp,
        most_similar_bp,
        analogy_bp,
        odd_one_out_bp,
        word_vector_bp,
        sentence_similarity_bp,
        cluster_bp,
        examples_bp,
    ):
        app.register_blueprint(blueprint)

    # ── Preload models ─────────────────────────────────────────────────────────
    with app.app_context():
        preload_languages()

    return app


# Module-level app instance for Gunicorn / WSGI servers
app = create_app()
