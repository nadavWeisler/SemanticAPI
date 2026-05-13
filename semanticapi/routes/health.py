from flask import Blueprint, jsonify

bp = Blueprint("health", __name__)


@bp.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    ---
    tags:
      - Utility
    responses:
      200:
        description: Service is healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
    """
    return jsonify({"status": "ok"})
