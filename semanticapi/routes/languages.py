from flask import Blueprint, jsonify

from semanticapi.models import loaded_languages

bp = Blueprint("languages", __name__)


@bp.route("/languages", methods=["GET"])
def languages():
    """
    List loaded language models.
    ---
    tags:
      - Utility
    responses:
      200:
        description: Language information
        schema:
          type: object
          properties:
            loaded:
              type: array
              items:
                type: string
            note:
              type: string
    """
    return jsonify({
        "loaded": loaded_languages(),
        "note": (
            "Any ISO 639-1 (2-letter) or ISO 639-2 (3-letter) language code is accepted. "
            "The model will be downloaded from HuggingFace Hub on first use."
        ),
    })
