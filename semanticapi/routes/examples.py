from flask import Blueprint, render_template

bp = Blueprint("examples", __name__, template_folder="../../templates")


@bp.route("/examples", methods=["GET"])
def examples():
    """Serve the interactive examples page."""
    return render_template("examples.html")
