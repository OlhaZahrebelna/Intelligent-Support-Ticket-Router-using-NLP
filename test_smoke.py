"""Basic import and route smoke tests.

Run from the repository root:
    pytest -q
"""

from app.main import app


def test_app_imports() -> None:
    assert app.title == "Intelligent Support Ticket Router API"
    assert app.version == "2.0.0"


def test_required_routes_exist() -> None:
    paths = {route.path for route in app.routes}

    assert "/health" in paths
    assert "/predict" in paths
    assert "/predict_batch" in paths
    assert "/review/confirm" in paths
