"""Страховочные тесты финальной поставки registry и встроенной витрины."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_web_uses_only_live_registry_search():
    app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert 'const API_URL = "/api/registry/search"' in app
    assert "function localSearch" not in app
    assert "function findResult" not in app
    assert "DATA.results" not in app


def test_registry_v6_is_the_compose_default_and_bundled():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile.registry").read_text(encoding="utf-8")
    assert "REGISTRY_BUNDLE:-/app/artifacts/registry_v6" in compose
    assert "COPY artifacts/registry_v6 ./artifacts/registry_v6" in dockerfile
