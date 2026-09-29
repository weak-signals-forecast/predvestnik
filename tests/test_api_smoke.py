"""Герметичный смоук API и хранилища. Пропускается, если сервисные зависимости не установлены
(в CI-профиле юнит-тестов нет fastapi и sqlalchemy)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")


def test_api_and_storage_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setattr("wsignals.db.ROOT", tmp_path)
    monkeypatch.setattr("wsignals.db._engine", None)
    (tmp_path / "data").mkdir(exist_ok=True)
    from scripts.smoke_api import main
    assert main() == 0
