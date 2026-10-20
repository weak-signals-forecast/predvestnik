#!/usr/bin/env python3
"""Сквозная проверка финального registry-контейнера без внешнего LLM."""
from __future__ import annotations

import json
import sys
import urllib.request


BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010").rstrip("/")


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=30) as response:
        assert response.status == 200, response.status
        return json.load(response)


def search(query: str) -> dict:
    request = urllib.request.Request(
        BASE + "/api/registry/search",
        data=json.dumps({"query": query, "use_llm": False}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        assert response.status == 200, response.status
        return json.load(response)


health = get("/api/registry/health")
assert health["base_size"] == 9437, health
assert health["cards"] == 9437, health

for query in ("многослойный графен", "водородные хранилища", "постквантовая криптография"):
    result = search(query)
    signals = result["signals"]
    assert result["ranking"] == "v2", result["ranking"]
    assert len(signals) == 15, (query, len(signals))
    assert all(signal.get("name") and isinstance(signal.get("rating"), dict) for signal in signals), query
    for signal in signals:
        for source in signal.get("sources") or []:
            url = source.get("url") or ""
            assert url.startswith(("https://", "http://")), (query, signal["name"], url)
            assert url.rstrip("/").lower() not in {"https://arxiv.org", "http://arxiv.org"}, url
    print(query, "—", len(signals), "карточек; первая:", signals[0]["name"])

print("registry release smoke: OK")
