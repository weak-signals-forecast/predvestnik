# Сервис слабых сигналов (ЛЦТ2026, Газпромбанк.Тех)
#   make dataset train timemachine   этап 1: датасет, модель, историческая проверка
#   make search Q="слабые сигналы в кибербезопасности"   этап 2 в консоли
#   make up                          docker compose: PostgreSQL + API + интерфейс
PY := .venv/bin/python
Q ?= слабые сигналы в кибербезопасности

.PHONY: dataset train timemachine-collect timemachine takeoff takeoff-score search api ui up down test-service

dataset:          ## признаки 298 технологий из открытых источников (кэшируется в data/cache)
	$(PY) -m wsignals.dataset

train:            ## обучение и отчёт reports/model_report.md
	$(PY) -m wsignals.model

timemachine-collect: ## исторические счётчики GitHub и Hacker News для проверки на 2021 год
	$(PY) -m wsignals.timemachine collect

timemachine:      ## машина времени и лестница зрелости: reports/timemachine.md
	$(PY) -m wsignals.timemachine backtest

takeoff:          ## прогноз взлёта: временная проверка и модель, reports/takeoff.md
	$(PY) -m wsignals.takeoff train

takeoff-score:    ## проверить реестр прогнозов по текущим данным (имеет смысл с 2029 года)
	$(PY) -m wsignals.takeoff score

search:           ## открытый запрос в консоли
	$(PY) -m wsignals.query "$(Q)" --json reports/last_search.json

api:              ## API на http://localhost:8000
	$(PY) -m uvicorn api.main:app --reload --port 8000

ui:               ## интерфейс на http://localhost:8501
	.venv/bin/streamlit run ui/app.py

up:               ## PostgreSQL + API + интерфейс в Docker
	docker compose up --build

down:
	docker compose down

test-service:     ## тесты сервиса без сети
	$(PY) -m pytest -q tests/test_wsignals.py

# ---------------------------------------------------------------------------------------------
# Исследовательский прототип (первая версия): корпус OpenAlex, кластеризация, прогноз на 2029–2031
FROM ?= 2015-01-01
SEEDS ?= 5

.PHONY: setup ingest patents embed cluster terms signals stabilize backtest forecast dashboard test all clean

setup:            ## виртуальное окружение и зависимости
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "Впишите OPENALEX_MAILTO (и ключ, если есть) в .env"

ingest:           ## статьи и препринты OpenAlex по всем направлениям с фильтром релевантности
	$(PY) src/ingest_openalex.py --all --from $(FROM)

patents:          ## патенты Google Patents через BigQuery (нужен GCP_PROJECT в .env)
	$(PY) src/ingest_patents.py --all --from $(FROM)

embed:            ## эмбеддинги bge-base с кэшем (считает только новые документы)
	$(PY) src/embed.py

cluster:          ## один прогон: темы, окна, сироты, признаки
	$(PY) src/cluster.py

terms:            ## лексический детектор всплеска терминов
	$(PY) src/terms.py

signals:          ## склейка трёх слоёв в карточки
	$(PY) src/signals.py

stabilize:        ## N прогонов с разными зёрнами -> устойчивый ранг (data/signals/stable.md)
	$(PY) src/stabilize.py --seeds $(SEEDS)

backtest:         ## ретроспектива лексического детектора (data/signals/backtest.md)
	$(PY) src/backtest_terms.py

forecast:         ## обучаемый предсказатель: временная валидация + прогноз на 2029-2031
	$(PY) src/forecast.py

score:            ## проверить зафиксированный прогноз по текущим данным (после 2029)
	$(PY) src/forecast.py --score data/forecast/forecast_terms.csv

dashboard:        ## самодостаточный HTML-дашборд (data/signals/dashboard.html)
	$(PY) src/dashboard.py

test:             ## юнит-тесты правил (не требуют данных)
	$(PY) -m pytest -q tests

all: ingest embed stabilize backtest forecast dashboard

clean:            ## удалить производные данные, оставить сырые
	rm -rf data/clusters data/runs data/signals_prev

help:
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*##' '{printf "  %-12s %s\n", $$1, $$2}'
