# Доступ к Google Patents Public Data через BigQuery

Датасет `patents-public-data.patents.publications` открытый, но запросы к нему
выполняются из вашего проекта Google Cloud. Бесплатно: 1 ТБ обработанных данных в месяц.

## Шаги (10 минут)

1. Зайти на https://console.cloud.google.com под Google-аккаунтом, создать проект,
   например `weak-signals`. Запомнить его ID (не имя), он вида `weak-signals-123456`.
2. Включить API: меню "APIs & Services" -> "Enable APIs" -> найти "BigQuery API" -> Enable.
   Привязка платёжного аккаунта не обязательна для бесплатного тира, но без неё
   консоль может попросить её включить. Лимит в скрипте защищает от счёта.
3. Создать сервисный аккаунт: "IAM & Admin" -> "Service Accounts" -> "Create".
   Роль: **BigQuery Job User** (этого достаточно, чтобы читать публичные датасеты).
4. Открыть созданный аккаунт -> вкладка "Keys" -> "Add key" -> JSON. Файл скачается.
5. Переложить файл в `~/.gcp/weak-signals.json` (или куда угодно вне репозитория)
   и прописать в `.env`:

       GCP_PROJECT=weak-signals-123456
       GOOGLE_APPLICATION_CREDENTIALS=/Users/margo/.gcp/weak-signals.json

## Проверка

    .venv/bin/python src/ingest_patents.py --domain hydrogen --dry-run

Покажет, сколько ГБ обработает запрос, ничего не скачивая. Затем полный сбор:

    .venv/bin/python src/ingest_patents.py --all --from 2015-01-01

## Про стоимость

Таблица publications большая, но запрос читает только нужные колонки. Ожидаемо
20-80 ГБ на направление. Четыре направления укладываются в бесплатный терабайт
с большим запасом. Переменная BQ_MAX_BYTES_BILLED в .env обрывает любой запрос
дороже 50 ГБ, поднимайте её осознанно.
