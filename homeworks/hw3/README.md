# Домашнее задание 3 — Ranking

Цель — построить согласованный второй этап рекомендательной системы на Amazon
Reviews 2014, Beauty 5-core.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python ../download_data.py
jupyter lab homework.ipynb
```

Общий скрипт создаёт `../data/interactions.parquet`, `../data/items.parquet` и
`../data/users.parquet`. Для итогового эксперимента используется полный набор
данных — не передавайте `--max-reviews`.

При первом запуске ноутбук также создаёт `../data/ranking_datamart` и
`../data/ranking_cache`. В них хранятся воспроизводимые Parquet-снимки и кэш
текстовых эмбеддингов. GPU не обязателен.

## Как устроена работа

```text
retrieval-train → retrieval models → ranker-train candidates
                → labels + point-in-time features + retrieval ranks
                → CatBoost model selection on validation
train + validation → retrieval candidates → deduplication → ranking → test
```

Каждый основной программный шаг задаёт сигнатуру, формулу и правила для граничных
случаев. В той же ячейке находится вызов `tests.check_*`. Переходите дальше только
после сообщения `All good! :)`.
