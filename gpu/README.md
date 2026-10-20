# Расчёты на GPU (A100)

Тяжёлая часть конвейера, которая считается один раз при обновлении корпуса на сервере с видеокартой. Скрипты не
зависят от остального кода (только torch и transformers), поэтому запускаются в стандартном контейнере
`huggingface/transformers-pytorch-gpu` без установки окружения. Входы и выходы — небольшие файлы, которые готовит и
разбирает код `registry/` и `scripts/` на обычном сервере.

| Скрипт | Что считает | Вход → выход |
|---|---|---|
| `embed_docs_hf.py` | векторы всех документов корпуса (BAAI/bge-base-en-v1.5) | корпус → `doc_emb/*.npz` |
| `prior_sim_gpu.py` | для каждой свежей работы — самая близкая работа из прошлого | `doc_emb/` → `prior_sim_<год>.npz` |
| `novelty_export.py` | новизна каждого свежего документа, кластеры работ без предшественников | → `novelty_<год>.npz` → `python -m registry.doc_novelty` |
| `llm_local.py` | открытая модель (Qwen2.5-7B/14B-Instruct): русские названия, описания, LLM-оси, оценка выдачи | промпты `registry.llm_offline prompts`, `scripts/judge_eval.py prompts` → ответы → `... import` / `score` |
| `teacher_scores.py` | кросс-энкодер BAAI/bge-reranker-v2-m3 — учитель для адаптера запроса | `scripts/distill_adapter.py inputs` → `teacher.npz` → `... train` |

Пример (одна A100 40 ГБ):

```bash
GPB_CORPUS=/data/gpb_corpus DOC_EMB=/data/doc_emb python3 embed_docs_hf.py --batch 512
DOC_EMB=/data/doc_emb python3 prior_sim_gpu.py --cutoff 2026
DOC_EMB=/data/doc_emb python3 novelty_export.py --cutoff 2026
python3 llm_local.py --model Qwen/Qwen2.5-7B-Instruct --inp prompts.jsonl --out answers.jsonl --batch 6
python3 teacher_scores.py --dir distill/ --out distill/teacher.npz
```

Все скрипты продолжают с места остановки при повторном запуске; файлы промптов можно делить на части
(`split -n l/4`) и считать параллельно на нескольких картах.
