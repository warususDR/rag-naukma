import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

_ENV_FILE = Path(__file__).parent.parent / "lightrag_pipeline" / ".env"
load_dotenv(_ENV_FILE if _ENV_FILE.exists() else None)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
RESPONSES_FILE = EVAL_DIR / "responses.json"
SCORES_FILE = EVAL_DIR / "eval_scores.json"
SUMMARY_FILE = EVAL_DIR / "eval_summary.json"

MODES = ["hybrid", "local", "naive"]
METRIC_NAMES = ["faithfulness", "answer_correctness", "context_relevance", "context_recall"]
BATCH_SIZE = 10
SLEEP_AFTER_BATCH = 10
SLEEP_AFTER_REQUEST = 3

_RELEVANCE_PROMPT = """
Ти оцінюєш корисність (релевантність) відібраних фрагментів тексту (контекстів) (вони розділені ---) для відповіді на запитання.
Поверни ЛИШЕ одне число від 0.00 до 1.00 з двома знаками після коми, без пояснень.

Правила оцінювання (Signal-to-Noise Ratio):
- Ти оцінюєш ЯКІСТЬ ВІДБОРУ (retrieval). 
- Ближче до 0.70 - 1.00 (в залежності від числа релевантних фрагментів): хоча б 5 - 10 із усіх фрагментів містять важливу інформацію для відповіді.
- Ближче до 0.40 - 0.60: 1 - 3 корисних фрагменти з усіх.
- 0.00: Жоден із відібраних фрагментів не допомагає відповісти на запитання.

Правила нормалізації:
- Різні відмінки одного імені вважай збігом.
- OCR-артефакти в контекстах (злиті слова, підмінені символи) не вважай помилкою.
- Семантично еквівалентні формулювання вважай збігом.

Запитання: {question}

Контексти:
{contexts}

Оцінка:"""

_RECALL_PROMPT = """
Ти оцінюєш, наскільки повно відібрані контексти (вони розділені ---) містять інформацію, необхідну для формування еталонної відповіді.
Поверни ЛИШЕ одне число від 0.00 до 1.00 з двома знаками після коми, без пояснень.

Правила оцінювання:
- ближче до 0.8 - 1.00: Контексти містять майже всі або всі ключові факти еталонної відповіді
- 0.50: Контексти містять приблизно половину ключових фактів еталонної відповіді
- 0.00: Контексти не містять жодної інформації, необхідної для еталонної відповіді
Використовуй весь діапазон від 0.00 до 1.00.

Правила нормалізації:
- Різні відмінки одного імені вважай збігом.
- OCR-артефакти в контекстах (злиті слова, підмінені символи) не вважай помилкою.
- Семантично еквівалентні формулювання вважай збігом.

Запитання: {question}
Еталонна відповідь: {ground_truth}

Контексти:
{contexts}

Оцінка:"""


def load_json(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def save_scores(scores: list[dict]) -> None:
    SCORES_FILE.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")


def needs_context_metrics(entry: dict) -> bool:
    return "error" not in entry and (
        entry.get("context_relevance") is None or entry.get("context_recall") is None
    )


def gpt_score(prompt: str, client: OpenAI) -> float | None:
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=10,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip().replace(",", ".")
        return round(float(text), 2)
    except Exception as e:
        logger.warning(f"GPT scoring failed: {e}")
        return None


def build_summary(scores: list[dict]) -> dict:
    summary = {}
    for mode in MODES:
        mode_scores = [s for s in scores if s["mode"] == mode and "error" not in s]
        if not mode_scores:
            continue
        summary[mode] = {}
        for metric in METRIC_NAMES:
            vals = [s[metric] for s in mode_scores if s.get(metric) is not None]
            summary[mode][metric] = round(sum(vals) / len(vals), 4) if vals else None
        summary[mode]["n"] = len(mode_scores)
    return summary


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment / .env")

    client = OpenAI(api_key=api_key)
    logger.info("Initialized OpenAI client (gpt-4.1-mini) for context_relevance, context_recall")

    responses = load_json(RESPONSES_FILE)
    scores = load_json(SCORES_FILE)

    response_lookup = {(r["question"], r["mode"]): r for r in responses}

    pending_indices = [i for i, s in enumerate(scores) if needs_context_metrics(s)]
    logger.info(f"Total score entries: {len(scores)} | Need context metrics: {len(pending_indices)}")

    for batch_num, batch_start in enumerate(range(0, len(pending_indices), BATCH_SIZE)):
        batch_idx = pending_indices[batch_start : batch_start + BATCH_SIZE]
        logger.info(
            f"Batch {batch_num + 1} "
            f"({batch_start + 1}–{min(batch_start + BATCH_SIZE, len(pending_indices))} "
            f"of {len(pending_indices)})"
        )

        for idx in batch_idx:
            entry = scores[idx]
            key = (entry["question"], entry["mode"])
            response = response_lookup.get(key)

            if response is None:
                logger.warning(f"No response found for: {entry['question'][:60]!r} [{entry['mode']}]")
                continue

            contexts = [c for c in response.get("contexts", []) if c.strip()]
            question = entry["question"]
            ground_truth = response.get("ground_truth", "")
            contexts_text = "\n---\n".join(contexts)

            if entry.get("context_relevance") is None:
                if not contexts:
                    entry["context_relevance"] = 0.0
                    logger.info(f"  context_relevance=0.0 (empty contexts): {question[:60]!r}")
                else:
                    prompt = _RELEVANCE_PROMPT.format(question=question, contexts=contexts_text)
                    entry["context_relevance"] = gpt_score(prompt, client)
                    logger.info(f"  context_relevance={entry['context_relevance']}: {question[:60]!r}")
                    time.sleep(SLEEP_AFTER_REQUEST)

            if entry.get("context_recall") is None:
                if not contexts:
                    entry["context_recall"] = 0.0
                    logger.info(f"  context_recall=0.0 (empty contexts): {question[:60]!r}")
                else:
                    prompt = _RECALL_PROMPT.format(
                        question=question, ground_truth=ground_truth, contexts=contexts_text
                    )
                    entry["context_recall"] = gpt_score(prompt, client)
                    logger.info(f"  context_recall={entry['context_recall']}: {question[:60]!r}")
                    time.sleep(SLEEP_AFTER_REQUEST)

        save_scores(scores)
        logger.info(f"Batch {batch_num + 1} saved. Sleeping {SLEEP_AFTER_BATCH}s...")
        time.sleep(SLEEP_AFTER_BATCH)

    summary = build_summary(scores)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Evaluation complete. Summary updated.")
    for mode, vals in summary.items():
        logger.info(
            f"[{mode}] faithfulness={vals.get('faithfulness')} | "
            f"answer_correctness={vals.get('answer_correctness')} | "
            f"context_relevance={vals.get('context_relevance')} | "
            f"context_recall={vals.get('context_recall')} | "
            f"n={vals.get('n')}"
        )
    logger.info(f"Scores  saved to {SCORES_FILE}")
    logger.info(f"Summary saved to {SUMMARY_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, sys
        traceback.print_exc()
        sys.exit(1)
