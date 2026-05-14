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
BATCH_SIZE = 10
SLEEP_AFTER_REQUEST = 3
SLEEP_AFTER_BATCH = 5

_FAITHFULNESS_PROMPT = """
Ти оцінюєш, наскільки відповідь ґрунтується ВИКЛЮЧНО на наданих контекстах (розділених ---).
Поверни ЛИШЕ одне число від 0.00 до 1.00 з двома знаками після коми, без пояснень.

Правила:
- 1.00: кожне твердження відповіді прямо підтверджено хоча б одним контекстом
- 0.50: приблизно половина тверджень підтверджена контекстами або твердження містять вигадану інформацію
- 0.00: відповідь суперечить контекстам або жодне твердження не підкріплено контекстами
Використовуй ВЕСЬ діапазон від 0.00 до 1.00 — наприклад 0.10, 0.35, 0.70, 0.85 тощо, спираючись на надані правила.
Не обмежуйся лише значеннями 0.00, 0.50 та 1.00.

Правила нормалізації (застосовуй перед порівнянням):
- Різні відмінки одного імені — збіг: «Зарецьку», «Зарецької», «Зарецька» — одна особа.
- Пробіли між ініціалами не важливі: «П.В.Зернецький», «П. В. Зернецький», «Зернецький П.В.» — збіг.
- OCR-артефакти в контекстах: «1» замість «і» (комун1кативних = комунікативних), «З.» замість «з», злиті/розділені слова — не вважай помилкою.
- Різні форми запису чисел: «0,5%», «0.5 відсотка», «½ відсотка» — збіг якщо значення однакове.
- Em-dash «—» та дефіс «-» вважай рівнозначними.

Контексти:
{contexts}

Відповідь: {answer}

Оцінка:"""

_CORRECTNESS_PROMPT = """
Ти оцінюєш якість відповіді на питання, порівнюючи її з еталонною відповіддю.
Поверни ЛИШЕ одне число від 0.00 до 1.00 з двома знаками після коми, без пояснень.

Правила оцінювання:
- Якщо відповідь містить всі факти еталонної відповіді і нічого зайвого — близько до 1.00
- Якщо відповідь містить всі факти еталонної відповіді, але також додаткову інформацію якої немає в еталонній відповіді — знімай 0.10–0.20
- Якщо відповідь містить лише частину фактів — відповідно менше
- Якщо відповідь хибна або не стосується питання — 0.00
Використовуй весь діапазон від 0.00 до 1.00.

Правила нормалізації (застосовуй перед порівнянням):
- Різні відмінки одного імені — збіг: «Зарецьку», «Зарецької», «Зарецька» — одна особа.
- Пробіли між ініціалами не важливі: «П.В.Зернецький», «П. В. Зернецький», «Зернецький П.В.» — збіг.
- Різні форми запису чисел та дат — збіг якщо значення однакове.
- Семантично еквівалентні формулювання вважай збігом.

Питання: {question}
Еталонна відповідь: {ground_truth}
Відповідь системи: {answer}

Оцінка:"""


def load_json(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def save_scores(scores: list[dict]) -> None:
    SCORES_FILE.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")


def needs_v2(entry: dict) -> bool:
    return "error" not in entry and (
        entry.get("faithfulness_v2") is None or entry.get("answer_correctness_v2") is None
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
    metric_sets = [
        ("faithfulness", "answer_correctness"),
        ("faithfulness_v2", "answer_correctness_v2"),
        ("context_relevance", "context_recall"),
    ]
    summary = {}
    for mode in MODES:
        ms = [s for s in scores if s["mode"] == mode and "error" not in s]
        if not ms:
            continue
        summary[mode] = {"n": len(ms)}
        for metric_group in metric_sets:
            for metric in metric_group:
                vals = [s[metric] for s in ms if s.get(metric) is not None]
                summary[mode][metric] = round(sum(vals) / len(vals), 4) if vals else None
    return summary


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment / .env")

    client = OpenAI(api_key=api_key)
    logger.info("Initialized OpenAI client (gpt-4.1-mini) for faithfulness_v2, answer_correctness_v2")

    responses = load_json(RESPONSES_FILE)
    scores = load_json(SCORES_FILE)

    response_lookup = {(r["question"], r["mode"]): r for r in responses}

    pending_indices = [i for i, s in enumerate(scores) if needs_v2(s)]
    logger.info(f"Total score entries: {len(scores)} | Need v2 scoring: {len(pending_indices)}")

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
            answer = response.get("answer", "")
            contexts_text = "\n---\n".join(contexts)

            if entry.get("faithfulness_v2") is None:
                if not contexts or not answer:
                    entry["faithfulness_v2"] = 0.0
                    logger.info(f"  faithfulness_v2=0.0 (empty): {question[:60]!r}")
                else:
                    prompt = _FAITHFULNESS_PROMPT.format(contexts=contexts_text, answer=answer)
                    entry["faithfulness_v2"] = gpt_score(prompt, client)
                    logger.info(f"  faithfulness_v2={entry['faithfulness_v2']}: {question[:60]!r}")
                    time.sleep(SLEEP_AFTER_REQUEST)

            if entry.get("answer_correctness_v2") is None:
                if not answer:
                    entry["answer_correctness_v2"] = 0.0
                    logger.info(f"  answer_correctness_v2=0.0 (empty): {question[:60]!r}")
                else:
                    prompt = _CORRECTNESS_PROMPT.format(
                        question=question, ground_truth=ground_truth, answer=answer
                    )
                    entry["answer_correctness_v2"] = gpt_score(prompt, client)
                    logger.info(f"  answer_correctness_v2={entry['answer_correctness_v2']}: {question[:60]!r}")
                    time.sleep(SLEEP_AFTER_REQUEST)

        save_scores(scores)
        logger.info(f"Batch {batch_num + 1} saved. Sleeping {SLEEP_AFTER_BATCH}s...")
        time.sleep(SLEEP_AFTER_BATCH)

    summary = build_summary(scores)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Rescoring complete. Summary updated with v2 metrics.")
    for mode, vals in summary.items():
        logger.info(
            f"[{mode}] "
            f"faithfulness={vals.get('faithfulness')} -> v2={vals.get('faithfulness_v2')} | "
            f"answer_correctness={vals.get('answer_correctness')} -> v2={vals.get('answer_correctness_v2')} | "
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
