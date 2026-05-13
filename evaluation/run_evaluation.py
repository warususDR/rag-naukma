import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import logging
import os
import time
from pathlib import Path
import numpy as np
import requests
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
METRIC_NAMES = ["faithfulness", "answer_correctness"]

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_EMBED_MODEL = "bge-m3"

BATCH_SIZE = 5
TOP_K = 15

_FAITHFULNESS_PROMPT = """
Ти оцінюєш, наскільки відповідь ґрунтується ВИКЛЮЧНО на наданих контекстах.
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

def load_responses() -> list[dict]:
    return json.loads(RESPONSES_FILE.read_text(encoding="utf-8"))


def load_scores() -> list[dict]:
    if SCORES_FILE.exists():
        try:
            return json.loads(SCORES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_scores(scores: list[dict]) -> None:
    SCORES_FILE.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")


def already_scored(scores: list[dict], question: str, mode: str) -> bool:
    return any(
        s["question"] == question and s["mode"] == mode and "error" not in s
        for s in scores
    )


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


def _bge_embed(texts: list[str], batch_size: int = 4) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        r = requests.post(OLLAMA_EMBED_URL, json={"model": OLLAMA_EMBED_MODEL, "input": chunk}, timeout=120)
        r.raise_for_status()
        all_embs.extend(r.json()["embeddings"])
    return np.array(all_embs)


def _bge_top_k_chunks(answer: str, contexts: list[str], k: int) -> list[str]:
    ctx_chunks = [c for c in contexts if c.strip()]
    if not ctx_chunks:
        return []
    embs = _bge_embed([answer] + ctx_chunks)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    sims = embs[1:] @ embs[0]
    top_indices = np.argsort(sims)[::-1][:k]
    return [ctx_chunks[i] for i in top_indices]


def gpt_faithfulness(answers: list[str], contexts_list: list[list[str]], client: OpenAI) -> list[float]:
    scores = []
    for answer, contexts in zip(answers, contexts_list):
        try:
            top_chunks = _bge_top_k_chunks(answer, contexts, k=TOP_K)
            if not top_chunks:
                scores.append(None)
                continue
            prompt = _FAITHFULNESS_PROMPT.format(
                contexts="\n---\n".join(top_chunks),
                answer=answer,
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            text = resp.choices[0].message.content.strip().replace(",", ".")
            scores.append(round(float(text), 2))
        except Exception as e:
            logger.warning(f"GPT faithfulness failed: {e}")
            scores.append(None)
    return scores


def gpt_answer_correctness(
    questions: list[str], answers: list[str], ground_truths: list[str], client: OpenAI
) -> list[float]:
    scores = []
    for question, answer, gt in zip(questions, answers, ground_truths):
        try:
            prompt = _CORRECTNESS_PROMPT.format(question=question, ground_truth=gt, answer=answer)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            text = resp.choices[0].message.content.strip().replace(",", ".")
            scores.append(round(float(text), 2))
        except Exception as e:
            logger.warning(f"GPT answer_correctness failed: {e}")
            scores.append(None)
    return scores


def evaluate_batch(batch: list[dict], openai_client: OpenAI) -> list[dict]:
    answers       = [e["answer"]       for e in batch]
    questions     = [e["question"]     for e in batch]
    ground_truths = [e["ground_truth"] for e in batch]
    contexts_list = [e["contexts"]     for e in batch]

    try:
        faithfulness_scores = gpt_faithfulness(answers, contexts_list, openai_client)
    except Exception as e:
        logger.warning(f"GPT faithfulness failed: {e}")
        faithfulness_scores = [None] * len(batch)

    try:
        correctness_scores = gpt_answer_correctness(questions, answers, ground_truths, openai_client)
    except Exception as e:
        logger.warning(f"GPT answer_correctness failed: {e}")
        correctness_scores = [None] * len(batch)

    scored = []
    for i, entry in enumerate(batch):
        scored.append({
            "question":           entry["question"],
            "mode":               entry["mode"],
            "question_file":      entry.get("question_file", ""),
            "faithfulness":       round(faithfulness_scores[i], 4) if faithfulness_scores[i] is not None else None,
            "answer_correctness": round(float(correctness_scores[i]), 4) if correctness_scores[i] is not None else None,
        })
    return scored


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment / .env")

    client = OpenAI(api_key=api_key)
    logger.info("Initialized OpenAI client (gpt-4o-mini) for faithfulness, answer_correctness")

    responses = load_responses()
    scores    = load_scores()
    logger.info(f"Responses: {len(responses)} | Already scored: {len([s for s in scores if 'error' not in s])}")

    for mode in MODES:
        mode_responses = [r for r in responses if r["mode"] == mode]
        pending = [r for r in mode_responses if not already_scored(scores, r["question"], mode)]
        logger.info(f"[{mode}] {len(mode_responses) - len(pending)}/{len(mode_responses)} done, {len(pending)} remaining")

        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start : batch_start + BATCH_SIZE]
            logger.info(
                f"[{mode}] Evaluating batch {batch_start // BATCH_SIZE + 1} "
                f"({batch_start + 1}–{min(batch_start + BATCH_SIZE, len(pending))} of {len(pending)})"
            )
            try:
                scores.extend(evaluate_batch(batch, client))
                save_scores(scores)
                logger.info(f"[{mode}] Batch saved. Total scored: {len([s for s in scores if 'error' not in s])}")
                time.sleep(10)
            except Exception as e:
                logger.error(f"[{mode}] Batch failed: {e}")
                for entry in batch:
                    scores.append({
                        "question":      entry["question"],
                        "mode":          mode,
                        "question_file": entry.get("question_file", ""),
                        "error":         str(e),
                    })
                save_scores(scores)

    summary = build_summary(scores)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Evaluation complete")
    for mode, vals in summary.items():
        logger.info(
            f"[{mode}] faithfulness={vals.get('faithfulness')} | "
            f"answer_correctness={vals.get('answer_correctness')} | "
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
