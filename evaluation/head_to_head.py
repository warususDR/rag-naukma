import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_ENV_FILE = Path(__file__).parent.parent / "lightrag_pipeline" / ".env"
load_dotenv(_ENV_FILE if _ENV_FILE.exists() else None)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR       = Path(__file__).parent
RESPONSES_FILE = EVAL_DIR / "responses.json"
OUTPUT_FILE    = EVAL_DIR / "head_to_head_results.json"

JUDGE_MODEL = "gpt-4.1-mini"
REPS = 2         
CRITERIA = ["comprehensiveness", "diversity", "empowerment"]
RETRY_DELAY = 5        
MAX_RETRIES = 3


_JUDGE_PROMPT = """\
Ти — неупереджений суддя, що оцінює якість двох відповідей на питання. \
Не враховуй стиль чи довжину — тільки змістовну якість.

Питання: {question}

Еталонна відповідь: {ground_truth}

Відповідь 1:
{answer_1}

Відповідь 2:
{answer_2}

Оціни за трьома критеріями (визначення — нижче):

- comprehensiveness: наскільки детально відповідь охоплює всі аспекти й деталі, \
присутні в еталонній відповіді?
- diversity: наскільки різноманітно відповідь висвітлює різні грані питання \
(різні джерела, точки зору, аспекти) — порівняно з тим, що передбачає еталон?
- empowerment: наскільки добре відповідь допомагає читачеві зрозуміти тему \
та прийняти обґрунтоване рішення на основі наданої інформації?

Для кожного критерію вкажи переможця: 1 (якщо краща Відповідь 1), \
2 (якщо краща Відповідь 2) або 0 (якщо вони рівноцінні або різниця несуттєва).

Правила нормалізації (застосовуй перед порівнянням):
- Різні відмінки одного імені — збіг: «Зарецьку», «Зарецької», «Зарецька» — одна особа.
- Пробіли між ініціалами не важливі: «П.В.Зернецький», «П. В. Зернецький» — збіг.
- Різні форми запису чисел: «0,5%», «0.5 відсотка» — збіг якщо значення однакове.
- Em-dash «—» та дефіс «-» вважай рівнозначними.

Поверни ЛИШЕ валідний JSON у такому форматі (без коментарів, без ``` обгортки):
{{"comprehensiveness": {{"winner": <0|1|2>, "reasoning": "<одне речення>"}}, \
"diversity": {{"winner": <0|1|2>, "reasoning": "<одне речення>"}}, \
"empowerment": {{"winner": <0|1|2>, "reasoning": "<одне речення>"}}}}"""



def load_responses() -> list[dict]:
    return json.loads(RESPONSES_FILE.read_text(encoding="utf-8"))


def pair_by_question(responses: list[dict]) -> list[dict]:
    by_q: dict[str, dict] = {}
    for r in responses:
        if r["mode"] not in ("hybrid", "naive"):
            continue
        q = r["question"]
        if q not in by_q:
            by_q[q] = {"question": q,
                       "ground_truth": r.get("ground_truth", ""),
                       "question_file": r.get("question_file", ""),
                       "hybrid": None, "naive": None}
        by_q[q][r["mode"]] = r["answer"]

    pairs = [v for v in by_q.values() if v["hybrid"] and v["naive"]]
    logger.info(f"Paired {len(pairs)} questions with both hybrid and naive answers")
    return pairs


def call_judge(question: str, ground_truth: str,
               answer_a: str, answer_b: str,
               client: OpenAI) -> dict | None:
    prompt = _JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        answer_1=answer_a,
        answer_2=answer_b,
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error (attempt {attempt+1}): {e} | raw: {raw[:200]}")
        except Exception as e:
            logger.warning(f"API error (attempt {attempt+1}): {e}")
        time.sleep(RETRY_DELAY)
    return None


def judge_question(pair: dict, client: OpenAI) -> dict:
    q  = pair["question"]
    gt = pair["ground_truth"]
    h  = pair["hybrid"]
    n  = pair["naive"]

    raw_judgments = []

    orders = [("hybrid", "naive"), ("naive", "hybrid")]
    for run_idx, (first, second) in enumerate(orders):
        a1 = h if first == "hybrid" else n
        a2 = h if second == "hybrid" else n

        verdict = call_judge(q, gt, a1, a2, client)
        if verdict is None:
            logger.warning(f"Skipping run {run_idx+1} for: {q[:60]}")
            continue

        raw_judgments.append({
            "run": run_idx + 1,
            "answer_1_mode": first,
            "answer_2_mode": second,
            "verdict": verdict,
        })

    tally: dict[str, dict[str, int]] = {c: {"hybrid": 0, "naive": 0, "tie": 0}
                                         for c in CRITERIA}

    for j in raw_judgments:
        for criterion in CRITERIA:
            if criterion not in j["verdict"]:
                continue
            w = j["verdict"][criterion].get("winner", 0)
            if w == 0:
                tally[criterion]["tie"] += 1
            elif w == 1:
                tally[criterion][j["answer_1_mode"]] += 1
            elif w == 2:
                tally[criterion][j["answer_2_mode"]] += 1

    return {
        "question":       q,
        "question_file":  pair["question_file"],
        "ground_truth":   gt,
        "raw_judgments":  raw_judgments,
        "tally":          tally,
    }


def aggregate(results: list[dict]) -> dict:
    """Compute overall and per-file win-rate percentages."""
    def stats_from_results(subset: list[dict]) -> dict:
        out = {}
        for criterion in CRITERIA:
            h_wins = sum(r["tally"][criterion]["hybrid"] for r in subset)
            n_wins = sum(r["tally"][criterion]["naive"]  for r in subset)
            ties   = sum(r["tally"][criterion]["tie"]    for r in subset)
            total  = h_wins + n_wins + ties
            out[criterion] = {
                "hybrid_wins":     h_wins,
                "naive_wins":      n_wins,
                "ties":            ties,
                "total_judgments": total,
                "hybrid_win_pct":  round(100 * h_wins / total, 1) if total else None,
                "naive_win_pct":   round(100 * n_wins / total, 1) if total else None,
                "tie_pct":         round(100 * ties   / total, 1) if total else None,
            }
        return out

    by_file: dict[str, list[dict]] = {}
    for r in results:
        f = r["question_file"]
        by_file.setdefault(f, []).append(r)

    return {
        "overall":  stats_from_results(results),
        "n_questions": len(results),
        "by_file":  {f: stats_from_results(items) for f, items in sorted(by_file.items())},
    }

def main():
    client = OpenAI()

    pairs = pair_by_question(load_responses())
    if not pairs:
        logger.error("No paired questions found – check responses.json")
        return

    existing: list[dict] = []
    if OUTPUT_FILE.exists():
        saved = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        existing = saved.get("results", [])
        logger.info(f"Resuming: {len(existing)} already judged")

    done_questions = {r["question"] for r in existing}
    todo = [p for p in pairs if p["question"] not in done_questions]
    logger.info(f"Remaining: {len(todo)} questions to judge")

    results = list(existing)
    for i, pair in enumerate(todo, 1):
        logger.info(f"[{i}/{len(todo)}] {pair['question'][:70]}")
        result = judge_question(pair, client)
        results.append(result)

        agg = aggregate(results)
        OUTPUT_FILE.write_text(
            json.dumps({"aggregate": agg, "results": results},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        time.sleep(0.3)  

    agg = aggregate(results)
    logger.info(f"Total questions: {agg['n_questions']}")
    for criterion, s in agg["overall"].items():
        logger.info(
            f"{criterion:14s}  hybrid {s['hybrid_win_pct']}%  "
            f"naive {s['naive_win_pct']}%  tie {s['tie_pct']}%"
        )
    logger.info(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
