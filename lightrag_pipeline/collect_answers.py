import json
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from lightrag_model import LightRAGModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent.parent / "evaluation"
OUTPUT_FILE = EVAL_DIR / "responses.json"
MODES = ["hybrid", "local", "naive"]

QUESTION_FILES = sorted(EVAL_DIR.glob("questions_batch*.json")) + [
    EVAL_DIR / "questions_reasoning.json",
    EVAL_DIR / "questions_reasoning2.json",
    EVAL_DIR / "questions_edge.json",
]


def load_questions() -> list[dict]:
    questions = []
    for path in QUESTION_FILES:
        if not path.exists():
            logger.warning(f"File not found, skipping: {path.name}")
            continue
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            item["_source_file"] = path.name
        questions.extend(items)
        logger.info(f"Loaded {len(items)} questions from {path.name}")
    logger.info(f"Total questions: {len(questions)}")
    return questions


def load_existing_responses() -> list[dict]:
    if OUTPUT_FILE.exists():
        try:
            return json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def already_done(responses: list[dict], question: str, mode: str) -> bool:
    return any(r["question"] == question and r["mode"] == mode for r in responses)


def append_response(responses: list[dict], entry: dict) -> None:
    responses.append(entry)
    OUTPUT_FILE.write_text(
        json.dumps(responses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    logger.info("Initializing LightRAG model...")
    rag = LightRAGModel(
        embedding_model="bge-m3",
        llm_model_name="mamaylum-12b",
        chroma_directory="../chroma_db",
        chroma_collection="naukma_documents_no_chunks",
    )
    logger.info("Model ready.")

    questions = load_questions()
    responses = load_existing_responses()
    skipped = 0

    total = len(questions) * len(MODES)
    done = len(responses)
    logger.info(f"Already collected: {done}/{total}. Resuming...")

    for mode in MODES:
        for i, item in enumerate(questions):
            question = item["question"]
            ground_truth = item.get("ground_truth", "")
            source_files = item.get("source_files", [])
            source_file = item.get("_source_file", "")

            if already_done(responses, question, mode):
                skipped += 1
                continue

            idx = len(responses) + 1
            logger.info(f"[{idx}/{total}] mode={mode} | {question[:80]}...")

            try:
                result = rag.query(question, mode=mode, history=None)
                entry = {
                    "question": question,
                    "answer": result["response"],
                    "contexts": result.get("chunks", []),
                    "ground_truth": ground_truth,
                    "source_files": source_files,
                    "references": result.get("references", []),
                    "mode": mode,
                    "question_file": source_file,
                }
                append_response(responses, entry)
                logger.info(
                    f"  -> saved (contexts={len(entry['contexts'])}, "
                    f"answer_len={len(entry['answer'])})"
                )
            except Exception as e:
                logger.error(f"  -> FAILED: {e}")
                entry = {
                    "question": question,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": ground_truth,
                    "source_files": source_files,
                    "references": [],
                    "mode": mode,
                    "question_file": source_file,
                    "error": str(e),
                }
                append_response(responses, entry)

            time.sleep(0.5)

    logger.info(
        f"Done. Total collected: {len(responses)}, skipped (already done): {skipped}"
    )
    logger.info(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
