"""
Extract text from PDFs in documents/ and save as .txt files for LightRAG ingestion.

Usage:
    python extract_documents.py [--docs-dir DIR] [--output-dir DIR] [--max-files N]

Output directory defaults to lightrag_pipeline/texts/ and can be passed directly
to LightRAGModel.insert_from_text_dir().
"""
import argparse
import logging
import os
from pathlib import Path

from pypdf import PdfReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_all(docs_dir: Path, output_dir: Path, max_files: int | None = None) -> tuple[int, int]:
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if max_files:
        pdf_files = pdf_files[:max_files]

    output_dir.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, 0
    total = len(pdf_files)

    for i, pdf_path in enumerate(pdf_files, 1):
        out_path = output_dir / (pdf_path.stem + ".txt")

        if out_path.exists():
            logger.info(f"[{i}/{total}] Skipping (already exists): {pdf_path.name}")
            ok += 1
            continue

        try:
            text = extract_text(pdf_path)
            if not text.strip():
                logger.warning(f"[{i}/{total}] Empty text: {pdf_path.name}")
                failed += 1
                continue

            out_path.write_text(text, encoding="utf-8")
            logger.info(f"[{i}/{total}] Extracted {len(text):,} chars → {out_path.name}")
            ok += 1
        except Exception as e:
            logger.error(f"[{i}/{total}] Failed {pdf_path.name}: {e}")
            failed += 1

    return ok, failed


def main():
    parser = argparse.ArgumentParser(description="Extract PDF text layer to .txt files")
    parser.add_argument(
        "--docs-dir",
        default=str(Path(__file__).parent.parent / "documents"),
        help="Directory containing PDF files (default: ../documents)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "texts"),
        help="Directory to write .txt files into (default: lightrag_pipeline/texts/)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of PDFs to process (useful for testing)",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    output_dir = Path(args.output_dir)

    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        return

    pdf_count = len(list(docs_dir.glob("*.pdf")))
    logger.info(f"Found {pdf_count} PDFs in {docs_dir}")
    logger.info(f"Output directory: {output_dir}")

    ok, failed = extract_all(docs_dir, output_dir, args.max_files)

    logger.info(f"\nDone: {ok} extracted, {failed} failed")
    if ok:
        logger.info(f"Text files are ready in: {output_dir}")
        logger.info("Pass this directory to LightRAGModel.insert_from_text_dir() to ingest into LightRAG.")


if __name__ == "__main__":
    main()
