import argparse
import logging
from pathlib import Path

import pymupdf4llm
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


def extract_text_md(pdf_path: Path) -> str:
    return pymupdf4llm.to_markdown(str(pdf_path))


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


def extract_all_md(docs_dir: Path, output_dir: Path, max_files: int | None = None) -> tuple[int, int]:
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if max_files:
        pdf_files = pdf_files[:max_files]

    output_dir.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, 0
    total = len(pdf_files)

    for i, pdf_path in enumerate(pdf_files, 1):
        out_path = output_dir / (pdf_path.stem + ".md")

        if out_path.exists():
            logger.info(f"[{i}/{total}] Skipping (already exists): {pdf_path.name}")
            ok += 1
            continue

        try:
            md = extract_text_md(pdf_path)
            if not md.strip():
                logger.warning(f"[{i}/{total}] Empty output: {pdf_path.name}")
                failed += 1
                continue

            out_path.write_text(md, encoding="utf-8")
            logger.info(f"[{i}/{total}] Extracted {len(md):,} chars → {out_path.name}")
            ok += 1
        except Exception as e:
            logger.error(f"[{i}/{total}] Failed {pdf_path.name}: {e}")
            failed += 1

    return ok, failed


def main():
    parser = argparse.ArgumentParser(description="Extract PDF text layer to .txt or .md files")
    parser.add_argument(
        "--docs-dir",
        default=str(Path(__file__).parent.parent / "documents"),
        help="Directory containing PDF files (default: ../documents)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "texts"),
        help="Directory to write output files into (default: lightrag_pipeline/texts/)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of PDFs to process (for testing)",
    )
    parser.add_argument(
        "--pymupdf",
        action="store_true",
        help="Use pymupdf4llm to extract as Markdown into texts_pymupdf/ instead of pypdf .txt",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)

    if args.pymupdf:
        output_dir = Path(__file__).parent / "texts_pymupdf"
    else:
        output_dir = Path(args.output_dir)

    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        return

    pdf_count = len(list(docs_dir.glob("*.pdf")))
    logger.info(f"Found {pdf_count} PDFs in {docs_dir}")
    logger.info(f"Output directory: {output_dir}")

    if args.pymupdf:
        ok, failed = extract_all_md(docs_dir, output_dir, args.max_files)
        ext = ".md"
    else:
        ok, failed = extract_all(docs_dir, output_dir, args.max_files)
        ext = ".txt"

    logger.info(f"\nDone: {ok} extracted, {failed} failed")
    if ok:
        logger.info(f"Files ({ext}) are ready in: {output_dir}")
        logger.info("Pass this directory to LightRAGModel.insert_from_text_dir() to ingest into LightRAG.")


if __name__ == "__main__":
    main()
