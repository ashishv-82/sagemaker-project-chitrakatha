"""SageMaker Processing step: Bronze → Silver (corpus + training split).

Why: Raw data in Bronze is heterogeneous (VTT, XLSX, plain text) and may
     contain encoding issues, duplicates, or empty documents. This step
     produces two clean, language-tagged JSONL outputs in Silver:

     S3 Silver /corpus/   — full-text chunks for RAG embedding (Flow A)
     S3 Silver /training/ — same content, chunk-sized, for RAFT synthesis (Flow B)

     Running as a SageMaker Processing Job (SKLearnProcessor) ensures:
       - Reproducible execution with SageMaker Lineage tracking.
       - Automatic retry and spot interruption recovery.
       - Isolated compute that dies after the job — no persistent infra.

Operations per document:
    1. Parse source format (.vtt strip timestamps, .xlsx flatten, .txt pass-through)
    2. NFC Unicode normalise — preserve Devanagari, never strip non-ASCII
    3. De-duplicate by SHA-256 content hash (MD5 used by upload_to_bronze is
       not collision-resistant enough for dedup decisions)
    4. Language-tag: "en", "hi", or "en-hi" (detected by Devanagari codepoint presence)
    5. Write to Silver /corpus/ (full text) and /training/ (chunked)

Constraints:
    - Raises DataIngestionError on completely unreadable documents.
    - Exits with code 1 if > 0 documents fail (SageMaker marks step FAILED).
    - Input/output paths follow SageMaker Processing Job channel conventions
      (/opt/ml/processing/input/bronze and /opt/ml/processing/output/).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

# openpyxl is not pre-installed in the SKLearnProcessor container.
# Install it before any imports that depend on it (e.g. pandas Excel reader).
subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl>=3.1.0", "--quiet"])

from chitrakatha.exceptions import DataIngestionError
from chitrakatha.ingestion.chunker import chunk_text

logger = logging.getLogger(__name__)

# SageMaker Processing Job channel mount points.
INPUT_DIR = Path("/opt/ml/processing/input/bronze")
CORPUS_OUTPUT_DIR = Path("/opt/ml/processing/output/corpus")
TRAINING_OUTPUT_DIR = Path("/opt/ml/processing/output/training")

# Devanagari Unicode block: U+0900–U+097F
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_VTT_TIMESTAMP_RE = re.compile(
    r"WEBVTT.*?\n|NOTE\s.*?\n\n|\d{2}:\d{2}[\d:.]+\s-->\s[\d:.]+[^\n]*\n",
    re.DOTALL,
)


def _detect_language(text: str) -> str:
    """Detect document language from character composition.

    Args:
        text: Document text (UTF-8).

    Returns:
        ``"hi"`` if only Devanagari detected, ``"en-hi"`` if mixed,
        ``"en"`` if no Devanagari characters found.
    """
    has_devanagari = bool(_DEVANAGARI_RE.search(text))
    has_ascii_words = bool(re.search(r"[a-zA-Z]{3,}", text))
    if has_devanagari and has_ascii_words:
        return "en-hi"
    if has_devanagari:
        return "hi"
    return "en"


def _parse_raw(path: Path) -> str:
    """Parse a raw source file to plain UTF-8 text.

    Args:
        path: Path to the source file within the Processing Job input directory.

    Returns:
        Cleaned UTF-8 text with Devanagari preserved.

    Raises:
        DataIngestionError: If the file cannot be read or produces empty output.
    """
    suffix = path.suffix.lower()
    try:
        raw = path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError) as exc:
        raise DataIngestionError(f"Cannot read {path}: {exc}") from exc

    if suffix == ".vtt":
        raw = _VTT_TIMESTAMP_RE.sub("", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

    elif suffix == ".xlsx":
        # openpyxl must be available in the Processing container.
        try:
            import openpyxl  # noqa: PLC0415
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            lines = []
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    row_text = "\t".join(str(c) for c in row if c is not None)
                    if row_text.strip():
                        lines.append(row_text)
            raw = "\n".join(lines)
        except Exception as exc:
            raise DataIngestionError(f"Cannot parse Excel {path}: {exc}") from exc

    # NFC normalisation — safe for Devanagari.
    raw = unicodedata.normalize("NFC", raw)

    if not raw.strip():
        raise DataIngestionError(f"Document {path.name} is empty after parsing.")

    return raw


def _sha256(text: str) -> str:
    """Return the hex SHA-256 digest of a UTF-8 string (for dedup)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process(input_dir: Path, corpus_out: Path, training_out: Path) -> int:
    """Process all raw files from Bronze and write Silver JSONL outputs.

    Args:
        input_dir: Root directory of the Bronze input channel.
        corpus_out: Output directory for Silver /corpus/ JSONL.
        training_out: Output directory for Silver /training/ JSONL.

    Returns:
        Count of documents that failed processing.
    """
    corpus_out.mkdir(parents=True, exist_ok=True)
    training_out.mkdir(parents=True, exist_ok=True)

    supported_suffixes = {".txt", ".md", ".vtt", ".xlsx"}
    source_files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in supported_suffixes
    ]
    logger.info("Found %d source file(s) in Bronze input channel.", len(source_files))

    seen_hashes: set[str] = set()
    corpus_lines: list[str] = []
    training_lines: list[str] = []
    errors = 0

    for path in sorted(source_files):
        try:
            text = _parse_raw(path)
        except DataIngestionError as exc:
            logger.error("Skipping %s: %s", path.name, exc)
            errors += 1
            continue

        content_hash = _sha256(text)
        if content_hash in seen_hashes:
            logger.info("Duplicate content detected — skipping %s.", path.name)
            continue
        seen_hashes.add(content_hash)

        language = _detect_language(text)

        # Flow A: full-text corpus record for RAG embedding.
        corpus_record = json.dumps(
            {
                "text": text,
                "source_document": path.name,
                "language": language,
                "content_hash": content_hash,
            },
            ensure_ascii=False,
        )
        corpus_lines.append(corpus_record)

        # Flow B: chunked records for RAFT synthesis input.
        try:
            chunks = chunk_text(text, source_document=path.name)
        except DataIngestionError as exc:
            logger.error("Chunking failed for %s: %s", path.name, exc)
            errors += 1
            continue

        for chunk in chunks:
            training_record = json.dumps(
                {
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "source_document": path.name,
                    "language": language,
                    "content_hash": content_hash,
                },
                ensure_ascii=False,
            )
            training_lines.append(training_record)

    # Write corpus JSONL (one file per Processing Job run).
    corpus_path = corpus_out / "corpus.jsonl"
    corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")
    logger.info("Wrote %d corpus record(s) to %s.", len(corpus_lines), corpus_path)

    # Write training JSONL.
    training_path = training_out / "training.jsonl"
    training_path.write_text("\n".join(training_lines), encoding="utf-8")
    logger.info("Wrote %d training chunk(s) to %s.", len(training_lines), training_path)

    return errors


def main() -> None:
    """SageMaker Processing Job entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    error_count = process(INPUT_DIR, CORPUS_OUTPUT_DIR, TRAINING_OUTPUT_DIR)
    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()
