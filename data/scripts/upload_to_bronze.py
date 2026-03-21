"""Upload raw source files to the S3 Bronze bucket.

Why: The Bronze bucket is the single entry point for all raw data. This script
     validates encoding, parses supported formats, computes checksums for
     lineage tracking, and uploads WITH metadata — so every object in Bronze
     is self-describing and reproducible.

Supported formats:
    - ``.txt`` / ``.md``   — plain UTF-8 text, passed through directly.
    - ``.vtt``             — WebVTT transcript; timestamps stripped, text retained.
    - ``.xlsx``            — Excel workbook; all sheets flattened to UTF-8 text.

Constraints:
    - Never strip Devanagari (Hindi) or any non-ASCII character.
    - Validate UTF-8 **before** uploading; reject and log any invalid files.
    - Store MD5 checksum in S3 object metadata (``x-amz-meta-md5``) for
      downstream lineage tracking in SageMaker.
    - All config (bucket name, KMS key ARN) read from environment — no
      hardcoded values.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import sys
from pathlib import Path
from typing import Final

import boto3
import openpyxl
from botocore.exceptions import BotoCoreError, ClientError

from chitrakatha.config import Settings
from chitrakatha.exceptions import DataIngestionError

logger = logging.getLogger(__name__)

# Supported file extensions and their S3 prefix destinations.
_PREFIX_MAP: Final[dict[str, str]] = {
    ".txt": "articles/",
    ".md": "articles/",
    ".vtt": "transcripts/",
    ".xlsx": "metadata/",
}


def _strip_vtt_timestamps(raw: str) -> str:
    """Remove WebVTT timestamp lines, keeping only spoken text.

    Args:
        raw: Raw VTT file content as a UTF-8 string.

    Returns:
        Plain text with timestamps and cue identifiers removed.
    """
    # Remove WEBVTT header, NOTE blocks, timestamps (00:00:00.000 --> ...) and
    # blank lines. Devanagari in cue text is preserved — regex targets only
    # the ASCII timestamp format.
    timestamp_re = re.compile(
        r"WEBVTT.*?\n|NOTE\s.*?\n\n|\d{2}:\d{2}[\d:.]+\s-->\s[\d:.]+[^\n]*\n",
        re.DOTALL,
    )
    cleaned = timestamp_re.sub("", raw)
    # Collapse multiple blank lines into one for readability.
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _flatten_xlsx(path: Path) -> str:
    """Flatten all Excel sheets into a single UTF-8 string.

    Args:
        path: Local filesystem path to the ``.xlsx`` file.

    Returns:
        All cell values joined by newlines, preserving Devanagari text.

    Raises:
        DataIngestionError: If the workbook cannot be opened or is empty.
    """
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise DataIngestionError(
            f"Cannot open Excel workbook {path}: {exc}"
        ) from exc

    lines: list[str] = []
    for sheet in wb.worksheets:
        for row in sheet.iter_rows(values_only=True):
            row_text = "\t".join(str(cell) for cell in row if cell is not None)
            if row_text.strip():
                lines.append(row_text)

    if not lines:
        raise DataIngestionError(f"Excel workbook {path} is empty — no data to ingest.")

    return "\n".join(lines)


def _validate_utf8(content: bytes, path: Path) -> str:
    """Decode bytes as strict UTF-8, preserving all Unicode including Devanagari.

    Args:
        content: Raw file bytes.
        path: Source path (used in error message only).

    Returns:
        Decoded UTF-8 string.

    Raises:
        DataIngestionError: If the file is not valid UTF-8.
    """
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataIngestionError(
            f"File {path} is not valid UTF-8 (byte position {exc.start}). "
            "Re-encode the source file to UTF-8 and retry."
        ) from exc


def _compute_md5(content: bytes) -> str:
    """Return the hex-encoded MD5 digest of ``content``."""
    return hashlib.md5(content, usedforsecurity=False).hexdigest()  # noqa: S324


def _parse_file(path: Path) -> bytes:
    """Read, validate, and normalise a source file to UTF-8 bytes.

    Args:
        path: Local path to the source file.

    Returns:
        Normalised UTF-8 bytes ready for S3 upload.

    Raises:
        DataIngestionError: On unsupported format, invalid encoding, or empty result.
    """
    suffix = path.suffix.lower()
    if suffix not in _PREFIX_MAP:
        raise DataIngestionError(
            f"Unsupported file extension '{suffix}'. "
            f"Supported: {list(_PREFIX_MAP.keys())}"
        )

    raw_bytes = path.read_bytes()

    if suffix == ".xlsx":
        text = _flatten_xlsx(path)
    else:
        text = _validate_utf8(raw_bytes, path)
        if suffix == ".vtt":
            text = _strip_vtt_timestamps(text)

    if not text.strip():
        raise DataIngestionError(
            f"File {path} produced empty output after parsing — skipping."
        )

    return text.encode("utf-8")


def upload_file(
    local_path: Path,
    settings: Settings,
    s3_client: boto3.client | None = None,
) -> str:
    """Parse and upload a single source file to S3 Bronze.

    Args:
        local_path: Path to the local source file.
        settings: Pydantic settings loaded from environment.
        s3_client: Optional pre-built boto3 S3 client (injected for testing).

    Returns:
        The full S3 URI of the uploaded object (``s3://bucket/key``).

    Raises:
        DataIngestionError: On parse failure or S3 upload error.
    """
    if not local_path.exists():
        raise DataIngestionError(f"File not found: {local_path}")

    suffix = local_path.suffix.lower()
    content_bytes = _parse_file(local_path)
    md5_hex = _compute_md5(content_bytes)

    s3_key = f"{_PREFIX_MAP[suffix]}{local_path.name}"

    client = s3_client or boto3.client("s3", region_name=settings.aws_region)

    try:
        client.put_object(
            Bucket=settings.s3_bronze_bucket,
            Key=s3_key,
            Body=content_bytes,
            ContentType="text/plain; charset=utf-8",
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId=settings.kms_key_arn,
            Metadata={
                "md5": md5_hex,
                "source-file": local_path.name,
                "project": "chitrakatha",
            },
        )
    except (BotoCoreError, ClientError) as exc:
        raise DataIngestionError(
            f"Failed to upload {local_path.name} to s3://{settings.s3_bronze_bucket}/{s3_key}: {exc}"
        ) from exc

    s3_uri = f"s3://{settings.s3_bronze_bucket}/{s3_key}"
    logger.info("Uploaded %s → %s (md5=%s)", local_path.name, s3_uri, md5_hex)
    return s3_uri


def main(paths: list[str]) -> None:
    """CLI entry point: upload one or more files to S3 Bronze.

    Args:
        paths: List of local file paths to upload (from sys.argv).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings()
    errors: list[str] = []

    for raw_path in paths:
        local_path = Path(raw_path)
        try:
            uri = upload_file(local_path, settings)
            print(f"✅  {local_path.name} → {uri}")
        except DataIngestionError as exc:
            logger.error("Skipping %s: %s", local_path, exc)
            errors.append(str(exc))

    if errors:
        print(f"\n❌  {len(errors)} file(s) failed. See logs above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_to_bronze.py <file1> [file2 ...]", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1:])
