"""Unit tests for the Bronze → Silver preprocessing step.

Tests cover:
    - Language detection: English, Hindi (Devanagari), mixed bilingual.
    - Raw file parsing: .txt passthrough, .vtt timestamp stripping, empty file rejection.
    - SHA-256 deduplication: identical content only appears once in output.
    - Full process(): corpus and training JSONL produced correctly.
    - Error propagation: unreadable files increment error count but don't abort.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

# Import the functions under test from the processing step module.
from pipeline.steps.preprocessing import (
    _detect_language,
    _parse_raw,
    _sha256,
    process,
)
from chitrakatha.exceptions import DataIngestionError


# ---------------------------------------------------------------------------
# _detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_english_only(self) -> None:
        assert _detect_language("Nagraj is a serpent superhero from Raj Comics.") == "en"

    def test_devanagari_only(self) -> None:
        assert _detect_language("नागराज राज कॉमिक्स का सुपरहीरो है।") == "hi"

    def test_bilingual_mixed(self) -> None:
        assert _detect_language("Nagraj नागराज is a superhero.") == "en-hi"

    def test_empty_string_defaults_to_en(self) -> None:
        # No Devanagari → falls through to "en".
        assert _detect_language("") == "en"

    def test_numbers_and_punctuation_only(self) -> None:
        assert _detect_language("1234 !@#$") == "en"


# ---------------------------------------------------------------------------
# _sha256
# ---------------------------------------------------------------------------

class TestSha256:
    def test_deterministic(self) -> None:
        text = "Doga is an anti-hero from Raj Comics."
        assert _sha256(text) == _sha256(text)

    def test_different_inputs_differ(self) -> None:
        assert _sha256("abc") != _sha256("xyz")

    def test_devanagari_text(self) -> None:
        text = "नागराज"
        digest = _sha256(text)
        assert len(digest) == 64  # hex SHA-256 is always 64 chars


# ---------------------------------------------------------------------------
# _parse_raw
# ---------------------------------------------------------------------------

class TestParseRaw:
    def test_plain_txt_passthrough(self, tmp_path: Path) -> None:
        content = "Nagraj is a serpent superhero."
        f = tmp_path / "article.txt"
        f.write_text(content, encoding="utf-8")
        result = _parse_raw(f)
        assert result == content

    def test_vtt_timestamps_stripped(self, tmp_path: Path) -> None:
        vtt_content = textwrap.dedent("""\
            WEBVTT

            00:00:01.000 --> 00:00:04.000
            Nagraj first appeared in 1986.

            00:00:05.000 --> 00:00:08.000
            He was created by Sanjay Gupta.
        """)
        f = tmp_path / "transcript.vtt"
        f.write_text(vtt_content, encoding="utf-8")
        result = _parse_raw(f)
        assert "00:00:01" not in result
        assert "Nagraj first appeared in 1986." in result
        assert "He was created by Sanjay Gupta." in result

    def test_devanagari_preserved_in_txt(self, tmp_path: Path) -> None:
        content = "नागराज एक सर्प सुपरहीरो है।"
        f = tmp_path / "hindi.txt"
        f.write_text(content, encoding="utf-8")
        result = _parse_raw(f)
        assert "नागराज" in result

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        with pytest.raises(DataIngestionError, match="empty after parsing"):
            _parse_raw(f)

    def test_whitespace_only_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "whitespace.txt"
        f.write_text("   \n\t\n   ", encoding="utf-8")
        with pytest.raises(DataIngestionError, match="empty after parsing"):
            _parse_raw(f)

    def test_md_file_treated_as_text(self, tmp_path: Path) -> None:
        content = "# Raj Comics\nNagraj is the flagship character."
        f = tmp_path / "article.md"
        f.write_text(content, encoding="utf-8")
        result = _parse_raw(f)
        assert "Nagraj" in result


# ---------------------------------------------------------------------------
# process() — integration of all sub-steps
# ---------------------------------------------------------------------------

class TestProcess:
    def _write_file(self, dir_: Path, name: str, content: str) -> Path:
        p = dir_ / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_single_file_produces_corpus_and_training_jsonl(self, tmp_path: Path) -> None:
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        corpus_out = tmp_path / "corpus"
        training_out = tmp_path / "training"
        self._write_file(bronze, "article.txt", "Nagraj " * 100)

        errors = process(bronze, corpus_out, training_out)

        assert errors == 0
        corpus_lines = (corpus_out / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        training_lines = (training_out / "training.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(corpus_lines) == 1
        assert len(training_lines) >= 1  # at least one chunk

    def test_corpus_record_schema(self, tmp_path: Path) -> None:
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        corpus_out = tmp_path / "corpus"
        training_out = tmp_path / "training"
        self._write_file(bronze, "doc.txt", "Chacha Chaudhary is a Diamond Comics character.")

        process(bronze, corpus_out, training_out)

        record = json.loads((corpus_out / "corpus.jsonl").read_text(encoding="utf-8"))
        assert "text" in record
        assert "source_document" in record
        assert "language" in record
        assert "content_hash" in record
        assert record["source_document"] == "doc.txt"
        assert record["language"] == "en"

    def test_training_chunk_schema(self, tmp_path: Path) -> None:
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        corpus_out = tmp_path / "corpus"
        training_out = tmp_path / "training"
        self._write_file(bronze, "doc.txt", "Suppandi " * 200)

        process(bronze, corpus_out, training_out)

        first_line = (training_out / "training.jsonl").read_text(encoding="utf-8").splitlines()[0]
        chunk = json.loads(first_line)
        for key in ("text", "chunk_id", "chunk_index", "token_count", "source_document", "language", "content_hash"):
            assert key in chunk, f"Missing key '{key}' in training chunk"

    def test_duplicate_files_deduplicated(self, tmp_path: Path) -> None:
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        corpus_out = tmp_path / "corpus"
        training_out = tmp_path / "training"
        identical_content = "Phantom from Indrajal Comics." * 50
        self._write_file(bronze, "a.txt", identical_content)
        self._write_file(bronze, "b.txt", identical_content)  # exact duplicate

        errors = process(bronze, corpus_out, training_out)

        assert errors == 0
        corpus_lines = (corpus_out / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(corpus_lines) == 1  # second file was deduped

    def test_empty_file_counted_as_error(self, tmp_path: Path) -> None:
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        corpus_out = tmp_path / "corpus"
        training_out = tmp_path / "training"
        self._write_file(bronze, "empty.txt", "")
        self._write_file(bronze, "good.txt", "Tinkle magazine characters include Suppandi." * 10)

        errors = process(bronze, corpus_out, training_out)

        assert errors == 1  # one bad file
        corpus_lines = (corpus_out / "corpus.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(corpus_lines) == 1  # good file still processed

    def test_devanagari_document_language_tagged_hi(self, tmp_path: Path) -> None:
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        corpus_out = tmp_path / "corpus"
        training_out = tmp_path / "training"
        self._write_file(bronze, "hindi.txt", "नागराज एक सर्प सुपरहीरो है। " * 50)

        process(bronze, corpus_out, training_out)

        record = json.loads((corpus_out / "corpus.jsonl").read_text(encoding="utf-8"))
        assert record["language"] == "hi"

    def test_unsupported_file_type_ignored(self, tmp_path: Path) -> None:
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        corpus_out = tmp_path / "corpus"
        training_out = tmp_path / "training"
        (bronze / "image.png").write_bytes(b"\x89PNG\r\n")  # binary, unsupported
        self._write_file(bronze, "article.txt", "Billu is a Diamond Comics character." * 20)

        errors = process(bronze, corpus_out, training_out)

        assert errors == 0  # PNG ignored, not counted as an error
        corpus_lines = (corpus_out / "corpus.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(corpus_lines) == 1

    def test_empty_bronze_directory_produces_empty_outputs(self, tmp_path: Path) -> None:
        bronze = tmp_path / "bronze"
        bronze.mkdir()
        corpus_out = tmp_path / "corpus"
        training_out = tmp_path / "training"

        errors = process(bronze, corpus_out, training_out)

        assert errors == 0
        assert (corpus_out / "corpus.jsonl").read_text(encoding="utf-8") == ""
        assert (training_out / "training.jsonl").read_text(encoding="utf-8") == ""
