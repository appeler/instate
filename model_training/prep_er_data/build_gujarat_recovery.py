# pyright: reportMissingImports=false
"""Build a deterministic Gujarati token corpus from the recovered 2017 roll."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterator


def _quoted(path: Path) -> str:
    return str(path).replace("'", "''")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_corpus(rows: Iterator[tuple[str, str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
    ):
        text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        writer = csv.writer(text, lineterminator="\n")
        writer.writerow(("gujarati", "english", "count"))
        writer.writerows(rows)
        text.flush()
    temporary.replace(path)


def build_gujarat_token_corpus(
    records_path: Path,
    parallel_csv_path: Path,
    corpus_path: Path,
    audit_path: Path,
    *,
    work_directory: Path | None = None,
) -> dict[str, object]:
    """Build compact weighted native/Latin pairs and measure recovered coverage.

    Args:
        records_path: Fully reconciled Gujarat elector records.
        parallel_csv_path: Restricted native/Latin companion CSV gzip.
        corpus_path: Destination deterministic weighted token corpus.
        audit_path: Destination JSON audit.
        work_directory: Optional parent for temporary DuckDB and Parquet files.

    Returns:
        The JSON-serializable audit payload.
    """
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    if not parallel_csv_path.is_file():
        raise FileNotFoundError(parallel_csv_path)
    required = {
        "source_filename",
        "epic_id",
        "elector_name_native",
    }
    missing = required.difference(pq.ParquetFile(records_path).schema_arrow.names)
    if missing:
        raise ValueError(f"recovered records are missing columns: {sorted(missing)}")

    with tempfile.TemporaryDirectory(dir=work_directory) as temporary_name:
        temporary = Path(temporary_name)
        reference_path = temporary / "reference.parquet"
        old_pairs_path = temporary / "old_pairs.parquet"
        combined_pairs_path = temporary / "combined_pairs.parquet"
        database = duckdb.connect()
        database.execute("SET threads=4")
        database.execute("SET memory_limit='4GB'")
        database.execute(f"SET temp_directory='{_quoted(temporary / 'duckdb')}'")
        parallel = _quoted(parallel_csv_path)
        database.execute(
            f"""
            COPY (
              SELECT filename || '.pdf' AS source_filename,
                     nullif(trim(id), '') AS epic_id,
                     regexp_extract(
                         lower(trim(elector_name_t13n)),
                         '^([^[:space:]]+)', 1
                     ) AS surname_latin
              FROM read_csv_auto(
                  '{parallel}', header=true, all_varchar=true, sample_size=100000
              )
              WHERE id IS NOT NULL AND trim(id) <> ''
                AND elector_name_t13n IS NOT NULL
                AND regexp_full_match(
                    regexp_extract(
                        lower(trim(elector_name_t13n)),
                        '^([^[:space:]]+)', 1
                    ),
                    '[a-z]+'
                )
            ) TO '{_quoted(reference_path)}'
              (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
            """  # noqa: S608 - paths are SQL-quoted local inputs
        )
        database.execute(
            f"""
            COPY (
              WITH names AS (
                SELECT regexp_split_to_array(
                           trim(elector_name), '\\s+'
                       ) AS native_tokens,
                       regexp_split_to_array(
                           lower(trim(elector_name_t13n)), '\\s+'
                       ) AS latin_tokens
                FROM read_csv_auto(
                    '{parallel}', header=true, all_varchar=true, sample_size=100000
                )
                WHERE elector_name IS NOT NULL AND trim(elector_name) <> ''
                  AND elector_name_t13n IS NOT NULL
                  AND trim(elector_name_t13n) <> ''
              ), aligned AS (
                SELECT native_tokens, latin_tokens
                FROM names
                WHERE array_length(native_tokens) = array_length(latin_tokens)
              ), pairs AS (
                SELECT unnest(native_tokens) AS native_token,
                       unnest(latin_tokens) AS latin_token
                FROM aligned
              )
              SELECT native_token, latin_token, count(*)::BIGINT AS occurrences
              FROM pairs
              WHERE native_token <> ''
                AND regexp_full_match(latin_token, '[a-z]+')
              GROUP BY native_token, latin_token
            ) TO '{_quoted(old_pairs_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """  # noqa: S608 - paths are SQL-quoted local inputs
        )
        records = _quoted(records_path)
        database.execute(
            f"""
            COPY (
              WITH recovered AS (
                SELECT source_filename, epic_id,
                       regexp_extract(
                           trim(elector_name_native),
                           '^([^[:space:]]+)', 1
                       ) AS native_token
                FROM read_parquet('{records}')
                WHERE epic_id IS NOT NULL AND epic_id <> ''
                  AND regexp_full_match(
                      trim(elector_name_native),
                      '[^[:space:]]+[[:space:]]+.*'
                  )
              ), learned AS (
                SELECT recovered.native_token,
                       reference.surname_latin AS latin_token,
                       count(*)::BIGINT AS occurrences
                FROM recovered
                JOIN read_parquet('{_quoted(reference_path)}') AS reference
                  USING (source_filename, epic_id)
                GROUP BY recovered.native_token, reference.surname_latin
              ), combined AS (
                SELECT * FROM read_parquet('{_quoted(old_pairs_path)}')
                UNION ALL
                SELECT * FROM learned
              )
              SELECT native_token, latin_token,
                     sum(occurrences)::BIGINT AS occurrences
              FROM combined
              GROUP BY native_token, latin_token
              ORDER BY native_token, latin_token
            ) TO '{_quoted(combined_pairs_path)}'
              (FORMAT PARQUET, COMPRESSION ZSTD)
            """  # noqa: S608 - paths are SQL-quoted local inputs
        )

        pair_rows = database.execute(
            f"""
            SELECT native_token, latin_token, occurrences
            FROM read_parquet('{_quoted(combined_pairs_path)}')
            ORDER BY native_token, latin_token
            """  # noqa: S608 - local temporary path
        ).fetchall()
        _write_corpus(iter(pair_rows), corpus_path)

        pair_summary = database.execute(
            f"""
            WITH ranked AS (
              SELECT *,
                     row_number() OVER (
                         PARTITION BY native_token
                         ORDER BY occurrences DESC, latin_token
                     ) AS rank,
                     lead(occurrences) OVER (
                         PARTITION BY native_token
                         ORDER BY occurrences DESC, latin_token
                     ) AS second_occurrences,
                     count(*) OVER (PARTITION BY native_token) AS variants
              FROM read_parquet('{_quoted(combined_pairs_path)}')
            )
            SELECT count(*) AS native_types,
                   count(*) FILTER (WHERE variants > 1) AS contested_types,
                   count(*) FILTER (
                       WHERE occurrences = coalesce(second_occurrences, -1)
                   ) AS tied_types,
                   sum(occurrences) AS source_occurrences
            FROM ranked
            WHERE rank = 1
            """  # noqa: S608 - local temporary path
        ).fetchone()
        coverage = database.execute(
            f"""
            WITH ranked AS (
              SELECT *,
                     row_number() OVER (
                         PARTITION BY native_token
                         ORDER BY occurrences DESC, latin_token
                     ) AS rank,
                     lead(occurrences) OVER (
                         PARTITION BY native_token
                         ORDER BY occurrences DESC, latin_token
                     ) AS second_occurrences
              FROM read_parquet('{_quoted(combined_pairs_path)}')
            ), accepted AS (
              SELECT native_token
              FROM ranked
              WHERE rank = 1
                AND occurrences > coalesce(second_occurrences, -1)
            ), recovered AS (
              SELECT source_filename, epic_id,
                     regexp_extract(
                         trim(elector_name_native),
                         '^([^[:space:]]+)', 1
                     ) AS native_token
              FROM read_parquet('{records}')
              WHERE regexp_full_match(
                  trim(elector_name_native),
                  '[^[:space:]]+[[:space:]]+.*'
              )
            )
            SELECT count(*) AS eligible_rows,
                   count(accepted.native_token) AS mapped_rows,
                   count(reference.surname_latin) AS exact_reference_rows
            FROM recovered
            LEFT JOIN accepted USING (native_token)
            LEFT JOIN read_parquet('{_quoted(reference_path)}') AS reference
              USING (source_filename, epic_id)
            """  # noqa: S608 - paths are SQL-quoted local inputs
        ).fetchone()
        parallel_rows = database.execute(
            f"SELECT count(*) FROM read_parquet('{_quoted(reference_path)}')"  # noqa: S608 - local temporary path
        ).fetchone()[0]

    audit: dict[str, object] = {
        "source": {
            "records_path": records_path.name,
            "records_sha256": _sha256(records_path),
            "records_rows": pq.ParquetFile(records_path).metadata.num_rows,
            "parallel_path": parallel_csv_path.name,
            "parallel_sha256": _sha256(parallel_csv_path),
            "parallel_rows_with_epic_and_latin_surname": parallel_rows,
        },
        "corpus": {
            "path": corpus_path.name,
            "sha256": _sha256(corpus_path),
            "candidate_pairs": len(pair_rows),
            "native_types": pair_summary[0],
            "contested_types": pair_summary[1],
            "tied_types": pair_summary[2],
            "source_token_occurrences": pair_summary[3],
        },
        "recovered_coverage": {
            "eligible_rows": coverage[0],
            "mapped_rows": coverage[1],
            "mapped_fraction": coverage[1] / coverage[0] if coverage[0] else None,
            "exact_reference_rows": coverage[2],
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    """Build the weighted Gujarati token corpus and its audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--parallel-csv", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path)
    args = parser.parse_args()
    build_gujarat_token_corpus(
        args.records,
        args.parallel_csv,
        args.corpus,
        args.audit,
        work_directory=args.work_directory,
    )


if __name__ == "__main__":
    main()
