"""Prepare, checkpoint and decode cropped Urdu-roll OCR without changing identities."""

import argparse
import fcntl
import hashlib
import itertools
import json
import math
import sqlite3
import struct
import sys
import zlib
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf

from model_training.prep_er_data.jk_urdu_unicode import native_text

MODEL = "muse-spark-1.3-contributor"
REVISION = "jk-paired-ocr-1"
RESERVE_NANO = 110_000_000
PROMPT = (
    "Read both Urdu names in every labeled crop. A crop ID is supplied immediately "
    "before its image. Return a JSON object keyed by every exact crop ID. Each "
    "value is a five-element array: [voter_name, relative_name, relationship, "
    "voter_issue, relative_issue]. Read the top voter-name field and then the "
    "relative-name field. Omit printed labels and colons. Copy the visible source "
    "letters and spacing exactly; do not correct, infer or complete names. Read "
    "relationship only from its printed label: father, mother, husband, or null. "
    "Ignore ages, house numbers, sex, and lower-edge control text. Each issue is "
    "null for a clearly readable field. For a blank field return a null name and "
    "issue blank. For clipped or unreadable text return a null name and a short "
    "issue. For a doubtful visible reading preserve it and give a short issue. "
    "No transliteration, commentary or extra keys."
)
DDL = """
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources(
 id INTEGER PRIMARY KEY, filename TEXT UNIQUE NOT NULL, sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cards(
 id INTEGER PRIMARY KEY, entry_key TEXT UNIQUE NOT NULL,
 source_id INTEGER NOT NULL REFERENCES sources(id), page INTEGER NOT NULL,
 bbox TEXT NOT NULL, missing INTEGER NOT NULL CHECK(missing IN (1,2,3)),
 crop_key TEXT REFERENCES crops(key));
CREATE TABLE IF NOT EXISTS crops(
 key TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'new', reading TEXT);
CREATE TABLE IF NOT EXISTS calls(
 id INTEGER PRIMARY KEY, status TEXT NOT NULL, reserve_nano INTEGER NOT NULL,
 cost_nano INTEGER, response BLOB, error TEXT, elapsed_seconds REAL);
CREATE TABLE IF NOT EXISTS request_items(
 call_id INTEGER NOT NULL REFERENCES calls(id), local_id TEXT NOT NULL,
 crop_key TEXT UNIQUE NOT NULL REFERENCES crops(key), PRIMARY KEY(call_id,local_id));
CREATE TABLE IF NOT EXISTS failed_attempts(
 call_id INTEGER NOT NULL REFERENCES calls(id), attempt INTEGER NOT NULL,
 status TEXT NOT NULL,
 reserve_nano INTEGER NOT NULL, cost_nano INTEGER, response BLOB,
 error TEXT, elapsed_seconds REAL, review_reason TEXT NOT NULL,
 PRIMARY KEY(call_id,attempt));
CREATE INDEX IF NOT EXISTS cards_crop ON cards(crop_key);
CREATE INDEX IF NOT EXISTS crops_status ON crops(status);
"""


def digest(path):
    """Hash source bytes without loading a complete inventory into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def crop_box(box):
    """Keep both name lines, excluding the voter-ID header and portrait column."""
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError("Invalid card rectangle")
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    if not 175 <= width <= 185 or not 61 <= height <= 84:
        raise ValueError("Unvalidated card geometry")
    bottom = y0 + 56 if height <= 71 else y1 - 10
    return x0 + 1, y0 + 17, x1 - 46, bottom


def crop_digest(width, height, pixels):
    """Identify identical pixels under the same reading and crop policy."""
    header = REVISION.encode() + struct.pack(">II", width, height)
    return hashlib.sha256(header + pixels).hexdigest()


def unique_object(pairs):
    """Reject duplicate response keys before a dictionary can hide them."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate response ID")
        result[key] = value
    return result


def parse_response(raw, expected_ids):
    """Retain raw readings and flag invalid script without inventing replacements."""
    if raw.get("status") != "completed":
        raise ValueError("Response did not complete")
    text = "".join(
        part["text"]
        for message in raw.get("output", [])
        if message.get("type") == "message"
        for part in message.get("content", [])
        if part.get("type") == "output_text"
    )
    answers = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(answers, dict) or set(answers) != set(expected_ids):
        raise ValueError("Response IDs do not match the submitted crops")
    results = {}
    for key, row in answers.items():
        if not isinstance(row, list) or len(row) != 5:
            raise ValueError("Expected five response fields")
        own, relative, relationship, own_issue, relative_issue = row
        if relationship not in ("father", "mother", "husband", None):
            raise ValueError("Invalid printed relationship")
        result = {"relationship": relationship}
        for role, name, issue in [
            ("own", own, own_issue),
            ("relative", relative, relative_issue),
        ]:
            if name is not None and (not isinstance(name, str) or not name.strip()):
                raise ValueError("Invalid name field")
            if issue is not None and (not isinstance(issue, str) or not issue.strip()):
                raise ValueError("Invalid uncertainty field")
            normalized = native_text(name) if name is not None else None
            if issue is None and normalized is None:
                issue = "missing_or_invalid_script"
            result[role] = {
                "raw": name,
                "issue": issue,
                "accepted": normalized if issue is None else None,
            }
        results[key] = result
    return results


def usage_cost(raw):
    """Count provider-reported usage in integer nanodollars, including reasoning."""
    usage = raw.get("usage")
    if not usage:
        return None
    incoming = usage.get("input_tokens")
    outgoing = usage.get("output_tokens")
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    if any(type(v) is not int or v < 0 for v in (incoming, outgoing, cached)):
        return None
    if cached > incoming:
        return None
    return (incoming - cached) * 100 + cached * 2 + outgoing * 200


@contextmanager
def journal(path):
    """Permit one process to mutate the checkpoint, including its spending state."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(DDL)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(failed_attempts)")}
        if "attempt" not in columns:
            with conn:
                conn.execute(
                    "ALTER TABLE failed_attempts RENAME TO failed_attempts_legacy"
                )
                conn.execute(
                    "CREATE TABLE failed_attempts("
                    "call_id INTEGER NOT NULL REFERENCES calls(id),"
                    "attempt INTEGER NOT NULL,status TEXT NOT NULL,"
                    "reserve_nano INTEGER NOT NULL,cost_nano INTEGER,response BLOB,"
                    "error TEXT,elapsed_seconds REAL,review_reason TEXT NOT NULL,"
                    "PRIMARY KEY(call_id,attempt))"
                )
                conn.execute(
                    "INSERT INTO failed_attempts "
                    "SELECT call_id,1,status,reserve_nano,cost_nano,response,error,"
                    "elapsed_seconds,review_reason FROM failed_attempts_legacy"
                )
                conn.execute("DROP TABLE failed_attempts_legacy")
        try:
            yield conn
        finally:
            conn.close()


def metadata(conn, key, value=None):
    """Read metadata or atomically replace one JSON value."""
    if value is not None:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?,?)",
                (key, json.dumps(value, sort_keys=True)),
            )
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def prepare(conn, workload, manifest_path, source_dir):
    """Stream all missing fields into a resumable card queue; never drop a role."""
    manifest = json.loads(Path(manifest_path).read_text())
    if digest(workload) != manifest["workload_sha256"]:
        raise ValueError("Workload hash differs from the validated manifest")
    config = {
        "workload_sha256": manifest["workload_sha256"],
        "manifest_sha256": digest(manifest_path),
        "model": MODEL,
        "revision": REVISION,
        "source_dir": str(Path(source_dir).resolve()),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
    }
    prior = metadata(conn, "config")
    if prior is not None and prior != config:
        raise ValueError("Checkpoint belongs to a different workload or crop policy")
    metadata(conn, "config", config)
    sources = {
        r["filename"]: (r["id"], r["sha256"])
        for r in conn.execute("SELECT * FROM sources")
    }
    fields = cards = 0
    input_digest = hashlib.sha256()
    buffered = []

    def all_rows():
        for batch in pq.ParquetFile(workload).iter_batches(batch_size=8192):
            yield from batch.to_pylist()

    for key, group in itertools.groupby(all_rows(), key=lambda r: r["entry_event_key"]):
        rows = list(group)
        first = rows[0]
        roles = [r["role"] for r in rows]
        if len(set(roles)) != len(roles) or not set(roles) <= {"own", "relative"}:
            raise ValueError("Duplicate or invalid missing field")
        for row in rows:
            if any(
                row[k] != first[k]
                for k in ("filename", "source_sha256", "page", "bbox")
            ):
                raise ValueError("Fields disagree on their source card")
            expected = hashlib.sha256((key + ":" + row["role"]).encode()).hexdigest()
            if row["task_id"] != expected:
                raise ValueError("Missing-field task ID is inconsistent")
        crop_box(first["bbox"])
        name, source_sha = first["filename"], first["source_sha256"]
        if Path(name).name != name or first["page"] < 1:
            raise ValueError("Invalid PDF identity")
        if name not in sources:
            cursor = conn.execute(
                "INSERT INTO sources(filename,sha256) VALUES (?,?)", (name, source_sha)
            )
            sources[name] = (cursor.lastrowid, source_sha)
        source_id, expected_sha = sources[name]
        if source_sha != expected_sha:
            raise ValueError("PDF hash changes within the workload")
        mask = (1 if "own" in roles else 0) | (2 if "relative" in roles else 0)
        buffered.append(
            (key, source_id, first["page"], json.dumps(first["bbox"]), mask)
        )
        input_digest.update(
            (json.dumps(buffered[-1], separators=(",", ":")) + "\n").encode()
        )
        fields += len(rows)
        cards += 1
        if len(buffered) == 8192:
            with conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO cards"
                    "(entry_key,source_id,page,bbox,missing) "
                    "VALUES (?,?,?,?,?)",
                    buffered,
                )
            buffered.clear()
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO cards"
            "(entry_key,source_id,page,bbox,missing) "
            "VALUES (?,?,?,?,?)",
            buffered,
        )
    saved = conn.execute(
        "SELECT count(*),sum((missing&1)+((missing&2)/2)) FROM cards"
    ).fetchone()
    if tuple(saved) != (cards, fields) or fields != manifest["rows"]:
        raise ValueError("Checkpoint does not account for every missing source field")
    disk_digest = hashlib.sha256()
    for row in conn.execute(
        "SELECT entry_key,source_id,page,bbox,missing FROM cards ORDER BY id"
    ):
        disk_digest.update(
            (json.dumps(tuple(row), separators=(",", ":")) + "\n").encode()
        )
    if input_digest.digest() != disk_digest.digest():
        raise ValueError("Checkpoint fields differ from the source workload")
    metadata(
        conn,
        "prepared",
        {
            "cards": cards,
            "missing_fields": fields,
            "card_fields_sha256": disk_digest.hexdigest(),
        },
    )
    return {"cards": cards, "missing_fields": fields}


class Renderer:
    """Read each PDF under its verified hash and render complete paired-name crops."""

    def __init__(self, source_dir):
        """Bind the renderer to an immutable source-PDF directory."""
        self.root = Path(source_dir)
        self.document = None
        self.source = None

    def close(self):
        """Close the current PDF, if any, and clear its identity."""
        if self.document is not None:
            self.document.close()
            self.document = None
            self.source = None

    def render(self, row):
        """Verify and render one complete paired-name crop."""
        identity = (row["filename"], row["sha256"])
        if identity != self.source:
            self.close()
            path = self.root / row["filename"]
            if digest(path) != row["sha256"]:
                raise ValueError("PDF bytes no longer match the source inventory")
            self.document = pymupdf.open(path)
            self.source = identity
        page = self.document[row["page"] - 1]
        clip = pymupdf.Rect(crop_box(json.loads(row["bbox"])))
        if not page.rect.contains(clip):
            raise ValueError("Name crop would be clipped by the physical page")
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(4, 4),
            clip=clip,
            colorspace=pymupdf.csGRAY,
            alpha=False,
        )
        return crop_digest(pix.width, pix.height, pix.samples), pix.tobytes("png")


def render_next(conn, renderer, count):
    """Index exact pixels locally, preserving every card linked to reused crops."""
    processed = 0
    while processed < count:
        rows = conn.execute(
            "SELECT c.*,s.filename,s.sha256 FROM cards c "
            "JOIN sources s ON c.source_id=s.id WHERE crop_key IS NULL "
            "ORDER BY c.id LIMIT ?",
            (min(count - processed, 128),),
        ).fetchall()
        if not rows:
            break
        with conn:
            for row in rows:
                key, _ = renderer.render(row)
                conn.execute("INSERT OR IGNORE INTO crops(key) VALUES (?)", (key,))
                conn.execute("UPDATE cards SET crop_key=? WHERE id=?", (key, row["id"]))
        processed += len(rows)
    return processed


def next_images(conn, renderer, size=26):
    """Reuse exact crop results and assemble at most 26 separate image inputs."""
    while True:
        pending = conn.execute(
            "SELECT key FROM crops WHERE status='new' LIMIT ?", (size,)
        ).fetchall()
        if len(pending) == size or not render_next(conn, renderer, size - len(pending)):
            break
    images = {}
    for number, crop in enumerate(pending):
        row = conn.execute(
            """SELECT c.*,s.filename,s.sha256 FROM cards c
            JOIN sources s ON c.source_id=s.id WHERE crop_key=? LIMIT 1""",
            (crop["key"],),
        ).fetchone()
        key, png = renderer.render(row)
        if key != crop["key"]:
            raise ValueError("Resumed crop pixels differ from the checkpoint")
        images[f"C{number:03d}"] = (key, png)
    return images


def exposure(conn):
    """Include unresolved attempts at their reserved maximum cost."""
    return conn.execute(
        "SELECT coalesce(sum(coalesce(cost_nano,reserve_nano)),0) FROM "
        "(SELECT cost_nano,reserve_nano FROM calls UNION ALL "
        "SELECT cost_nano,reserve_nano FROM failed_attempts)"
    ).fetchone()[0]


def selected_images(conn, renderer, entries, size=26):
    """Yield only named cards, reusing completed crops and preserving source checks."""
    if not entries or any(not isinstance(key, str) or not key for key in entries):
        raise ValueError("Selection must contain nonempty source record keys")
    if len(set(entries)) != len(entries):
        raise ValueError("Selection contains duplicate source record keys")
    rows = []
    for entry in entries:
        row = conn.execute(
            "SELECT c.*,s.filename,s.sha256 FROM cards c "
            "JOIN sources s ON c.source_id=s.id WHERE c.entry_key=?",
            (entry,),
        ).fetchone()
        if row is None:
            raise ValueError("Selected record is outside the prepared workload")
        rows.append(row)
    images = {}
    batch_keys = set()
    for row in rows:
        if row["crop_key"] is not None:
            saved = conn.execute(
                "SELECT status FROM crops WHERE key=?", (row["crop_key"],)
            ).fetchone()
            if saved is not None and saved[0] in (
                "abandoned",
                "complete",
                "submitted",
            ):
                continue
        key, png = renderer.render(row)
        if row["crop_key"] is not None and key != row["crop_key"]:
            raise ValueError("Selected source pixels differ from the checkpoint")
        with conn:
            conn.execute("INSERT OR IGNORE INTO crops(key) VALUES (?)", (key,))
            conn.execute("UPDATE cards SET crop_key=? WHERE id=?", (key, row["id"]))
        status = conn.execute(
            "SELECT status FROM crops WHERE key=?", (key,)
        ).fetchone()[0]
        if status in ("complete", "submitted") or key in batch_keys:
            continue
        if status != "new":
            raise ValueError("Selected crop has an unresolved prior result")
        images[f"C{len(images):03d}"] = (key, png)
        batch_keys.add(key)
        if len(images) == size:
            yield images
            images = {}
            batch_keys = set()
    if images:
        yield images


def reserve(conn, images, budget_nano):
    """Commit the spending reservation and exact ID binding before any paid call."""
    if not images or exposure(conn) + RESERVE_NANO > budget_nano:
        raise ValueError("No images or insufficient authorized budget")
    with conn:
        call = conn.execute(
            "INSERT INTO calls(status,reserve_nano) VALUES ('reserved',?)",
            (RESERVE_NANO,),
        ).lastrowid
        for local_id, (key, _) in images.items():
            changed = conn.execute(
                "UPDATE crops SET status='submitted' WHERE key=? AND status='new'",
                (key,),
            ).rowcount
            if changed != 1:
                raise ValueError("Crop was already submitted")
            conn.execute(
                "INSERT INTO request_items VALUES (?,?,?)", (call, local_id, key)
            )
    return call


def receive(conn, call, raw, elapsed):
    """Persist a provider receipt before parsing, so a restart needs no new call."""
    cost = usage_cost(raw)
    with conn:
        conn.execute(
            "UPDATE calls SET status='received',response=?,cost_nano=?,"
            "elapsed_seconds=? WHERE id=?",
            (zlib.compress(json.dumps(raw).encode()), cost, elapsed, call),
        )
    if cost is not None and cost > RESERVE_NANO:
        raise ValueError("Provider usage exceeded the pre-submission reservation")


def decode_received(conn):
    """Replay stored receipts without resubmission; quarantine malformed outputs."""
    for call in conn.execute("SELECT * FROM calls WHERE status='received'").fetchall():
        items = conn.execute(
            "SELECT local_id,crop_key FROM request_items WHERE call_id=?", (call["id"],)
        ).fetchall()
        try:
            raw = json.loads(zlib.decompress(call["response"]))
            answers = parse_response(raw, [r["local_id"] for r in items])
        except (ValueError, KeyError, TypeError) as error:
            with conn:
                conn.execute(
                    "UPDATE calls SET status='invalid',error=? WHERE id=?",
                    (str(error), call["id"]),
                )
                conn.executemany(
                    "UPDATE crops SET status='invalid' WHERE key=?",
                    [(r["crop_key"],) for r in items],
                )
            continue
        with conn:
            for item in items:
                conn.execute(
                    "UPDATE crops SET status='complete',reading=? WHERE key=?",
                    (
                        json.dumps(answers[item["local_id"]], ensure_ascii=False),
                        item["crop_key"],
                    ),
                )
            conn.execute("UPDATE calls SET status='complete' WHERE id=?", (call["id"],))


def checkpoint_summary(conn):
    """Report separate counts for records, exact-pixel reuse and paid results."""
    return {
        "prepared": metadata(conn, "prepared"),
        "indexed_cards": conn.execute(
            "SELECT count(*) FROM cards WHERE crop_key IS NOT NULL"
        ).fetchone()[0],
        "unique_crops": conn.execute("SELECT count(*) FROM crops").fetchone()[0],
        "crop_states": dict(
            conn.execute("SELECT status,count(*) FROM crops GROUP BY status")
        ),
        "call_states": dict(
            conn.execute("SELECT status,count(*) FROM calls GROUP BY status")
        ),
        "archived_failed_attempts": conn.execute(
            "SELECT count(*) FROM failed_attempts"
        ).fetchone()[0],
        "total_request_attempts": conn.execute(
            "SELECT (SELECT count(*) FROM calls WHERE status<>'abandoned') + "
            "(SELECT count(*) FROM failed_attempts)"
        ).fetchone()[0],
        "accounted_job_exposure_usd": exposure(conn) / 1e9,
    }


def export_readings(conn, output):
    """Write one OCR result per originally missing field, including abstentions."""
    pending = conn.execute(
        "SELECT count(*) FROM cards c LEFT JOIN crops p ON c.crop_key=p.key "
        "WHERE p.status IS NULL OR p.status<>'complete'"
    ).fetchone()[0]
    if pending or not metadata(conn, "prepared"):
        raise ValueError("Every queued crop must finish before final export")
    schema = pa.schema(
        [
            (name, pa.string())
            for name in [
                "entry_event_key",
                "filename",
                "source_sha256",
                "role",
                "ocr_name_raw",
                "ocr_name",
                "ocr_issue",
                "relationship",
                "crop_key",
                "model",
                "revision",
            ]
        ]
    )
    output = Path(output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    emitted = 0
    buffer = []
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for row in conn.execute(
            "SELECT c.entry_key,c.missing,c.crop_key,s.filename,s.sha256,p.reading "
            "FROM cards c JOIN sources s ON c.source_id=s.id "
            "JOIN crops p ON c.crop_key=p.key ORDER BY c.id"
        ):
            reading = json.loads(row["reading"])
            for role, bit in [("own", 1), ("relative", 2)]:
                if not row["missing"] & bit:
                    continue
                field = reading[role]
                buffer.append(
                    {
                        "entry_event_key": row["entry_key"],
                        "filename": row["filename"],
                        "source_sha256": row["sha256"],
                        "role": role,
                        "ocr_name_raw": field["raw"],
                        "ocr_name": field["accepted"],
                        "ocr_issue": field["issue"],
                        "relationship": reading["relationship"],
                        "crop_key": row["crop_key"],
                        "model": MODEL,
                        "revision": REVISION,
                    }
                )
                emitted += 1
                if len(buffer) == 8192:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    if emitted != metadata(conn, "prepared")["missing_fields"]:
        raise ValueError("Export changed the number of missing source fields")
    if pq.ParquetFile(temporary).metadata.num_rows != emitted:
        raise ValueError("On-disk OCR output is incomplete")
    temporary.replace(output)
    return {"rows": emitted, "sha256": digest(output)}


def main():
    """Prepare and inspect an OCR checkpoint without sending paid requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "index", "status", "export"])
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-cards", type=int, default=1000)
    args = parser.parse_args()
    with journal(args.journal) as conn:
        if args.command == "prepare":
            if not all((args.workload, args.manifest, args.source_dir)):
                parser.error("prepare requires workload, manifest and source-dir")
            prepare(conn, args.workload, args.manifest, args.source_dir)
        elif args.command == "index":
            if not metadata(conn, "prepared") or args.max_cards < 1:
                parser.error("index requires a prepared journal and positive max-cards")
            renderer = Renderer(metadata(conn, "config")["source_dir"])
            try:
                render_next(conn, renderer, args.max_cards)
            finally:
                renderer.close()
        elif args.command == "export":
            if not args.output:
                parser.error("export requires output")
            export_readings(conn, args.output)
        sys.stdout.write(json.dumps(checkpoint_summary(conn)) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
