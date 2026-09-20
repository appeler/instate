"""Review and promote transliterations for selected J&K Urdu surname tokens."""

from __future__ import annotations

import argparse
import csv
import difflib
import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import unicodedata
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from model_training.prep_er_data.jk_urdu_ocr import MODEL

RUBRIC = (
    "Transliterate each standalone Urdu-script token into a plausible conventional "
    "ASCII Latin name spelling. Preserve its written letters; do not translate "
    "nouns, infer a person's identity or hereditary surname, or silently correct "
    "unusual source spellings into familiar names. Urdu vowels and Latin conventions "
    "may be ambiguous: provide at most three plausible alternatives, or use uncertain "
    "when no defensible reading is available. A source_warning is required for unusual "
    "or apparently damaged native spellings; never repair them silently. "
    "Non-Arabic-script strings, replacement characters and digits-only input are "
    'unsupported. Return only JSON {"items":[{"id":"...","latin":"..." '
    'or null,"alternatives":["..."],"status":"supported" or "uncertain" '
    'or "unsupported","source_warning":false or true}]}. Return exactly one '
    "result per ID, with no extra fields. All nonnull Latin forms must be single ASCII "
    "letter tokens of at least two letters. No explanations, tools or web calls."
)
PASSES = ("a", "b", "c")
INPUT_RATE = Decimal("0.1")
OUTPUT_RATE = Decimal("0.2")
CACHED_RATE = Decimal("0.002")


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically write indented JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_references(reference_dir: Path) -> dict[str, set[str]]:
    """Load verified Dakshina Urdu forms after checking its local manifest."""
    manifest_path = reference_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    references: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        path = reference_dir / f"ur.translit.sampled.{split}.tsv"
        if sha256(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError(f"Reference hash mismatch: {path}")
        with path.open() as stream:
            for native, latin, _count in csv.reader(stream, delimiter="\t"):
                if re.fullmatch(r"[A-Za-z]{2,}", latin):
                    key = unicodedata.normalize("NFC", native)
                    references.setdefault(key, set()).add(latin.lower())
    return references


def prompt(items: list[dict[str, object]]) -> str:
    """Render one deterministic, data-minimized review prompt."""
    submitted = [{key: row[key] for key in ("id", "native")} for row in items]
    return RUBRIC + "\n" + json.dumps(submitted, ensure_ascii=False)


def prepare(
    surnames: Path,
    current_map: Path,
    current_transliterations: Path,
    reference_dir: Path,
    controls_manifest: Path,
    output: Path,
) -> dict[str, object]:
    """Create three independently ordered reviews for every selected unmapped token."""
    output.mkdir(exist_ok=False)
    mapping = json.loads(current_map.read_text())
    existing = {
        row["native"]: row
        for row in pq.read_table(current_transliterations).to_pylist()
    }
    with duckdb.connect() as connection:
        rows = connection.execute(
            "SELECT surname_source_normalized AS native,count(*) AS occurrences "
            "FROM read_parquet(?) WHERE NOT abstained GROUP BY 1 ORDER BY 1",
            [str(surnames)],
        ).fetchall()
    missing = [
        {"native": native, "occurrences": occurrences}
        for native, occurrences in rows
        if native not in mapping
    ]
    references = load_references(reference_dir)
    controls_source = json.loads(controls_manifest.read_text())
    controls = [
        row for row in controls_source["batches"][0]["items"] if row["role"] != "source"
    ]
    pq.write_table(pa.Table.from_pylist(missing), output / "input_tokens.parquet")
    batches = []
    for pass_name in PASSES:
        ordered = sorted(
            missing,
            key=lambda row: hashlib.sha256(
                f"{pass_name}:{row['native']}".encode()
            ).hexdigest(),
        )
        for start in range(0, len(ordered), 128):
            number = start // 128 + 1
            source_items = [
                {
                    "id": f"{pass_name}{start + offset:05d}",
                    "native": row["native"],
                    "role": "source",
                    "occurrences": row["occurrences"],
                    "attested_forms": sorted(references.get(row["native"], ())),
                    "prior_reviewed": row["native"] in existing,
                }
                for offset, row in enumerate(ordered[start : start + 128])
            ]
            items = source_items + controls
            items.sort(key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
            content = prompt(items)
            reserve = (
                Decimal(len(content.encode()) + 1024) * INPUT_RATE
                + Decimal(10_000) * OUTPUT_RATE
            ) / Decimal(1_000_000)
            batches.append(
                {
                    "pass": pass_name,
                    "number": number,
                    "items": items,
                    "prompt_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "reserve_usd": float(reserve),
                }
            )
    manifest = {
        "model": MODEL,
        "passes": list(PASSES),
        "source_surnames_sha256": sha256(surnames),
        "current_map_sha256": sha256(current_map),
        "current_transliterations_sha256": sha256(current_transliterations),
        "reference_manifest_sha256": sha256(reference_dir / "source_manifest.json"),
        "selected_unique_tokens": len(rows),
        "already_mapped_tokens": len(rows) - len(missing),
        "review_tokens": len(missing),
        "prior_reviewed_tokens": sum(row["native"] in existing for row in missing),
        "never_reviewed_tokens": sum(row["native"] not in existing for row in missing),
        "maximum_cost_usd": sum(batch["reserve_usd"] for batch in batches),
        "concurrency": 4,
        "automatic_retries": 0,
        "ledger_prefix": f"urdu-selected-{output.name}",
        "scope": (
            "Standalone selected Urdu tokens and synthetic controls only; no voter "
            "rows, identifiers, relationships, locations, PDFs, images or filenames."
        ),
        "batches": batches,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def response_text(raw: dict[str, object]) -> str:
    """Extract output text from one Responses API receipt."""
    return "".join(
        part["text"]
        for message in raw.get("output", [])
        if message.get("type") == "message"
        for part in message.get("content", [])
        if part.get("type") == "output_text"
    )


def parse_response(
    batch: dict[str, object], raw: dict[str, object], response_digest: str
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Validate one response and score its embedded controls."""
    if raw.get("status") != "completed" or not raw.get("usage"):
        raise ValueError("Incomplete response or missing usage")
    text = response_text(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    payload = json.loads(text)
    answers = payload if isinstance(payload, list) else payload["items"]
    by_id = {row["id"]: row for row in batch["items"]}
    seen: set[str] = set()
    positive = negative = reference_cases = reference_matches = 0
    rows = []
    format_errors = []
    for answer in answers:
        if set(answer) != {"id", "latin", "alternatives", "status", "source_warning"}:
            raise ValueError("Unexpected response fields")
        key = answer["id"]
        if key not in by_id or key in seen:
            raise ValueError("Duplicate or unknown response ID")
        seen.add(key)
        if answer["status"] not in {"supported", "uncertain", "unsupported"}:
            raise ValueError("Invalid response status")
        if not isinstance(answer["source_warning"], bool):
            raise ValueError("Invalid source warning")
        if (
            not isinstance(answer["alternatives"], list)
            or len(answer["alternatives"]) > 3
            or any(not isinstance(value, str) for value in answer["alternatives"])
            or not isinstance(answer["latin"], (str, type(None)))
        ):
            raise ValueError("Invalid Latin response type")
        forms = [answer["latin"], *answer["alternatives"]]
        if any(
            value is not None and re.fullmatch(r"[A-Za-z]{2,}", value) is None
            for value in forms
        ):
            format_errors.append(key)
        item = by_id[key]
        normalized = {value.lower() for value in forms if value is not None}
        if item["role"] == "positive_control":
            positive += bool(
                answer["status"] == "supported"
                and (answer["latin"] or "").lower() in item["attested_forms"]
                and not answer["source_warning"]
            )
        elif item["role"] == "negative_control":
            negative += bool(
                answer["status"] == "unsupported"
                and answer["latin"] is None
                and not answer["alternatives"]
            )
        else:
            if item["attested_forms"]:
                reference_cases += 1
                reference_matches += bool(normalized & set(item["attested_forms"]))
            rows.append(
                {
                    **item,
                    **answer,
                    "pass": batch["pass"],
                    "batch": batch["number"],
                    "response_sha256": response_digest,
                }
            )
    omitted_ids = sorted(set(by_id) - seen)
    passed = not format_errors and not omitted_ids and positive == 8 and negative == 4
    gate = {
        "pass": batch["pass"],
        "batch": batch["number"],
        "source_tokens": len(rows),
        "positive_controls_passed": positive,
        "negative_controls_passed": negative,
        "reference_cases": reference_cases,
        "reference_matches": reference_matches,
        "reference_overlap_fraction": (
            reference_matches / reference_cases if reference_cases else None
        ),
        "reference_overlap_is_diagnostic_only": True,
        "format_error_ids": format_errors,
        "omitted_ids": omitted_ids,
        "gate_passed": passed,
        "response_sha256": response_digest,
    }
    return rows, gate


def refresh_ledger(ledger: dict[str, object]) -> None:
    """Recompute reported and conservatively reserved exposure."""
    reported = sum(row.get("estimated_cost_usd", 0) for row in ledger["calls"])
    reserved = sum(
        row.get("reserved_usd", 0)
        for row in ledger["calls"]
        if "estimated_cost_usd" not in row
    )
    ledger.update(
        reported_usage_estimated_cost_usd=round(reported, 9),
        unreported_usage_reserved_usd=round(reserved, 9),
        accounted_exposure_usd=round(reported + reserved, 9),
    )


def verify_model(api_key: str) -> None:
    """Verify the exact model and current provider rates before spending."""
    request = urllib.request.Request(
        "https://api.meta.ai/muse-code/models",
        headers={"Authorization": "Bearer " + api_key},
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        models = json.load(response)
    candidates = models.get("data", []) if isinstance(models, dict) else models
    model = next((row for row in candidates if row["id"] == MODEL), None)
    if model is None:
        raise ValueError("Authorized Contributor model is unavailable")
    rates = model["metadata"]["muse-code"]["cost"]
    if rates.get("currency") != "USD" or any(
        Decimal(str(rates[key])) != expected
        for key, expected in (
            ("input", INPUT_RATE),
            ("output", OUTPUT_RATE),
            ("cached", CACHED_RATE),
        )
    ):
        raise ValueError("Provider pricing changed")


def usage_cost(raw: dict[str, object]) -> float:
    """Compute provider-reported token cost at the verified rates."""
    usage = raw["usage"]
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    return float(
        (
            Decimal(usage["input_tokens"] - cached) * INPUT_RATE
            + Decimal(cached) * CACHED_RATE
            + Decimal(usage["output_tokens"]) * OUTPUT_RATE
        )
        / Decimal(1_000_000)
    )


def run(
    output: Path,
    ledger_path: Path,
    auth_path: Path,
    *,
    continue_after_gate_failure: bool = False,
) -> dict[str, object]:
    """Execute or replay the bounded three-pass Contributor review."""
    from openai import OpenAI

    manifest = json.loads((output / "manifest.json").read_text())
    auth = json.loads(auth_path.read_text())
    api_key = auth["providers"]["meta"]["api_key"]
    verify_model(api_key)
    client = OpenAI(
        base_url="https://api.meta.ai/v1",
        api_key=api_key,
        max_retries=0,
        timeout=180,
    )
    ledger = json.loads(ledger_path.read_text())
    if ledger["model"] != MODEL or ledger["authorized_total_cap_usd"] != 10:
        raise PermissionError("Ledger does not authorize this model and total cap")
    manifest_digest = sha256(output / "manifest.json")
    ledger_prefix = manifest.get("ledger_prefix", "urdu-selected-final")
    decisions = []
    for start in range(0, len(manifest["batches"]), 4):
        wave = manifest["batches"][start : start + 4]
        pending = []
        records = {}
        receipts = {}
        for batch in wave:
            call_id = f"{ledger_prefix}-{batch['pass']}-{batch['number']:02d}"
            response_path = output / (
                f"response_{batch['pass']}_{batch['number']:02d}.json"
            )
            found = [row for row in ledger["calls"] if row["id"] == call_id]
            if response_path.exists():
                if len(found) != 1:
                    raise ValueError("Receipt and ledger disagree")
                records[call_id] = found[0]
                receipts[call_id] = json.loads(response_path.read_text())
                continue
            if found:
                raise ValueError("Unfinished request cannot be repeated silently")
            if (
                ledger["accounted_exposure_usd"] + batch["reserve_usd"]
                > ledger["authorized_total_cap_usd"]
            ):
                raise PermissionError("Authorized total cap would be exceeded")
            record = {
                "id": call_id,
                "status": "started",
                "model": MODEL,
                "reserved_usd": batch["reserve_usd"],
                "automatic_retries": 0,
                "manifest_sha256": manifest_digest,
                "prompt_sha256": batch["prompt_sha256"],
            }
            ledger["calls"].append(record)
            refresh_ledger(ledger)
            write_json(ledger_path, ledger)
            records[call_id] = record
            pending.append((batch, call_id, response_path))

        def submit(batch: dict[str, object]) -> dict[str, object]:
            content = prompt(batch["items"])
            if hashlib.sha256(content.encode()).hexdigest() != batch["prompt_sha256"]:
                raise ValueError("Prompt changed after authorization")
            return client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": content}],
                reasoning={"effort": "low"},
                max_output_tokens=10_000,
                store=False,
            ).model_dump(mode="json")

        errors = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [(item, pool.submit(submit, item[0])) for item in pending]
            for (_batch, call_id, response_path), future in futures:
                record = records[call_id]
                try:
                    raw = future.result()
                except Exception as error:
                    record.update(status="error", exception_type=type(error).__name__)
                    errors.append(type(error).__name__)
                    refresh_ledger(ledger)
                    write_json(ledger_path, ledger)
                    continue
                write_json(response_path, raw)
                receipts[call_id] = raw
                record.update(
                    status=raw.get("status"),
                    usage=raw.get("usage"),
                    estimated_cost_usd=usage_cost(raw),
                )
                refresh_ledger(ledger)
                write_json(ledger_path, ledger)
        gates = []
        for batch in wave:
            call_id = f"{ledger_prefix}-{batch['pass']}-{batch['number']:02d}"
            if call_id not in receipts:
                continue
            response_path = output / (
                f"response_{batch['pass']}_{batch['number']:02d}.json"
            )
            rows, gate = parse_response(batch, receipts[call_id], sha256(response_path))
            write_json(
                output / f"gate_{batch['pass']}_{batch['number']:02d}.json", gate
            )
            gates.append(gate["gate_passed"])
            decisions.extend(
                {**row, "batch_gate_passed": gate["gate_passed"]} for row in rows
            )
            if records[call_id]["estimated_cost_usd"] > batch["reserve_usd"]:
                raise ValueError("Call exceeded its reservation")
        write_json(output / "decisions.json", decisions)
        if errors or (not all(gates) and not continue_after_gate_failure):
            raise RuntimeError(
                f"Review wave failed; stopped before more calls: {errors}"
            )
    expected = manifest["review_tokens"] * len(PASSES)
    if len(decisions) > expected:
        raise ValueError("Decision count exceeds the manifest")
    unique = {(row["pass"], row["native"]) for row in decisions}
    if len(unique) != len(decisions):
        raise ValueError("Review repeats a token within a pass")
    return {
        "review_decisions": len(decisions),
        "omitted_decisions": expected - len(decisions),
        "reported_phase_cost_usd": sum(
            row.get("estimated_cost_usd", 0)
            for row in ledger["calls"]
            if row["id"].startswith(ledger_prefix + "-")
        ),
        "accounted_total_usd": ledger["accounted_exposure_usd"],
    }


def majority_reading(rows: list[dict[str, object]]) -> str | None:
    """Choose a unique supported primary with at least two independent votes."""
    votes = Counter(
        row["latin"].lower()
        for row in rows
        if row["batch_gate_passed"]
        and row["status"] == "supported"
        and isinstance(row["latin"], str)
    )
    if not votes:
        return None
    ranked = votes.most_common()
    if ranked[0][1] < 2 or (len(ranked) > 1 and ranked[1][1] == ranked[0][1]):
        return None
    return ranked[0][0]


def adjudicated_reading(
    rows: list[dict[str, object]], prior: str | None
) -> tuple[str | None, str]:
    """Resolve a supported tie by prior-form similarity after repeated review."""
    strict = majority_reading(rows)
    if strict is not None:
        return strict, "selected_multi_pass_majority"
    votes = Counter(
        row["latin"].lower()
        for row in rows
        if row["batch_gate_passed"]
        and row["status"] == "supported"
        and isinstance(row["latin"], str)
    )
    if not votes or max(votes.values()) < 2:
        return None, "insufficient_supported_votes"
    maximum = max(votes.values())
    leaders = sorted(value for value, count in votes.items() if count == maximum)
    normalized_prior = prior.lower() if isinstance(prior, str) else None
    if normalized_prior in leaders:
        return normalized_prior, "selected_tie_prior_primary"
    if normalized_prior is not None:
        leaders.sort(
            key=lambda value: (
                -difflib.SequenceMatcher(None, normalized_prior, value).ratio(),
                value,
            )
        )
        return leaders[0], "selected_tie_prior_similarity"
    return leaders[0], "selected_tie_lexical"


def promote_reviews(
    surnames: Path,
    current_map: Path,
    current_transliterations: Path,
    review_dirs: list[Path],
    shared_root: Path,
    ledger_path: Path,
) -> dict[str, object]:
    """Promote corroborated selected-token readings into shared local artifacts."""
    if (
        shared_root != current_map.parent
        or shared_root != current_transliterations.parent
    ):
        raise ValueError("Shared artifacts must have one root")
    mapping = json.loads(current_map.read_text())
    prior_rows = pq.read_table(current_transliterations).to_pylist()
    prior_by_native = {row["native"]: row for row in prior_rows}
    if len(prior_by_native) != len(prior_rows):
        raise ValueError("Current transliteration artifact repeats native tokens")
    with duckdb.connect() as connection:
        selected = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT surname_source_normalized FROM read_parquet(?) "
                "WHERE NOT abstained",
                [str(surnames)],
            ).fetchall()
        }
    missing = selected - set(mapping)
    grouped: dict[str, list[dict[str, object]]] = {}
    phase_reports = []
    for directory in review_dirs:
        manifest = json.loads((directory / "manifest.json").read_text())
        decisions = json.loads((directory / "decisions.json").read_text())
        for row in decisions:
            if row["native"] not in missing:
                raise ValueError(
                    "Review contains an already mapped or unselected token"
                )
            grouped.setdefault(row["native"], []).append(row)
        gates = [
            json.loads(path.read_text())
            for path in sorted(directory.glob("gate_*.json"))
        ]
        phase_reports.append(
            {
                "directory": directory.name,
                "manifest_sha256": sha256(directory / "manifest.json"),
                "decisions_sha256": sha256(directory / "decisions.json"),
                "review_tokens": manifest["review_tokens"],
                "maximum_cost_usd": manifest["maximum_cost_usd"],
                "batches": len(gates),
                "passing_batches": sum(gate["gate_passed"] for gate in gates),
                "quarantined_batches": sum(not gate["gate_passed"] for gate in gates),
            }
        )
    if set(grouped) != missing:
        raise ValueError("Reviews do not cover every selected unmapped token")
    promoted = {}
    statuses = Counter()
    for native in sorted(missing):
        prior = prior_by_native.get(native, {}).get("latin")
        latin, status = adjudicated_reading(grouped[native], prior)
        if latin is None or re.fullmatch(r"[a-z]{2,}", latin) is None:
            raise ValueError(f"No supported adjudication for {native}")
        promoted[native] = latin
        statuses[status] += 1
    if set(mapping) & set(promoted):
        raise ValueError("Promotion would replace existing map entries")
    history = shared_root / "history" / "before_calibrated_selection"
    names = (
        "transliterations.parquet",
        "urdu.csv.gz",
        "upnaam_map.json",
        "provenance.json",
        "validation.json",
    )
    if not history.exists():
        history.mkdir(parents=True)
        for name in names:
            shutil.copy2(shared_root / name, history / name)
    else:
        for name in names:
            if not (history / name).exists():
                raise ValueError("Incomplete pre-promotion history snapshot")
    final_rows = dict(prior_by_native)
    for native, latin in promoted.items():
        reviews = grouped[native]
        prior = prior_by_native.get(native)
        alternatives = sorted(
            {
                value.lower()
                for row in reviews
                for value in [row.get("latin"), *row.get("alternatives", [])]
                if isinstance(value, str)
                and re.fullmatch(r"[A-Za-z]{2,}", value)
                and value.lower() != latin
            }
        )
        response_digest = hashlib.sha256(
            "|".join(sorted({row["response_sha256"] for row in reviews})).encode()
        ).hexdigest()
        _chosen, status = adjudicated_reading(
            reviews, prior.get("latin") if prior else None
        )
        final_rows[native] = {
            "native": native,
            "latin": latin,
            "alternatives": alternatives,
            "model_status": "supported",
            "source_warning": any(row["source_warning"] for row in reviews),
            "batch_gate_passed": True,
            "evidence_status": status,
            "lookup_eligible": True,
            "attested_latin_forms": sorted(
                {
                    value.lower()
                    for row in reviews
                    for value in row.get("attested_forms", [])
                }
            ),
            "response_sha256": response_digest,
            "wave_gate_passed": True,
        }
    schema = pq.read_schema(current_transliterations)
    pq.write_table(
        pa.Table.from_pylist(
            [final_rows[key] for key in sorted(final_rows)], schema=schema
        ),
        current_transliterations,
        compression="zstd",
    )
    mapping.update(promoted)
    ordered_mapping = {key: mapping[key] for key in sorted(mapping)}
    current_map.write_text(
        json.dumps(ordered_mapping, ensure_ascii=False, indent=2) + "\n"
    )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["urdu", "english"])
    writer.writerows(ordered_mapping.items())
    corpus = shared_root / "urdu.csv.gz"
    corpus.write_bytes(
        gzip.compress(buffer.getvalue().encode(), compresslevel=9, mtime=0)
    )
    ledger = json.loads(ledger_path.read_text())
    prefixes = {
        (
            "urdu-selected-final-"
            if directory.name == "selected_final"
            else f"urdu-selected-{directory.name}-"
        )
        for directory in review_dirs
    }
    paid_calls = [
        row for row in ledger["calls"] if any(row["id"].startswith(p) for p in prefixes)
    ]
    phase_cost = sum(row.get("estimated_cost_usd", 0) for row in paid_calls)
    transliterations = shared_root / "transliterations.parquet"
    provenance_path = shared_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    evidence_counts = Counter(row["evidence_status"] for row in final_rows.values())
    provenance.update(
        unique_decoded_tokens=len(final_rows),
        reviewed_tokens=len(final_rows),
        pending_tokens=0,
        lookup_pairs=len(ordered_mapping),
        evidence_status_counts=dict(sorted(evidence_counts.items())),
    )
    provenance["selected_calibrated_extension"] = {
        "selected_unique_tokens": len(selected),
        "previously_mapped": len(selected) - len(promoted),
        "promoted": len(promoted),
        "unmapped_after_promotion": len(selected - set(ordered_mapping)),
        "selection_policy": (
            "Use supported answers from batches with perfect controls, complete IDs "
            "and valid ASCII output; choose a unique repeated winner, then resolve "
            "persistent top-count ties by the closest pre-existing reviewed primary."
        ),
        "status_counts": dict(sorted(statuses.items())),
        "phases": phase_reports,
        "paid_calls": len(paid_calls),
        "reported_phase_cost_usd": phase_cost,
        "accounted_project_exposure_usd": ledger["accounted_exposure_usd"],
        "surnames_sha256": sha256(surnames),
    }
    provenance["artifacts"] = {
        "transliterations.parquet": sha256(transliterations),
        "urdu.csv.gz": sha256(corpus),
    }
    provenance["upnaam_map"] = {
        "path": "upnaam_map.json",
        "sha256": sha256(current_map),
        "entries": len(ordered_mapping),
        "derivation": "Accepted corpus pairs plus corroborated selected-token reviews.",
    }
    write_json(provenance_path, provenance)
    validation_path = shared_root / "validation.json"
    validation = json.loads(validation_path.read_text())
    validation.update(
        status="shared_local_corpus_validated",
        reviewed_unique_tokens=len(final_rows),
        pending_tokens=0,
        lookup_pairs=len(ordered_mapping),
        quarantined_rows_in_lookup=0,
        selected_handoff={
            "unique_selected_tokens": len(selected),
            "mapped_selected_tokens": len(selected & set(ordered_mapping)),
            "unmapped_selected_tokens": len(selected - set(ordered_mapping)),
        },
    )
    validation["artifacts"].update(
        {
            "transliterations.parquet": sha256(transliterations),
            "urdu.csv.gz": sha256(corpus),
            "upnaam_map.json": sha256(current_map),
        }
    )
    validation.pop("installed_downstream_validation", None)
    write_json(validation_path, validation)
    return {
        "selected_unique_tokens": len(selected),
        "promoted_tokens": len(promoted),
        "mapped_selected_tokens": len(selected & set(ordered_mapping)),
        "lookup_pairs": len(ordered_mapping),
        "reported_phase_cost_usd": phase_cost,
        "accounted_project_exposure_usd": ledger["accounted_exposure_usd"],
    }


def main() -> None:
    """Prepare, run, or inspect the selected-token review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "promote"))
    parser.add_argument("--surnames", type=Path, required=True)
    parser.add_argument("--current-map", type=Path, required=True)
    parser.add_argument("--current-transliterations", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--controls-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--auth", type=Path, required=True)
    parser.add_argument("--continue-after-gate-failure", action="store_true")
    parser.add_argument("--review-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(
            args.surnames,
            args.current_map,
            args.current_transliterations,
            args.reference_dir,
            args.controls_manifest,
            args.output,
        )
        sys.stdout.write(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "selected_unique_tokens",
                        "already_mapped_tokens",
                        "review_tokens",
                        "prior_reviewed_tokens",
                        "never_reviewed_tokens",
                        "maximum_cost_usd",
                    )
                },
                indent=2,
            )
            + "\n"
        )
    elif args.action == "run":
        result = run(
            args.output,
            args.ledger,
            args.auth,
            continue_after_gate_failure=args.continue_after_gate_failure,
        )
        sys.stdout.write(json.dumps(result, indent=2))
        sys.stdout.write("\n")
    else:
        if not args.review_dir:
            parser.error("promote requires at least one --review-dir")
        result = promote_reviews(
            args.surnames,
            args.current_map,
            args.current_transliterations,
            args.review_dir,
            args.output,
            args.ledger,
        )
        sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
