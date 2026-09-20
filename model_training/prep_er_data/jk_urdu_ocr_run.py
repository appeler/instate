"""Submit a prepared Urdu OCR checkpoint under an explicitly approved total budget."""

import argparse
import base64
import fcntl
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from model_training.prep_er_data.jk_urdu_ocr import (
    MODEL,
    PROMPT,
    RESERVE_NANO,
    REVISION,
    Renderer,
    checkpoint_summary,
    decode_received,
    digest,
    exposure,
    journal,
    metadata,
    next_images,
    receive,
    reserve,
    selected_images,
)


def nanos(value):
    """Convert a dollar amount to an exact integer accounting unit."""
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("Invalid budget amount")
    return int(amount * 1_000_000_000)


def save_json(path, value):
    """Replace a budget receipt atomically."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def locked_ledger(path):
    """Serialize reservation changes across separate OCR journals."""
    path = Path(path).resolve()
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield json.loads(path.read_text())


def claim_budget(conn, authorization_path):
    """Check explicit approval and reserve the job's whole allowance in the ledger."""
    authorization = json.loads(Path(authorization_path).read_text())
    config = metadata(conn, "config")
    if authorization.get("status") != "approved":
        raise PermissionError("The proposed full OCR budget has not been approved")
    if not metadata(conn, "prepared") or authorization.get("model") != MODEL:
        raise ValueError("Unprepared checkpoint or wrong authorized model")
    if authorization.get("workload_sha256") != config["workload_sha256"]:
        raise ValueError("Authorization is for a different workload")
    if authorization.get("revision") != REVISION:
        raise ValueError("Authorization is for a different OCR policy")
    ledger_path = Path(authorization["ledger_path"]).resolve()
    with locked_ledger(ledger_path) as ledger:
        maximum = nanos(authorization["maximum_total_usd"])
        if ledger["model"] != MODEL or maximum > nanos(
            ledger["authorized_total_cap_usd"]
        ):
            raise PermissionError(
                "Requested job exceeds the ledger's authorized total cap"
            )
        identity = "jk-full-ocr-" + REVISION + "-" + config["workload_sha256"]
        others = [r for r in ledger["calls"] if r["id"] != identity]
        if any(r.get("status") == "started" for r in others):
            raise ValueError("Another ledger-writing request is still active")
        prior = sum(
            nanos(r.get("estimated_cost_usd", r.get("reserved_usd", 0))) for r in others
        )
        budget = maximum - prior
        if budget <= 0:
            raise PermissionError("No authorized funds remain")
        binding = {
            "authorization_sha256": digest(authorization_path),
            "journal_path": str(
                Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve()
            ),
            "ledger_path": str(ledger_path),
            "job_id": identity,
            "budget_nano": budget,
            "prior_exposure_nano": prior,
        }
        saved = metadata(conn, "authorization")
        if saved is not None and saved != binding:
            raise ValueError(
                "Existing job authorization changed; inspect before resuming"
            )
        found = [r for r in ledger["calls"] if r["id"] == identity]
        if len(found) > 1:
            raise ValueError("Duplicate job reservation")
        if not found:
            ledger["calls"].append(
                {
                    "id": identity,
                    "status": "job_reserved",
                    "model": MODEL,
                    "reserved_usd": budget / 1e9,
                    "authorization_sha256": binding["authorization_sha256"],
                    "automatic_retries": 0,
                    "journal_path": binding["journal_path"],
                }
            )
            refresh_ledger(ledger)
            save_json(ledger_path, ledger)
        elif (
            found[0]["authorization_sha256"] != binding["authorization_sha256"]
            or found[0].get("journal_path") != binding["journal_path"]
        ):
            raise ValueError("Ledger reservation belongs to a different authorization")
        metadata(conn, "authorization", binding)
        return budget


def refresh_ledger(ledger):
    """Keep unresolved requests and the unspent job allowance reserved."""
    reported = sum(nanos(r.get("estimated_cost_usd", 0)) for r in ledger["calls"])
    reserved = sum(
        nanos(r.get("reserved_usd", 0))
        for r in ledger["calls"]
        if "estimated_cost_usd" not in r
    )
    ledger.update(
        reported_usage_estimated_cost_usd=reported / 1e9,
        unreported_usage_reserved_usd=reserved / 1e9,
        accounted_exposure_usd=(reported + reserved) / 1e9,
    )


class MuseTransport:
    """Use the validated Contributor model, with no SDK or application retries."""

    def __init__(self):
        """Load credentials, verify current pricing, and configure the client."""
        from openai import OpenAI

        auth = json.loads((Path.home() / ".config/muse/auth.json").read_text())
        key = auth["providers"]["meta"]["api_key"]
        request = urllib.request.Request(
            "https://api.meta.ai/muse-code/models",
            headers={"Authorization": "Bearer " + key},
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            models = json.load(response)
        candidates = models.get("data", []) if isinstance(models, dict) else models
        model = next((row for row in candidates if row["id"] == MODEL), None)
        if model is None:
            raise ValueError("Authorized Contributor model is not available")
        details = model["metadata"]["muse-code"]
        rates = details["cost"]
        if rates.get("currency") != "USD":
            raise ValueError("Provider billing currency changed")
        if any(
            Decimal(str(rates[k])) != expected
            for k, expected in [
                ("input", Decimal(".10")),
                ("output", Decimal(".20")),
                ("cached", Decimal(".002")),
            ]
        ):
            raise ValueError("Provider prices changed; recalculate before submitting")
        if details["limit"]["context"] > 1_007_997:
            raise ValueError("Model context bound changed; verify call reservation")
        self.client = OpenAI(
            base_url="https://api.meta.ai/v1", api_key=key, max_retries=0, timeout=180
        )

    def __call__(self, images):
        """Submit one bounded set of paired Urdu-name crops."""
        content = [{"type": "input_text", "text": PROMPT}]
        for local_id, (_, png) in images.items():
            content.extend(
                [
                    {"type": "input_text", "text": "ID: " + local_id},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,"
                        + base64.b64encode(png).decode(),
                    },
                ]
            )
        item = {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "items": {"type": ["string", "null"]},
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": list(images),
            "properties": dict.fromkeys(images, item),
        }
        return self.client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": "low"},
            max_output_tokens=10000,
            store=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "paired_urdu_names",
                    "strict": True,
                    "schema": schema,
                }
            },
        ).model_dump(mode="json")


def execute(
    conn, transport, budget_nano, max_requests, concurrency=4, selected_cards=None
):
    """Resume receipts and refill available slots, stopping on request failures."""
    if max_requests < 1 or not 1 <= concurrency <= 16:
        raise ValueError("Invalid request or concurrency bound")
    config = metadata(conn, "config")
    if (
        config["model"] != MODEL
        or config["revision"] != REVISION
        or config["prompt_sha256"] != hashlib.sha256(PROMPT.encode()).hexdigest()
    ):
        raise ValueError("OCR policy differs from the prepared checkpoint")
    decode_received(conn)
    failures = conn.execute(
        "SELECT count(*) FROM calls WHERE status IN ('reserved','failed','invalid') "
        "OR cost_nano IS NULL OR cost_nano>reserve_nano"
    ).fetchone()[0]
    if failures:
        raise ValueError(
            "Unresolved prior request: inspect its receipt before continuing"
        )
    renderer = Renderer(metadata(conn, "config")["source_dir"])
    batches = (
        selected_images(conn, renderer, selected_cards)
        if selected_cards is not None
        else None
    )
    made = 0
    stopped = False
    stop_reason = "request_limit"

    def submit(images):
        start = time.monotonic()
        return transport(images), time.monotonic() - start

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = {}
            run_error = None
            while pending or (made < max_requests and not stopped):
                while (
                    len(pending) < concurrency and made < max_requests and not stopped
                ):
                    if exposure(conn) + RESERVE_NANO > budget_nano:
                        if not pending:
                            stop_reason = "authorized_budget_limit"
                            stopped = True
                        break
                    try:
                        images = (
                            next(batches, {})
                            if batches is not None
                            else next_images(conn, renderer)
                        )
                        if not images:
                            stop_reason = "queue_exhausted"
                            stopped = True
                            break
                        call = reserve(conn, images, budget_nano)
                    except Exception as error:
                        run_error = error
                        stopped = True
                        break
                    pending[pool.submit(submit, images)] = call
                    made += 1
                if not pending:
                    break
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    call = pending.pop(future)
                    try:
                        raw, elapsed = future.result()
                    except Exception as error:
                        with conn:
                            conn.execute(
                                "UPDATE calls SET status='failed',error=? WHERE id=?",
                                (
                                    json.dumps(
                                        {
                                            "type": type(error).__name__,
                                            "provider_body": getattr(
                                                error, "body", None
                                            ),
                                        }
                                    ),
                                    call,
                                ),
                            )
                        stopped = True
                        continue
                    try:
                        receive(conn, call, raw, elapsed)
                    except ValueError as error:
                        run_error = error
                        stopped = True
                    if raw.get("usage") is None:
                        stop_reason = "missing_usage"
                        stopped = True
                decode_received(conn)
                if conn.execute(
                    "SELECT count(*) FROM calls WHERE status IN ('invalid','failed')"
                ).fetchone()[0]:
                    stop_reason = "request_or_response_failure"
                    stopped = True
            if run_error is not None:
                raise run_error
    finally:
        renderer.close()
    result = {
        **checkpoint_summary(conn),
        "new_requests": made,
        "stop_reason": stop_reason,
    }
    metadata(conn, "last_run", result)
    return result


def retry_failed_requests(
    conn,
    transport,
    budget_nano,
    call_ids,
    reason,
    concurrency=1,
    maximum_attempts=2,
):
    """Retry inspected receipt-less failures once, retaining every original attempt."""
    if (
        not call_ids
        or len(set(call_ids)) != len(call_ids)
        or any(type(call_id) is not int for call_id in call_ids)
        or not reason.strip()
        or not 1 <= concurrency <= 16
        or maximum_attempts < 2
    ):
        raise ValueError(
            "Unique failed call IDs, a reason and valid concurrency are required"
        )
    retries = {}
    for call_id in call_ids:
        retries[call_id] = retry_images(conn, call_id, maximum_attempts)
    if exposure(conn) + RESERVE_NANO * len(retries) > budget_nano:
        raise PermissionError("Retries plus the failed attempts exceed the budget")
    with conn:
        for call_id in retries:
            attempt = (
                conn.execute(
                    "SELECT count(*) FROM failed_attempts WHERE call_id=?", (call_id,)
                ).fetchone()[0]
                + 1
            )
            conn.execute(
                "INSERT INTO failed_attempts "
                "SELECT id,?,status,reserve_nano,cost_nano,response,error,"
                "elapsed_seconds,? FROM calls WHERE id=?",
                (attempt, reason, call_id),
            )
            conn.execute(
                "UPDATE calls SET status='reserved',cost_nano=NULL,response=NULL,"
                "error=NULL,elapsed_seconds=NULL WHERE id=?",
                (call_id,),
            )

    def submit(images):
        started = time.monotonic()
        return transport(images), time.monotonic() - started

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {
            pool.submit(submit, images): call_id for call_id, images in retries.items()
        }
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                call_id = pending.pop(future)
                try:
                    raw, elapsed = future.result()
                except Exception as error:
                    with conn:
                        conn.execute(
                            "UPDATE calls SET status='failed',error=? WHERE id=?",
                            (
                                json.dumps(
                                    {
                                        "type": type(error).__name__,
                                        "provider_body": getattr(error, "body", None),
                                    }
                                ),
                                call_id,
                            ),
                        )
                else:
                    receive(conn, call_id, raw, elapsed)
                decode_received(conn)
    return {
        **checkpoint_summary(conn),
        "retried_calls": call_ids,
        "new_requests": len(call_ids),
    }


def retry_images(conn, call_id, maximum_attempts):
    """Validate and reproduce the original images for one retryable call."""
    call = conn.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
    if call is None or call["status"] != "failed":
        raise ValueError("A failed call is required")
    error = json.loads(call["error"])
    if (
        error.get("type")
        not in {
            "APITimeoutError",
            "InternalServerError",
        }
        or call["response"] is not None
    ):
        raise ValueError(
            "Only an inspected server error or timeout without a receipt can retry"
        )
    prior_failures = conn.execute(
        "SELECT count(*) FROM failed_attempts WHERE call_id=?", (call_id,)
    ).fetchone()[0]
    if prior_failures >= maximum_attempts - 1:
        raise ValueError("This call has reached its authorized attempt limit")
    config = metadata(conn, "config")
    if (
        config["model"] != MODEL
        or config["revision"] != REVISION
        or config["prompt_sha256"] != hashlib.sha256(PROMPT.encode()).hexdigest()
    ):
        raise ValueError("OCR policy differs from the prepared checkpoint")
    items = conn.execute(
        "SELECT local_id,crop_key FROM request_items WHERE call_id=? ORDER BY local_id",
        (call_id,),
    ).fetchall()
    if not 1 <= len(items) <= 26:
        raise ValueError("Invalid original request size")
    images = {}
    renderer = Renderer(config["source_dir"])
    try:
        for item in items:
            row = conn.execute(
                "SELECT c.*,s.filename,s.sha256 FROM cards c "
                "JOIN sources s ON s.id=c.source_id "
                "JOIN crops p ON p.key=c.crop_key "
                "WHERE c.crop_key=? AND p.status='submitted' LIMIT 1",
                (item["crop_key"],),
            ).fetchone()
            if row is None:
                raise ValueError("The failed request's crop state changed")
            key, png = renderer.render(row)
            if key != item["crop_key"]:
                raise ValueError("Retry source pixels differ from the failed request")
            images[item["local_id"]] = (key, png)
    finally:
        renderer.close()
    return images


def retry_failed_request(
    conn, transport, budget_nano, call_id, reason, maximum_attempts=2
):
    """Retry one inspected receipt-less failure."""
    result = retry_failed_requests(
        conn,
        transport,
        budget_nano,
        [call_id],
        reason,
        concurrency=1,
        maximum_attempts=maximum_attempts,
    )
    result["retried_call"] = call_id
    return result


def abandon_failed_call(conn, call_id, reason):
    """Quarantine a repeatedly failed crop set while preserving its last attempt."""
    call = conn.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
    if (
        call is None
        or call["status"] != "failed"
        or call["response"] is not None
        or not reason.strip()
    ):
        raise ValueError(
            "A receipt-less failed call and an explicit reason are required"
        )
    attempt = (
        conn.execute(
            "SELECT count(*) FROM failed_attempts WHERE call_id=?", (call_id,)
        ).fetchone()[0]
        + 1
    )
    with conn:
        conn.execute(
            "INSERT INTO failed_attempts "
            "SELECT id,?,status,reserve_nano,cost_nano,response,error,"
            "elapsed_seconds,? FROM calls WHERE id=?",
            (attempt, reason, call_id),
        )
        conn.execute(
            "UPDATE calls SET status='abandoned',reserve_nano=0,cost_nano=0,"
            "error=? WHERE id=?",
            (json.dumps({"reason": reason, "archived_attempt": attempt}), call_id),
        )
        conn.execute(
            "UPDATE crops SET status='abandoned' WHERE key IN "
            "(SELECT crop_key FROM request_items WHERE call_id=?)",
            (call_id,),
        )
    return {**checkpoint_summary(conn), "abandoned_call": call_id}


def settle(conn, *, close=False, close_reason=None):
    """Publish observed cost and release the unused allowance when work is closed."""
    binding = metadata(conn, "authorization")
    ledger_path = Path(binding["ledger_path"])
    with locked_ledger(ledger_path) as ledger:
        entry = next(r for r in ledger["calls"] if r["id"] == binding["job_id"])
        remaining = conn.execute(
            "SELECT count(*) FROM cards c LEFT JOIN crops p ON c.crop_key=p.key "
            "WHERE p.status IS NULL OR p.status<>'complete'"
        ).fetchone()[0]
        unpriced = conn.execute(
            "SELECT (SELECT count(*) FROM calls WHERE cost_nano IS NULL) + "
            "(SELECT count(*) FROM failed_attempts WHERE cost_nano IS NULL)"
        ).fetchone()[0]
        entry["observed_job_exposure_usd"] = exposure(conn) / 1e9
        if close:
            unresolved_calls = conn.execute(
                "SELECT count(*) FROM calls WHERE status IN "
                "('reserved','received','failed','invalid')"
            ).fetchone()[0]
            if unresolved_calls:
                raise ValueError("Cannot close a job with unresolved calls")
            if not isinstance(close_reason, str) or not close_reason.strip():
                raise ValueError("Closing a job requires a reason")
            entry.update(
                status="closed",
                estimated_cost_usd=exposure(conn) / 1e9,
                close_reason=close_reason.strip(),
                unattempted_or_unselected_cards=remaining,
                unpriced_archived_attempts=unpriced,
            )
        elif remaining == 0 and unpriced == 0:
            entry.update(status="completed", estimated_cost_usd=exposure(conn) / 1e9)
        refresh_ledger(ledger)
        save_json(ledger_path, ledger)


def main():
    """Require an approved, workload-bound budget receipt before creating a client."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retry-call", type=int)
    parser.add_argument("--retry-calls-file", type=Path)
    parser.add_argument("--retry-reason")
    parser.add_argument("--maximum-attempts", type=int, default=2)
    parser.add_argument("--abandon-call", type=int)
    parser.add_argument("--abandon-reason")
    parser.add_argument("--cards-file", type=Path)
    parser.add_argument("--close-job")
    args = parser.parse_args()
    retry_sources = sum(
        value is not None for value in (args.retry_call, args.retry_calls_file)
    )
    if retry_sources > 1 or bool(retry_sources) != (args.retry_reason is not None):
        parser.error("Supply one retry source together with --retry-reason")
    abandoning = (args.abandon_call is not None) or (args.abandon_reason is not None)
    if (args.abandon_call is None) != (args.abandon_reason is None):
        parser.error("--abandon-call and --abandon-reason must be supplied together")
    if (
        sum(
            (
                bool(retry_sources),
                abandoning,
                args.cards_file is not None,
                args.close_job is not None,
            )
        )
        > 1
    ):
        parser.error("Retry, abandonment and new selection are separate operations")
    selected = None
    if args.cards_file is not None:
        selected = json.loads(args.cards_file.read_text())
        if not isinstance(selected, list):
            parser.error("--cards-file must contain a JSON list of source record keys")
    with journal(args.journal) as conn:
        budget = claim_budget(conn, args.authorization)
        if args.close_job is not None:
            result = checkpoint_summary(conn)
        elif args.abandon_call is not None:
            result = abandon_failed_call(conn, args.abandon_call, args.abandon_reason)
        elif args.retry_call is not None:
            result = retry_failed_request(
                conn,
                MuseTransport(),
                budget,
                args.retry_call,
                args.retry_reason,
                args.maximum_attempts,
            )
        elif args.retry_calls_file is not None:
            calls = json.loads(args.retry_calls_file.read_text())
            if not isinstance(calls, list):
                parser.error("--retry-calls-file must contain a JSON list")
            result = retry_failed_requests(
                conn,
                MuseTransport(),
                budget,
                calls,
                args.retry_reason,
                args.concurrency,
                args.maximum_attempts,
            )
        else:
            result = execute(
                conn,
                MuseTransport(),
                budget,
                args.max_requests,
                args.concurrency,
                selected_cards=selected,
            )
        settle(conn, close=args.close_job is not None, close_reason=args.close_job)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
