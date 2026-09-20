"""Verify OCR identity preservation, exact reuse and interruption-safe spending."""

import hashlib
import json
import threading
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf
import pytest

from model_training.prep_er_data.jk_urdu_ocr import (
    MODEL,
    RESERVE_NANO,
    REVISION,
    Renderer,
    checkpoint_summary,
    crop_box,
    crop_digest,
    digest,
    export_readings,
    exposure,
    journal,
    next_images,
    parse_response,
    prepare,
    receive,
    reserve,
    selected_images,
    usage_cost,
)
from model_training.prep_er_data.jk_urdu_ocr_run import (
    abandon_failed_call,
    claim_budget,
    execute,
    retry_failed_request,
    retry_failed_requests,
    settle,
)


def response(ids, value=None):
    value = value or ["علی", "علی خان", "father", None, None]
    return {
        "status": "completed",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_tokens_details": {"cached_tokens": 10},
        },
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            dict.fromkeys(ids, value), ensure_ascii=False
                        ),
                    }
                ],
            }
        ],
    }


@pytest.fixture
def workload(tmp_path):
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((35, 47), "TEST")
    document.save(tmp_path / "a.pdf")
    document.close()
    (tmp_path / "b.pdf").write_bytes((tmp_path / "a.pdf").read_bytes())
    rows = []
    for name, roles in [("a.pdf", ["own", "relative"]), ("b.pdf", ["relative"])]:
        for role in roles:
            key = name + ":1:1"
            rows.append(
                {
                    "entry_event_key": key,
                    "filename": name,
                    "source_sha256": digest(tmp_path / name),
                    "page": 1,
                    "bbox": [26.0, 20.0, 206.0, 86.0],
                    "role": role,
                    "task_id": hashlib.sha256((key + ":" + role).encode()).hexdigest(),
                }
            )
    path = tmp_path / "workload.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"workload_sha256": digest(path), "rows": 3}))
    return path, manifest, tmp_path


def prepared(conn, workload):
    return prepare(conn, *workload)


def server_failed_call(conn, workload, error_type="InternalServerError"):
    prepared(conn, workload)
    renderer = Renderer(workload[2])
    images = next_images(conn, renderer)
    renderer.close()
    call = reserve(conn, images, RESERVE_NANO)
    error = json.dumps({"type": error_type, "provider_body": {"message": "retry"}})
    with conn:
        conn.execute(
            "UPDATE calls SET status='failed',error=? WHERE id=?", (error, call)
        )
    return call, images, error


@pytest.mark.parametrize("error_type", ["InternalServerError", "APITimeoutError"])
def test_manual_retry_preserves_error_cost_and_original_image_bindings(
    workload, error_type
):
    with journal(workload[2] / "job.sqlite") as conn:
        call, images, error = server_failed_call(conn, workload, error_type)
        seen = []

        def transport(current):
            seen.append(current)
            return response(current)

        result = retry_failed_request(
            conn, transport, RESERVE_NANO * 2, call, "Inspected transient server error"
        )
        assert seen == [images]
        archived = conn.execute("SELECT * FROM failed_attempts").fetchone()
        assert archived["error"] == error
        assert archived["call_id"] == call
        assert archived["cost_nano"] is None
        assert result["total_request_attempts"] == 2
        assert result["call_states"] == {"complete": 1}
        assert exposure(conn) == RESERVE_NANO + usage_cost(response(images))
        assert export_readings(conn, workload[2] / "readings.parquet")["rows"] == 3


def test_bounded_batch_retry_preserves_each_timeout_attempt(workload):
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((35, 47), "DIFFERENT")
    replacement = workload[2] / "replacement.pdf"
    document.save(replacement)
    document.close()
    replacement.replace(workload[2] / "b.pdf")
    rows = pq.read_table(workload[0]).to_pylist()
    for row in rows:
        if row["filename"] == "b.pdf":
            row["source_sha256"] = digest(workload[2] / "b.pdf")
    pq.write_table(pa.Table.from_pylist(rows), workload[0])
    workload[1].write_text(
        json.dumps({"workload_sha256": digest(workload[0]), "rows": 3})
    )
    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        renderer = Renderer(workload[2])
        calls = []
        images_by_call = {}
        for _ in range(2):
            images = next_images(conn, renderer, size=1)
            call = reserve(conn, images, RESERVE_NANO * 4)
            with conn:
                conn.execute(
                    "UPDATE calls SET status='failed',error=? WHERE id=?",
                    (
                        json.dumps({"type": "APITimeoutError", "provider_body": None}),
                        call,
                    ),
                )
            calls.append(call)
            images_by_call[call] = images
        renderer.close()
        seen = []

        def transport(images):
            seen.append(images)
            return response(images)

        result = retry_failed_requests(
            conn,
            transport,
            RESERVE_NANO * 4,
            calls,
            "Inspected receipt-less timeouts",
            concurrency=2,
        )
        assert {next(iter(images)) for images in seen} == {
            next(iter(images)) for images in images_by_call.values()
        }
        assert result["retried_calls"] == calls
        assert result["call_states"] == {"complete": 2}
        assert result["archived_failed_attempts"] == 2
        assert exposure(conn) == 2 * RESERVE_NANO + sum(
            usage_cost(response(images)) for images in images_by_call.values()
        )


def test_manual_retry_checks_combined_budget_before_mutation(workload):
    with journal(workload[2] / "job.sqlite") as conn:
        call, _, error = server_failed_call(conn, workload)
        with pytest.raises(PermissionError, match="budget"):
            retry_failed_request(
                conn,
                lambda _: pytest.fail("Exceeded budget"),
                RESERVE_NANO * 2 - 1,
                call,
                "Inspected server failure",
            )
        assert conn.execute("SELECT error FROM calls").fetchone()[0] == error
        assert conn.execute("SELECT count(*) FROM failed_attempts").fetchone()[0] == 0
        assert exposure(conn) == RESERVE_NANO


def test_third_attempt_requires_explicit_limit_and_preserves_both_failures(workload):
    class InternalServerError(Exception):
        body: ClassVar = {"message": "server unavailable"}

    def fail(_):
        raise InternalServerError

    with journal(workload[2] / "job.sqlite") as conn:
        call, _, _ = server_failed_call(conn, workload)
        result = retry_failed_request(
            conn, fail, RESERVE_NANO * 3, call, "Inspected first failure"
        )
        assert result["call_states"] == {"failed": 1}
        assert exposure(conn) == RESERVE_NANO * 2
        with pytest.raises(ValueError, match="attempt limit"):
            retry_failed_request(
                conn,
                lambda _: pytest.fail("Retried twice"),
                RESERVE_NANO * 3,
                call,
                "Second inspection",
            )
        result = retry_failed_request(
            conn,
            lambda images: response(images),
            RESERVE_NANO * 3,
            call,
            "User authorized a third submission",
            maximum_attempts=3,
        )
        assert result["call_states"] == {"complete": 1}
        assert result["archived_failed_attempts"] == 2
        assert exposure(conn) == 2 * RESERVE_NANO + usage_cost(response({"C000": 1}))


def test_abandon_preserves_failures_and_skips_the_exact_crop(workload):
    class APITimeoutError(Exception):
        body = None

    with journal(workload[2] / "job.sqlite") as conn:
        call, _, _ = server_failed_call(conn, workload, "APITimeoutError")
        retry_failed_request(
            conn,
            lambda _: (_ for _ in ()).throw(APITimeoutError()),
            RESERVE_NANO * 3,
            call,
            "Inspected first timeout",
        )
        result = abandon_failed_call(conn, call, "Three attempts exhausted")
        assert result["abandoned_call"] == call
        assert result["call_states"] == {"abandoned": 1}
        assert result["archived_failed_attempts"] == 2
        assert result["total_request_attempts"] == 2
        assert exposure(conn) == 2 * RESERVE_NANO
        renderer = Renderer(workload[2])
        assert list(selected_images(conn, renderer, ["a.pdf:1:1", "b.pdf:1:1"])) == []
        renderer.close()


def test_manual_retry_rechecks_source_before_reserving_again(workload):
    with journal(workload[2] / "job.sqlite") as conn:
        call, _, _ = server_failed_call(conn, workload)
        (workload[2] / "a.pdf").write_bytes(b"changed source")
        with pytest.raises(ValueError, match="PDF bytes"):
            retry_failed_request(
                conn,
                lambda _: pytest.fail("Sent changed source"),
                RESERVE_NANO * 2,
                call,
                "Inspected server failure",
            )
        assert conn.execute("SELECT count(*) FROM failed_attempts").fetchone()[0] == 0


def test_selected_batch_limits_records_and_resumes_without_repaying(workload):
    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        result = execute(
            conn,
            lambda images: response(images),
            RESERVE_NANO * 2,
            10,
            selected_cards=["b.pdf:1:1"],
        )
        assert result["new_requests"] == 1
        assert result["indexed_cards"] == 1
        assert (
            conn.execute(
                "SELECT crop_key FROM cards WHERE entry_key='a.pdf:1:1'"
            ).fetchone()[0]
            is None
        )
        result = execute(
            conn,
            lambda _: pytest.fail("Paid for identical completed crop"),
            RESERVE_NANO * 2,
            10,
            selected_cards=["a.pdf:1:1", "b.pdf:1:1"],
        )
        assert result["new_requests"] == 0
        assert result["indexed_cards"] == 2
        assert export_readings(conn, workload[2] / "readings.parquet")["rows"] == 3


@pytest.mark.parametrize(
    "entries", [[], ["a.pdf:1:1", "absent"], ["a.pdf:1:1", "a.pdf:1:1"]]
)
def test_bad_selection_is_rejected_before_any_paid_call(workload, entries):
    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        with pytest.raises(ValueError, match=r"Selection|Selected record"):
            execute(
                conn,
                lambda _: pytest.fail("Sent invalid selection"),
                RESERVE_NANO * 2,
                10,
                selected_cards=entries,
            )
        assert exposure(conn) == 0
        assert checkpoint_summary(conn)["indexed_cards"] == 0


@pytest.mark.parametrize("height", [66, 73, 76, 77, 79, 82])
def test_crop_uses_validated_card_height(height):
    box = crop_box([10, 20, 190, 20 + height])
    assert box[:3] == (11, 37, 144)
    assert box[3] == (76 if height <= 71 else 10 + height)


@pytest.mark.parametrize(
    "box", [[0, 0, 180, 90], [0, 0, 100, 66], [0, 0, float("nan"), 66]]
)
def test_unvalidated_crop_geometry_is_rejected(box):
    with pytest.raises(ValueError, match=r"card geometry|card rectangle"):
        crop_box(box)


def test_pixel_dimensions_and_ink_are_part_of_cache_key():
    assert crop_digest(2, 3, b"abcdef") != crop_digest(3, 2, b"abcdef")
    assert crop_digest(2, 3, b"abcdef") != crop_digest(2, 3, b"abcdeg")


def test_prepare_resume_preserves_both_record_identities_and_field_masks(workload):
    with journal(workload[2] / "job.sqlite") as conn:
        assert prepared(conn, workload) == {"cards": 2, "missing_fields": 3}
        assert prepared(conn, workload) == {"cards": 2, "missing_fields": 3}
        rows = conn.execute(
            "SELECT entry_key,missing FROM cards ORDER BY id"
        ).fetchall()
        assert [tuple(r) for r in rows] == [("a.pdf:1:1", 3), ("b.pdf:1:1", 2)]


def test_changed_workload_hash_is_rejected_before_preparation(workload):
    path, manifest, root = workload
    path.write_bytes(path.read_bytes() + b"changed")
    with journal(root / "job.sqlite") as conn, pytest.raises(ValueError, match="hash"):
        prepare(conn, path, manifest, root)


def test_duplicate_pixel_reuse_keeps_every_missing_record_field(workload):
    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        submissions = []

        def transport(images):
            submissions.append(list(images))
            return response(images)

        execute(conn, transport, RESERVE_NANO * 2, 2, concurrency=1)
        assert submissions == [["C000"]]
        assert checkpoint_summary(conn)["indexed_cards"] == 2
        assert checkpoint_summary(conn)["unique_crops"] == 1
        output = workload[2] / "readings.parquet"
        assert export_readings(conn, output)["rows"] == 3
        rows = pq.read_table(output).to_pylist()
        assert {(r["entry_event_key"], r["role"]) for r in rows} == {
            ("a.pdf:1:1", "own"),
            ("a.pdf:1:1", "relative"),
            ("b.pdf:1:1", "relative"),
        }
        assert all(r["ocr_name"] for r in rows)
        execute(conn, transport, RESERVE_NANO * 2, 2, concurrency=1)
        assert len(submissions) == 1


def test_reserved_but_unreceived_call_is_not_repeated(workload):
    path = workload[2] / "job.sqlite"
    with journal(path) as conn:
        prepared(conn, workload)
        renderer = Renderer(workload[2])
        images = next_images(conn, renderer)
        renderer.close()
        reserve(conn, images, RESERVE_NANO)
    with journal(path) as conn:
        with pytest.raises(ValueError, match="Unresolved prior request"):
            execute(
                conn,
                lambda _: pytest.fail("Repeated an unresolved paid call"),
                RESERVE_NANO * 2,
                1,
            )
        assert exposure(conn) == RESERVE_NANO


def test_stored_receipt_resumes_without_a_new_paid_call(workload):
    path = workload[2] / "job.sqlite"
    with journal(path) as conn:
        prepared(conn, workload)
        renderer = Renderer(workload[2])
        images = next_images(conn, renderer)
        renderer.close()
        call = reserve(conn, images, RESERVE_NANO)
        receive(conn, call, response(images), 0.2)
    with journal(path) as conn:
        execute(
            conn,
            lambda _: pytest.fail("Receipt was unnecessarily resubmitted"),
            RESERVE_NANO,
            1,
        )
        assert checkpoint_summary(conn)["call_states"] == {"complete": 1}
        assert exposure(conn) == usage_cost(response(["C000"]))


def test_budget_check_has_no_partial_submission_side_effect(workload):
    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        renderer = Renderer(workload[2])
        images = next_images(conn, renderer)
        renderer.close()
        with pytest.raises(ValueError, match="budget"):
            reserve(conn, images, RESERVE_NANO - 1)
        assert checkpoint_summary(conn)["call_states"] == {}
        assert checkpoint_summary(conn)["crop_states"] == {"new": 1}


def test_changed_pdf_is_rejected_before_paid_submission(workload):
    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        (workload[2] / "a.pdf").write_bytes(b"changed")
        with pytest.raises(ValueError, match="PDF bytes"):
            execute(
                conn, lambda _: pytest.fail("Submitted changed PDF"), RESERVE_NANO, 1
            )
        assert exposure(conn) == 0


def test_duplicate_or_missing_response_ids_cannot_shift_record_assignments():
    raw = response(["C000"])
    raw["output"][0]["content"][0]["text"] = '{"C000":[],"C000":[]}'
    with pytest.raises(ValueError, match="Duplicate"):
        parse_response(raw, ["C000"])
    with pytest.raises(ValueError, match="IDs"):
        parse_response(response(["C000"]), ["C000", "C001"])


def test_invalid_script_is_retained_but_not_accepted():
    raw = response(["X"], ["AGE 42", "علی خان", "father", None, None])
    parsed = parse_response(raw, ["X"])["X"]
    assert parsed["own"]["raw"] == "AGE 42"
    assert parsed["own"]["accepted"] is None
    assert parsed["own"]["issue"] == "missing_or_invalid_script"
    assert parsed["relative"]["accepted"] == "علی خان"


def test_invalid_response_is_quarantined_and_charged(workload):
    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        execute(conn, lambda _: response(["wrong-id"]), RESERVE_NANO, 1)
        assert checkpoint_summary(conn)["call_states"] == {"invalid": 1}
        assert exposure(conn) > 0
        with pytest.raises(ValueError, match="finish"):
            export_readings(conn, workload[2] / "invalid.parquet")


def test_usage_accounts_for_cached_input_and_reasoning_output():
    assert usage_cost(response(["X"])) == 90 * 100 + 10 * 2 + 50 * 200
    assert usage_cost({}) is None
    invalid = response(["X"])
    invalid["usage"]["input_tokens_details"]["cached_tokens"] = 101
    assert usage_cost(invalid) is None


def test_budget_requires_explicit_approval_and_matches_the_ledger(workload):
    root = workload[2]
    ledger = root / "ledger.json"
    ledger.write_text(
        json.dumps({"model": MODEL, "authorized_total_cap_usd": 1, "calls": []})
    )
    auth = root / "authorization.json"
    data = {
        "status": "pending",
        "model": MODEL,
        "revision": REVISION,
        "workload_sha256": digest(workload[0]),
        "maximum_total_usd": 1,
        "ledger_path": str(ledger),
    }
    auth.write_text(json.dumps(data))
    with journal(root / "job.sqlite") as conn:
        prepared(conn, workload)
        with pytest.raises(PermissionError, match="not been approved"):
            claim_budget(conn, auth)
        assert json.loads(ledger.read_text())["calls"] == []
        data.update(status="approved", maximum_total_usd=2)
        auth.write_text(json.dumps(data))
        with pytest.raises(PermissionError, match="cap"):
            claim_budget(conn, auth)
        data["maximum_total_usd"] = 1
        auth.write_text(json.dumps(data))
        budget = claim_budget(conn, auth)
        assert budget == 1_000_000_000
        assert claim_budget(conn, auth) == budget
        assert len(json.loads(ledger.read_text())["calls"]) == 1
        execute(conn, lambda images: response(images), budget, 1)
        settle(conn)
        saved = json.loads(ledger.read_text())
        assert saved["calls"][0]["status"] == "completed"
        assert saved["accounted_exposure_usd"] < 0.001


def test_explicit_job_closure_releases_unused_allowance(workload):
    root = workload[2]
    ledger = root / "ledger.json"
    ledger.write_text(
        json.dumps({"model": MODEL, "authorized_total_cap_usd": 1, "calls": []})
    )
    auth = root / "authorization.json"
    auth.write_text(
        json.dumps(
            {
                "status": "approved",
                "model": MODEL,
                "revision": REVISION,
                "workload_sha256": digest(workload[0]),
                "maximum_total_usd": 1,
                "ledger_path": str(ledger),
            }
        )
    )
    with journal(root / "job.sqlite") as conn:
        prepared(conn, workload)
        claim_budget(conn, auth)
        settle(conn, close=True, close_reason="calibrated sample complete")
    saved = json.loads(ledger.read_text())
    assert saved["calls"][0]["status"] == "closed"
    assert saved["calls"][0]["estimated_cost_usd"] == 0
    assert saved["calls"][0]["unattempted_or_unselected_cards"] > 0
    assert saved["accounted_exposure_usd"] == 0


def test_replayed_preparation_detects_changed_card_fields(workload):
    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        with conn:
            conn.execute("UPDATE cards SET page=2 WHERE id=1")
        with pytest.raises(ValueError, match="fields differ"):
            prepared(conn, workload)


def test_missing_usage_keeps_reservation_and_prevents_further_submission(workload):
    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)

        def transport(images):
            raw = response(images)
            raw.pop("usage")
            return raw

        result = execute(conn, transport, RESERVE_NANO * 2, 1)
        assert result["stop_reason"] == "missing_usage"
        assert exposure(conn) == RESERVE_NANO
        with pytest.raises(ValueError, match="Unresolved"):
            execute(
                conn,
                lambda _: pytest.fail("Ignored missing usage"),
                RESERVE_NANO * 2,
                1,
            )


def test_source_failure_drains_already_submitted_receipts(workload, monkeypatch):
    from model_training.prep_er_data import jk_urdu_ocr_run

    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        renderer = Renderer(workload[2])
        images = next_images(conn, renderer)
        renderer.close()
        invocations = []

        def next_batch(*_):
            invocations.append(True)
            if len(invocations) == 1:
                return images
            raise ValueError("next source changed")

        monkeypatch.setattr(jk_urdu_ocr_run, "next_images", next_batch)
        with pytest.raises(ValueError, match="next source changed"):
            execute(
                conn, lambda batch: response(batch), RESERVE_NANO * 3, 2, concurrency=2
            )
        assert checkpoint_summary(conn)["call_states"] == {"complete": 1}
        assert conn.execute("SELECT response FROM calls").fetchone()[0]


def test_one_authorization_cannot_fund_two_independent_journals(workload):
    root = workload[2]
    ledger = root / "ledger.json"
    ledger.write_text(
        json.dumps({"model": MODEL, "authorized_total_cap_usd": 1, "calls": []})
    )
    auth = root / "authorization.json"
    auth.write_text(
        json.dumps(
            {
                "status": "approved",
                "model": MODEL,
                "revision": REVISION,
                "workload_sha256": digest(workload[0]),
                "maximum_total_usd": 1,
                "ledger_path": str(ledger),
            }
        )
    )
    with journal(root / "first.sqlite") as first:
        prepared(first, workload)
        claim_budget(first, auth)
    with journal(root / "second.sqlite") as second:
        prepared(second, workload)
        with pytest.raises(ValueError, match="different authorization"):
            claim_budget(second, auth)
    assert len(json.loads(ledger.read_text())["calls"]) == 1


def test_alias_path_cannot_bypass_the_active_journal_lock(workload):
    root = workload[2]
    original = root / "job.sqlite"
    with journal(original):
        alias = root / "alias.sqlite"
        alias.symlink_to(original)
        with pytest.raises(BlockingIOError), journal(alias):
            pytest.fail("Alias bypassed the active job lock")


def test_changed_prompt_cannot_mix_new_results_into_an_existing_job(
    workload, monkeypatch
):
    from model_training.prep_er_data import jk_urdu_ocr_run

    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        monkeypatch.setattr(jk_urdu_ocr_run, "PROMPT", "different reading policy")
        with pytest.raises(ValueError, match="policy differs"):
            execute(
                conn, lambda _: pytest.fail("Changed prompt submitted"), RESERVE_NANO, 1
            )


def test_completed_slot_refills_before_slow_request_finishes(workload, monkeypatch):
    from model_training.prep_er_data import jk_urdu_ocr_run

    third_started = threading.Event()
    batches = iter({str(i): (str(i), b"image")} for i in range(3))
    observed = []

    def transport(images):
        item = next(iter(images))
        if item == "0":
            observed.append(third_started.wait(timeout=5))
        elif item == "2":
            third_started.set()
        return response(images)

    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        with conn:
            conn.executemany(
                "INSERT INTO crops(key) VALUES (?)", [(str(i),) for i in range(3)]
            )
        monkeypatch.setattr(
            jk_urdu_ocr_run, "next_images", lambda *_: next(batches, {})
        )
        result = execute(conn, transport, RESERVE_NANO * 3, 3, concurrency=2)
        assert observed == [True]
        assert result["call_states"] == {"complete": 3}
        assert result["new_requests"] == 3


def test_inflight_reserve_is_released_before_budget_stop(workload, monkeypatch):
    from model_training.prep_er_data import jk_urdu_ocr_run

    batches = iter({str(i): (str(i), b"image")} for i in range(2))
    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        with conn:
            conn.executemany("INSERT INTO crops(key) VALUES (?)", [("0",), ("1",)])
        monkeypatch.setattr(
            jk_urdu_ocr_run, "next_images", lambda *_: next(batches, {})
        )
        result = execute(conn, response, RESERVE_NANO + 100000, 2, concurrency=2)
        assert result["call_states"] == {"complete": 2}
        assert result["stop_reason"] == "request_limit"


def test_completed_selected_cards_do_not_rerender_on_resume(workload, monkeypatch):
    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        execute(conn, response, RESERVE_NANO * 2, 1, selected_cards=["b.pdf:1:1"])
        monkeypatch.setattr(
            Renderer, "render", lambda *_: pytest.fail("Rerendered completed source")
        )
        result = execute(
            conn,
            lambda _: pytest.fail("Repaid completed source"),
            RESERVE_NANO * 2,
            1,
            selected_cards=["b.pdf:1:1"],
        )
        assert result["new_requests"] == 0
        assert result["stop_reason"] == "queue_exhausted"


def test_unfinished_selected_source_still_checks_pdf_hash(workload):
    with journal(workload[2] / "job.sqlite") as conn:
        prepared(conn, workload)
        (workload[2] / "b.pdf").write_bytes(b"changed source")
        with pytest.raises(ValueError, match="PDF bytes"):
            execute(
                conn,
                lambda _: pytest.fail("Paid for changed selected source"),
                RESERVE_NANO * 2,
                1,
                selected_cards=["b.pdf:1:1"],
            )
        assert exposure(conn) == 0
