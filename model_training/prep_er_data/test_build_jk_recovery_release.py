"""Cross-script release links fail closed on duplicate or missing identities."""

import json

import duckdb
import pytest

from model_training.prep_er_data.build_jk_recovery_release import (
    urdu_handoff_prefix,
    validate_link_contract,
)


@pytest.mark.parametrize(
    "revision", ["jk-urdu-2018-calibrated", "jk-urdu-2018-demographics"]
)
def test_urdu_prefix_comes_from_an_explicit_supported_revision(tmp_path, revision):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"source_revision": revision}))
    assert urdu_handoff_prefix(path) == revision + ":"


def test_unknown_urdu_revision_is_rejected(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"source_revision": "invented"}))
    with pytest.raises(ValueError, match="Unsupported"):
        urdu_handoff_prefix(path)


def connection_with_links(links: list[tuple[str, str]]) -> duckdb.DuckDBPyConnection:
    """Return an in-memory connection with minimal handoff and link tables."""
    connection = duckdb.connect()
    connection.execute("CREATE TABLE hindi(entry_event_key VARCHAR)")
    connection.execute("CREATE TABLE urdu(entry_event_key VARCHAR)")
    connection.execute(
        "CREATE TABLE links(hindi_entry_event_key VARCHAR,urdu_entry_event_key VARCHAR)"
    )
    connection.executemany("INSERT INTO hindi VALUES (?)", [("h1",), ("h2",)])
    connection.executemany("INSERT INTO urdu VALUES (?)", [("u1",), ("u2",)])
    connection.executemany("INSERT INTO links VALUES (?,?)", links)
    return connection


def test_link_contract_accepts_exact_one_to_one_join() -> None:
    connection = connection_with_links([("h1", "u1"), ("h2", "u2")])
    validate_link_contract(connection, 2)


@pytest.mark.parametrize(
    ("links", "match"),
    [
        ([("h1", "u1"), ("h1", "u2")], "not one-to-one"),
        ([("h1", "u1"), ("h2", "missing")], "do not join exactly once"),
    ],
)
def test_link_contract_rejects_duplicate_or_missing_identity(
    links: list[tuple[str, str]], match: str
) -> None:
    connection = connection_with_links(links)
    with pytest.raises(ValueError, match=match):
        validate_link_contract(connection, 2)


def test_duplicate_handoff_cannot_mask_a_missing_link() -> None:
    connection = connection_with_links([("h1", "u1"), ("h2", "u2")])
    connection.execute(
        "UPDATE hindi SET entry_event_key='h1' WHERE entry_event_key='h2'"
    )
    with pytest.raises(ValueError, match="handoff keys"):
        validate_link_contract(connection, 2)


@pytest.mark.parametrize("table", ["hindi", "urdu"])
@pytest.mark.parametrize("key", [None, "", "h1"])
def test_invalid_unlinked_handoff_key_is_rejected(table: str, key: str | None) -> None:
    connection = connection_with_links([("h1", "u1")])
    if key == "h1" and table == "urdu":
        key = "u1"
    query = {
        "hindi": "INSERT INTO hindi VALUES (?)",
        "urdu": "INSERT INTO urdu VALUES (?)",
    }[table]
    connection.execute(query, [key])
    with pytest.raises(ValueError, match="handoff keys"):
        validate_link_contract(connection, 1)
