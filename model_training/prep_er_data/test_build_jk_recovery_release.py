"""Cross-script release links fail closed on duplicate or missing identities."""

import duckdb
import pytest

from model_training.prep_er_data.build_jk_recovery_release import (
    validate_link_contract,
)


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
