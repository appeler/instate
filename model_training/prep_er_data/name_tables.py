"""Phase 1 for instate: per-state English voter-name count tables.

Aggregates a roll to ``(english_name, n_times)`` and writes ``<out_dir>/names_<state>.csv.gz``
(default out_dir is instate's ``data/``). Three romanization paths, one subcommand each:

  corpus         --state X            eroll corpus word-map (exact LLM transliterations)
  english        --roll F --lang X    roll already in English (CSV/parquet, --where filter)
  lstm           --roll F --lang X    indicate's local Hindi/Punjabi model (--script)
  devanagari-pdf --roll G --lang X    Devanagari roll from PDF text: repair, corpus, model
  merge          --lang X --inputs..  sum several names_<slug> tables (e.g. jk English+Hindi)

Per-state sources, commands and coverage: SOURCES.md next to this file.

Scale: duckdb collapses the roll to unique names first (100Ms of rows -> millions), so we only
transliterate uniques, then re-sum counts by the romanized name. Build-tooling only -- imports
``eroll`` (Python 3.13); run in an env where eroll is installed, e.g. eroll's own venv:

    ../../../eroll_transliteration/.venv/bin/python name_tables.py corpus --state west_bengal
"""

import csv
import gzip
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import click
import duckdb
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "instate"
DEFAULT_OUT = PROJECT_ROOT / "data"

# indicate ships trained LSTM transliterators for these scripts only; (lo, hi) = native block.
LSTM_SCRIPTS = {"hindi": ("ऀ", "ॿ"), "punjabi": ("਀", "੿")}


def to_ascii(s: str) -> str:
    """Lowercase ASCII-letters-and-spaces only: strip diacritics (ī->i), drop digits/punct.

    Cleans the rare LSTM glitch (e.g. ``0lī`` -> ``li``) and normalizes any stray chars.
    """
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = "".join(c if (c.isascii() and c.isalpha()) or c == " " else " " for c in s)
    return " ".join(s.split()).strip().lower()


def _group_names(roll_path: str | Path, name_col: str) -> duckdb.DuckDBPyConnection:
    """Return a duckdb relation of ``(name, n)`` grouped over a roll's non-empty name col."""
    con = duckdb.connect()
    return con.execute(
        f'SELECT "{name_col}" AS nm, COUNT(*) AS n '
        f"FROM read_csv(?, header = true, all_varchar = true, ignore_errors = true) "
        f'WHERE "{name_col}" IS NOT NULL AND "{name_col}" <> \'\' '
        f'GROUP BY "{name_col}"',
        [str(roll_path)],
    )


def name_counts_via_corpus(
    roll_path, *, name_col, native_run, word_map, batch=100_000
) -> tuple[Counter, dict[str, int]]:
    """Romanize each name via the corpus word-map; drop names with residual native script."""
    rel = _group_names(roll_path, name_col)

    def romanize(text: str) -> str:
        return native_run.sub(lambda m: word_map.get(m.group(0), m.group(0)), text)

    counts: Counter = Counter()
    total = residual = 0
    while rows := rel.fetchmany(batch):
        for nm, n in rows:
            total += n
            sub = romanize(nm)
            if native_run.search(
                sub
            ):  # a token wasn't in the corpus -> not fully English
                residual += n
                continue
            eng = to_ascii(sub)
            if eng:
                counts[eng] += n
    return counts, {"total_voters": total, "residual_voters": residual}


def name_counts_via_english(
    roll_path, *, name_col, batch=100_000
) -> tuple[Counter, dict]:
    """Roll is already English -> just clean + aggregate."""
    rel = _group_names(roll_path, name_col)
    counts: Counter = Counter()
    total = dropped = 0
    while rows := rel.fetchmany(batch):
        for nm, n in rows:
            total += n
            eng = to_ascii(nm)
            if eng:
                counts[eng] += n
            else:
                dropped += n
    return counts, {"total_voters": total, "residual_voters": dropped}


def _source_sql(roll_path) -> str:
    """duckdb FROM-clause for a roll: parquet by extension, otherwise a lenient CSV read."""
    p = str(roll_path)
    if p.endswith(".parquet"):
        return f"read_parquet('{p}')"
    # union_by_name: a glob of per-part CSVs (the J&K Hindi roll) varies in columns
    return (
        f"read_csv('{p}', header = true, all_varchar = true, ignore_errors = true, "
        "union_by_name = true)"
    )


def _coalesce_sql(cols: str) -> str:
    """SQL for a comma-separated column list: the first non-empty value wins.

    UT rolls (Daman, Dadra) split the relation into father's/husband's/mother's name columns.
    """
    parts = [f"NULLIF(\"{c.strip()}\", '')" for c in cols.split(",")]
    return parts[0] if len(parts) == 1 else f"COALESCE({', '.join(parts)})"


def _group2(roll_path, voter_col, father_col, where: str | None = None):
    """duckdb relation of ``(voter, father, n)`` grouped over both name columns."""
    con = duckdb.connect()
    extra = f" AND ({where})" if where else ""
    return con.execute(
        f'SELECT "{voter_col}" AS v, {_coalesce_sql(father_col)} AS f, COUNT(*) AS n '
        f"FROM {_source_sql(roll_path)} "
        f'WHERE "{voter_col}" IS NOT NULL AND "{voter_col}" <> \'\'{extra} '
        "GROUP BY 1, 2"
    )


def name_counts2_corpus(
    roll_path,
    *,
    voter_col,
    father_col,
    native_run,
    word_map,
    batch=100_000,
    where=None,
):
    """Two-column: romanize voter (strict) + father/husband (best-effort) via the corpus."""
    rel = _group2(roll_path, voter_col, father_col, where)

    def sub(t):
        return native_run.sub(lambda m: word_map.get(m.group(0), m.group(0)), t or "")

    counts: Counter = Counter()
    total = dropped = 0
    while rows := rel.fetchmany(batch):
        for v, f, n in rows:
            total += n
            vs = sub(v)
            if native_run.search(vs):  # voter must be fully English
                dropped += n
                continue
            ve = to_ascii(vs)
            if not ve:
                dropped += n
                continue
            fe = (
                to_ascii(sub(f)) if f else ""
            )  # father best-effort (to_ascii drops residual)
            counts[(ve, fe)] += n
    return counts, {"total_voters": total, "residual_voters": dropped}


def name_counts2_english(
    roll_path, *, voter_col, father_col, batch=100_000, where=None
):
    """Two-column for an already-English roll."""
    rel = _group2(roll_path, voter_col, father_col, where)
    counts: Counter = Counter()
    total = dropped = 0
    while rows := rel.fetchmany(batch):
        for v, f, n in rows:
            total += n
            ve = to_ascii(v)
            if not ve:
                dropped += n
                continue
            counts[(ve, to_ascii(f) if f else "")] += n
    return counts, {"total_voters": total, "residual_voters": dropped}


def write_name_table2(
    counts, out_path, header=("english_name", "father_husband_name", "n_times")
) -> int:
    """Write ``english_name,father_husband_name,n_times`` sorted by count desc (atomic)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        ((v, f, c) for (v, f), c in counts.items()), key=lambda x: (-x[2], x[0], x[1])
    )
    tmp = out_path.with_name(out_path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(header))
        w.writerows(rows)
    os.replace(tmp, out_path)
    return len(rows)


def _lstm_word_map(tokens: list[str], script: str, chunk: int = 4000) -> dict[str, str]:
    """Romanize unique native tokens with indicate's local model (greedy beam)."""
    from indicate import transliterate_batch

    word_map: dict[str, str] = {}
    for i in tqdm(range(0, len(tokens), chunk), desc=f"{script} lstm tokens"):
        part = tokens[i : i + chunk]
        out = transliterate_batch(part, source=script, beam=1)
        for tok, raw in zip(part, out, strict=False):
            word_map[tok] = to_ascii(
                raw if isinstance(raw, str) else (raw[0] if raw else "")
            )
    return word_map


def name_counts_via_lstm(
    roll_path, *, name_col, script, chunk=4000
) -> tuple[Counter, dict]:
    """Romanize via indicate's Hindi/Punjabi LSTM at the TOKEN level (memory-bounded).

    Transliterating millions of full names directly OOMs (beam search over huge batches),
    so we LSTM only the unique native TOKENS (~5x fewer, single words the model is built
    for), build a word-map, then substitute it onto every name -- the fast corpus path.
    """
    lo, hi = LSTM_SCRIPTS[script]
    native_run = re.compile(f"[{lo}-{hi}]+")

    rel = _group_names(roll_path, name_col)
    rows = rel.fetchall()
    tokens = sorted({t for nm, _ in rows for t in native_run.findall(nm)})
    word_map = _lstm_word_map(tokens, script, chunk)

    counts: Counter = Counter()
    total = dropped = 0
    for nm, n in rows:
        total += n
        eng = to_ascii(native_run.sub(lambda m: word_map.get(m.group(0), " "), nm))
        if eng:
            counts[eng] += n
        else:
            dropped += n
    return counts, {"total_voters": total, "residual_voters": dropped}


def write_name_table(
    counts: Counter, out_path: Path, header=("english_name", "n_times")
) -> int:
    """Write ``english_name,n_times`` sorted by count desc (atomic). Returns row count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    tmp = out_path.with_name(out_path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(header))
        w.writerows(pairs)
    os.replace(tmp, out_path)
    return len(pairs)


def _finish(lang: str, counts: Counter, stats: dict, out_dir: str | None) -> None:
    out = (Path(out_dir) if out_dir else DEFAULT_OUT) / f"names_{lang}.csv.gz"
    n = write_name_table(counts, out)
    tot, drop = stats["total_voters"], stats["residual_voters"]
    click.echo(
        f"[{lang}] {n:,} unique english names -> {out}\n"
        f"  voters {tot:,}; dropped {drop:,} ({100 * drop / max(1, tot):.1f}%); kept {tot - drop:,}"
    )


def _finish2(lang: str, counts: Counter, stats: dict, out_dir: str | None) -> None:
    out = (Path(out_dir) if out_dir else DEFAULT_OUT) / f"names_{lang}.csv.gz"
    n = write_name_table2(counts, out)
    tot, drop = stats["total_voters"], stats["residual_voters"]
    click.echo(
        f"[{lang}] {n:,} (voter, father/husband) pairs -> {out}\n"
        f"  voters {tot:,}; dropped {drop:,} ({100 * drop / max(1, tot):.1f}%)"
    )


@click.group()
def cli() -> None:
    """Build per-state (voter, father/husband) name count tables for instate Phase 1."""


@cli.command()
@click.option("--state", required=True)
@click.option("--name-col", default="elector_name", show_default=True)
@click.option("--father-col", default="father_or_husband_name", show_default=True)
@click.option(
    "--roll", default=None, help="Roll path (default: the state's input_path)."
)
@click.option("--out-dir", default=None)
def corpus(state, name_col, father_col, roll, out_dir):
    """Romanize voter + father/husband via the state's eroll corpus word-map."""
    from eroll.states import STATES

    if state not in STATES:
        raise click.BadParameter(f"unknown state: {state}", param_hint="--state")
    cfg = STATES[state]
    word_map = _load_word_map(cfg.corpus_csv)
    click.echo(f"[{cfg.name}] {len(word_map):,} word-map entries; aggregating ...")
    counts, stats = name_counts2_corpus(
        Path(roll) if roll else cfg.input_path,
        voter_col=name_col,
        father_col=father_col,
        native_run=cfg.native_run,
        word_map=word_map,
    )
    _finish2(cfg.name, counts, stats, out_dir)


@cli.command()
@click.option("--roll", required=True)
@click.option("--lang", required=True, help="State name for names_<lang>.csv.gz.")
@click.option("--name-col", default="elector_name", show_default=True)
@click.option("--father-col", default="father_or_husband_name", show_default=True)
@click.option(
    "--where", default=None, help="SQL row filter, e.g. roll_section = 'main'."
)
@click.option("--out-dir", default=None)
def english(roll, lang, name_col, father_col, where, out_dir):
    """Roll already in English (CSV or parquet) -> aggregate (voter, father/husband)."""
    click.echo(f"[{lang}] aggregating English roll {Path(roll).name} ...")
    counts, stats = name_counts2_english(
        roll, voter_col=name_col, father_col=father_col, where=where
    )
    _finish2(lang, counts, stats, out_dir)


@cli.command()
@click.option("--roll", required=True)
@click.option("--lang", required=True, help="State name for names_<lang>.csv.gz.")
@click.option("--script", required=True, type=click.Choice(sorted(LSTM_SCRIPTS)))
@click.option("--name-col", default="elector_name", show_default=True)
@click.option("--out-dir", default=None)
def lstm(roll, lang, script, name_col, out_dir):
    """Romanize native names with indicate's trained Hindi/Punjabi LSTM."""
    click.echo(f"[{lang}] {script} LSTM over {Path(roll).name} ...")
    counts, stats = name_counts_via_lstm(roll, name_col=name_col, script=script)
    _finish(lang, counts, stats, out_dir)


# ---------------------------------------------------------------------------
# Devanagari rolls parsed from PDF text (the J&K 2018 Hindi roll): repair the
# extraction artifacts, romanize through the Hindi corpus, LSTM the residue.
# ---------------------------------------------------------------------------

DEVANAGARI = re.compile("[ऀ-ॿ]+")
_CONS = "[क-ह]"
_MATRA = "[ा-ौ]"
_STUB = re.compile(f"^{_CONS}{_MATRA}$")
_LONE = re.compile(f"^{_CONS}$")
_NUKTA = "़"


def strip_nukta(tok: str) -> str:
    return unicodedata.normalize(
        "NFC", unicodedata.normalize("NFD", tok).replace(_NUKTA, "")
    )


def repair_devanagari_pdf(name: str) -> list[str]:
    """Undo the artifacts PDF text extraction leaves in Devanagari names; return tokens.

    Seen in the J&K 2018 Hindi roll: the i-matra printed before its consonant (िसंह ->
    सिंह); a consonant+matra stub split off the front of a word (कु मार -> कुमार, राके श ->
    राकेश); reph printed after the syllable it precedes (शमार् -> शर्मा); conjuncts lost to
    U+FFFD (गु�ता). Damaged tokens are dropped; the caller decides whether a name whose
    LAST token was damaged is still usable (it is not: the surname is gone).
    """
    name = re.sub("ि(\\S)", r"\1ि", name or "")
    raw = [t for t in name.split() if "�" not in t]
    toks: list[str] = []
    for t in raw:
        if toks and _LONE.match(t):
            toks[-1] += t
        elif toks and _STUB.match(toks[-1]) and DEVANAGARI.match(t):
            toks[-1] += t
        else:
            toks.append(t)
    out = []
    for t in toks:
        if t.endswith("र्") and len(t) > 3:
            body = t[:-2]
            m = re.search(f"{_CONS}{_MATRA}?$", body)
            if m:
                t = body[: m.start()] + "र्" + body[m.start() :]
        out.append(strip_nukta(t))
    return out


def _load_word_map_nukta_tolerant(corpus_csv) -> dict[str, str]:
    """Corpus word-map keyed by exact spelling first, then nukta-stripped spelling.

    Exact spellings take precedence so a nukta variant (क़ुमार -> qumar) never shadows the
    plain one (कुमार -> kumar) once the roll's own nukta is stripped for lookup.
    """
    exact = _load_word_map(corpus_csv)
    word_map = dict(exact)
    for src, eng in exact.items():
        word_map.setdefault(strip_nukta(src), eng)
    return word_map


def name_counts2_devanagari_pdf(
    roll_glob, *, voter_col, father_col, word_map, script="hindi"
) -> tuple[Counter, dict]:
    """Two-column counts for a Devanagari roll extracted from PDF text."""
    rel = _group2(roll_glob, voter_col, father_col)
    rows = rel.fetchall()
    total = dropped = 0
    cleaned: list[tuple[list[str], list[str], int]] = []
    for v, f, n in rows:
        total += n
        raw = (v or "").split()
        if not raw or "�" in raw[-1]:
            dropped += n
            continue
        vt = repair_devanagari_pdf(v)
        if not vt:
            dropped += n
            continue
        cleaned.append((vt, repair_devanagari_pdf(f or ""), n))
    missing = sorted(
        {
            t
            for vt, ft, _ in cleaned
            for t in vt + ft
            if DEVANAGARI.search(t) and t not in word_map
        }
    )
    word_map = dict(word_map)
    word_map.update(_lstm_word_map(missing, script))

    def roman(tokens: list[str]) -> str:
        return to_ascii(
            " ".join(word_map.get(t, t) if DEVANAGARI.search(t) else t for t in tokens)
        )

    counts: Counter = Counter()
    for vt, ft, n in cleaned:
        ve = roman(vt)
        if not ve:
            dropped += n
            continue
        counts[(ve, roman(ft))] += n
    return counts, {"total_voters": total, "residual_voters": dropped}


@cli.command(name="devanagari-pdf")
@click.option("--roll", required=True, help="CSV path or glob (duckdb read_csv).")
@click.option("--lang", required=True, help="State slug for names_<lang>.csv.gz.")
@click.option("--corpus", required=True, help="eroll hindi.csv.gz word-map.")
@click.option("--name-col", default="elector_name", show_default=True)
@click.option("--father-col", default="father_or_husband_name", show_default=True)
@click.option("--out-dir", default=None)
def devanagari_pdf(roll, lang, corpus, name_col, father_col, out_dir):
    """Devanagari roll from PDF text -> repair artifacts, corpus + LSTM romanize."""
    word_map = _load_word_map_nukta_tolerant(corpus)
    click.echo(f"[{lang}] {len(word_map):,} word-map entries; aggregating {roll} ...")
    counts, stats = name_counts2_devanagari_pdf(
        roll, voter_col=name_col, father_col=father_col, word_map=word_map
    )
    _finish2(lang, counts, stats, out_dir)


@cli.command(name="merge")
@click.option("--lang", required=True, help="Output slug for names_<lang>.csv.gz.")
@click.option("--inputs", required=True, multiple=True, help="names_*.csv.gz to sum.")
@click.option("--out-dir", default=None)
def merge(lang, inputs, out_dir):
    """Sum several (voter, father/husband, n_times) tables into one names_<lang> table."""
    counts: Counter = Counter()
    total = 0
    for path in inputs:
        for v, f, n in iter_name_table(path):
            counts[(v, f)] += n
            total += n
    _finish2(lang, counts, {"total_voters": total, "residual_voters": 0}, out_dir)


# ---------------------------------------------------------------------------
# Last-name extraction (Phase 2): resolve a surname per (voter, father/husband)
# row of the names_<state> tables, weighted by n_times, then aggregate per state.
# ---------------------------------------------------------------------------

# Map each names_<slug>.csv.gz to its full state/UT name (the column headers used by
# instate's surname-occurrence state-share product). Covers all 34 tables.
FILE2STATE: dict[str, str] = {
    "andaman": "Andaman and Nicobar Islands",
    "andhra": "Andhra Pradesh",
    "arunachal": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chandigarh": "Chandigarh",
    "dadra": "Dadra and Nagar Haveli",
    "daman": "Daman and Diu",
    "delhi": "Delhi",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal": "Himachal Pradesh",
    "jharkhand": "Jharkhand",
    "jk": "Jammu and Kashmir and Ladakh",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "madhya_pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "puducherry": "Puducherry",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "telugu": "Telangana",
    "tamil_nadu": "Tamil Nadu",
    "tripura": "Tripura",
    "uttar_pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "west_bengal": "West Bengal",
}

# Null / placeholder tokens: never a surname (and never "content").
NULLS: frozenset[str] = frozenset(
    {
        "fnu",
        "lnu",
        "lnf",
        "fnf",
        "nfu",
        "nlu",
        "na",
        "nil",
        "null",
        "none",
        "nan",
        "unknown",
        "unknwn",
        "baby",
        "minor",
    }
)

# Name-prefix particles + honorific titles: stripped from both name token lists before
# tiering (a prefix is never the surname, and removing it lets the shared-token tier match,
# e.g. "she husen" / "she husen" -> "husen"). Honorific *suffixes* (kumar, devi, lal, ...)
# are intentionally NOT here -- they're kept as content because they carry state signal.
PARTICLES: frozenset[str] = frozenset(
    {
        # Islamic name prefixes / patronymics
        "md",
        "mohd",
        "mohmd",
        "mohamad",
        "mohamed",
        "mohammad",
        "mohammed",
        "muhammad",
        "muhammed",
        "mo",
        "sk",
        "sekh",
        "shekh",
        "sheikh",
        "shaikh",
        "shek",
        "she",
        "syed",
        "sayed",
        "sayyed",
        "sayyad",
        "saiyad",
        "saiyed",
        "abdul",  # "servant of": the household tier would otherwise pick it for siblings
        "bin",
        "binte",
        "bint",
        "ibn",
        "abu",
        # honorific titles / relationship prefixes
        "smt",
        "shrimati",
        "shri",
        "sri",
        "mr",
        "mrs",
        "miss",
        "ms",
        "mast",
        "master",
        "kum",
        "km",
        "late",
    }
)


def _resolve_last_name(
    voter: str, father: str, stop: frozenset[str]
) -> tuple[str | None, str]:
    """Resolve a surname from a (voter, father/husband) name pair. Returns (surname, tier).

    Tiers (first match wins): T1 a content token shared by voter & father; T2 a multi-token
    voter's last content token; T3 a single-token voter inherits the father/husband's last
    content token; T4 drop. ``stop`` is the set of tokens that may never be selected (nulls
    [+ singh/kaur if asked]).

    T1 auto-detects name order so the *surname* wins over a shared father's-given-name:
    Indian rolls use either "Given ... Surname" (surname last -- North/Sikkim) or
    "Surname Given FatherGiven" (surname first -- e.g. Maharashtra: ``patil parvati shankar``
    / father ``patil shankar``, where both ``patil`` and ``shankar`` are shared). If the
    voter and father share their *leading* token it's a surname-first roll -> take the first
    token; otherwise take the *last* shared token.
    """
    v = [t for t in voter.split() if t not in PARTICLES]
    f = [t for t in father.split() if t not in PARTICLES] if father else []

    def content(t: str) -> bool:
        return len(t) > 2 and t not in stop and t not in NULLS

    fset = set(f)
    # T1: a content token shared by voter & father.
    if any(t in fset and content(t) for t in v):
        # surname-first convention: voter & father share their leading token.
        if v and f and v[0] == f[0] and content(v[0]):
            return v[0], "T1"
        # surname-last convention: take the last shared content token.
        for t in reversed(v):
            if t in fset and content(t):
                return t, "T1"
    # T2: multi-token voter -> last content token.
    if len(v) >= 2:
        for t in reversed(v):
            if content(t):
                return t, "T2"
    # T3: single-token (or no-content) voter -> father/husband's last content token.
    for t in reversed(f):
        if content(t):
            return t, "T3"
    return None, "DROP"


def iter_name_table(path: str | Path):
    """Yield ``(english_name, father_husband_name, n_times)`` rows from a names_<state> gz."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 3 and row[2].isdigit():
                yield row[0], row[1], int(row[2])


def build_last_names(
    slug: str, in_dir: Path, out_dir: Path, *, extra_stop=frozenset(), singh_stop=False
) -> dict:
    """Resolve surnames for one state's names_<slug> table -> last_names_<slug> (atomic).

    Streams the table, accumulates ``surname -> Σ n_times`` (bounded by #distinct surnames,
    so memory-safe), writes sorted/atomic, and returns coverage + per-tier weight stats.
    """
    src = in_dir / f"names_{slug}.csv.gz"
    stop = (
        NULLS
        | extra_stop
        | (frozenset({"singh", "kaur"}) if singh_stop else frozenset())
    )
    counts: Counter = Counter()
    tier_w = Counter()
    total = kept = 0
    for voter, father, n in iter_name_table(src):
        total += n
        ln, tier = _resolve_last_name(voter, father, stop)
        tier_w[tier] += n
        if ln is not None:
            counts[ln] += n
            kept += n
    out = out_dir / f"last_names_{slug}.csv.gz"
    rows = write_name_table(counts, out, header=("last_name", "n_times"))
    return {
        "slug": slug,
        "out": out,
        "surnames": rows,
        "total": total,
        "kept": kept,
        "tiers": dict(tier_w),
        "top": counts.most_common(30),
    }


def _report_last_names(st: dict) -> None:
    tot, kept = st["total"], st["kept"]
    t = st["tiers"]

    def pct(x):
        return 100 * x / max(1, tot)

    click.echo(
        f"[{st['slug']}] {st['surnames']:,} surnames -> {st['out']}\n"
        f"  weight {tot:,}; kept {kept:,} ({pct(kept):.1f}%); "
        f"dropped {tot - kept:,} ({pct(tot - kept):.1f}%)\n"
        f"  tiers: T1 {pct(t.get('T1', 0)):.1f}%  T2 {pct(t.get('T2', 0)):.1f}%  "
        f"T3 {pct(t.get('T3', 0)):.1f}%  drop {pct(t.get('DROP', 0)):.1f}%\n"
        f"  top: {', '.join(f'{nm}({c:,})' for nm, c in st['top'][:15])}"
    )


@cli.command(name="lastnames")
@click.option("--state", default=None, help="One slug (e.g. bihar); omit with --all.")
@click.option(
    "--all", "all_states", is_flag=True, help="Process every names_<slug> table."
)
@click.option(
    "--in-dir",
    default=None,
    help="Where names_<slug>.csv.gz live (default instate/data).",
)
@click.option(
    "--out-dir", default=None, help="Output dir (default instate/data/last_names)."
)
@click.option("--extra-stop", default="", help="Comma-separated extra stop tokens.")
@click.option(
    "--singh-mode",
    type=click.Choice(["content", "stop"]),
    default="content",
    show_default=True,
    help="Treat singh/kaur as real surnames (content) or honorifics (stop).",
)
def lastnames(state, all_states, in_dir, out_dir, extra_stop, singh_mode):
    """Resolve a last name per (voter, father/husband) row -> last_names_<slug>.csv.gz."""
    indir = Path(in_dir) if in_dir else DEFAULT_OUT
    outdir = Path(out_dir) if out_dir else DEFAULT_OUT / "last_names"
    extra = frozenset(t.strip() for t in extra_stop.split(",") if t.strip())
    singh_stop = singh_mode == "stop"
    if all_states:
        slugs = [s for s in FILE2STATE if (indir / f"names_{s}.csv.gz").exists()]
    elif state:
        slugs = [state]
    else:
        raise click.UsageError("pass --state <slug> or --all")
    for slug in slugs:
        st = build_last_names(
            slug, indir, outdir, extra_stop=extra, singh_stop=singh_stop
        )
        _report_last_names(st)


# ---------------------------------------------------------------------------
# Household tier (T0). Rolls that mix surname-first and surname-last names
# within one part (the Telangana and Andhra English rolls) defeat the
# position tiers: "etham jayamma" and "kavita namala" sit in the same part.
# Electors at one house number in one part are a household, and a content
# token two or more of them share is the surname, whichever end it sits at.
# Reads the electors parquet a roll parser writes (parse_searchable_rolls schema).
# ---------------------------------------------------------------------------


def _content(t: str, stop: frozenset[str]) -> bool:
    return len(t) > 2 and t not in stop and t not in NULLS and t not in PARTICLES


VOWELS = frozenset("aeiouy")


def _spelling_slack(token: str) -> int:
    """Edits a household spelling may differ by and still be the same name.

    A slip in a long token (bayikadi / baikadi, komatiareddy / komatireddy) is a variant,
    not another family. Under four letters: exact only (ram / rao). Four to six: one edit,
    and only a vowel change or an aspiration h (begam / begum, gaud / goud, sing / singh,
    jadav / jadhav), never a consonant (rani / ravi, rajesh / ramesh, kaleem / saleem).
    Seven to nine: one edit of any kind. Ten or more: two.
    """
    return 0 if len(token) < 4 else 1 if len(token) < 10 else 2


def _soft_edit(a: str, b: str) -> bool:
    """True when a and b differ by one vowel change or one aspiration h."""
    from rapidfuzz.distance import Levenshtein

    ops = Levenshtein.editops(a, b)
    if len(ops) != 1:
        return False
    op = ops[0]
    src = a[op.src_pos] if op.tag != "insert" else ""
    dst = b[op.dest_pos] if op.tag != "delete" else ""
    changed = {c for c in (src, dst) if c}
    return changed <= VOWELS or changed == {"h"}


def _same_spelling(a: str, b: str) -> bool:
    """Equal, or within the longer token's edit slack (soft edits only under seven)."""
    if a == b:
        return True
    from rapidfuzz.distance import Levenshtein

    longer = max(a, b, key=len)
    slack = _spelling_slack(longer)
    if slack == 0 or Levenshtein.distance(a, b, score_cutoff=slack) > slack:
        return False
    return len(longer) >= 7 or _soft_edit(a, b)


def household_spellings(spellings: Counter) -> dict[str, str]:
    """Map each spelling seen in a household to the household's majority spelling.

    Greedy: the most frequent spelling anchors a cluster and absorbs every unclustered
    spelling within its edit slack; ties on frequency go to the shorter, then
    alphabetical, spelling so the mapping is deterministic.
    """
    canon: dict[str, str] = {}
    for tok in sorted(spellings, key=lambda t: (-spellings[t], len(t), t)):
        if tok in canon:
            continue
        canon[tok] = tok
        slack = _spelling_slack(tok)
        if not slack:
            continue
        for other in spellings:
            if other not in canon and _same_spelling(tok, other):
                canon[other] = tok
    return canon


def resolve_household(
    members: list[tuple[str, str]], stop: frozenset[str]
) -> list[tuple[str | None, str]]:
    """Resolve surnames for one household; T0 where the evidence agrees, else T1 to T3.

    Candidates for a member are its content tokens that another household member also
    carries, plus the ones its relation name carries. The strongest evidence wins: a
    token both the household and the relation share, then a household-shared token,
    then a relation-shared one; ties go to the rightmost token. Reddy, rao, singh and
    the like are surnames people go by and are never demoted in favour of a rarer
    token. Spellings that differ by a letter or two in a long token count as the same
    token and resolve to the household's majority spelling. Members with no candidate
    fall through to the position tiers.
    """
    raw_tokens = [[t for t in v.split() if _content(t, stop)] for v, _ in members]
    raw_relations = [
        [t for t in f.split() if _content(t, stop)] if f else [] for _, f in members
    ]
    spellings: Counter = Counter()
    for toks in raw_tokens:
        spellings.update(set(toks))
    for toks in raw_relations:
        spellings.update(set(toks))
    canon = household_spellings(spellings) if len(spellings) > 1 else {}
    tokens = [[canon.get(t, t) for t in toks] for toks in raw_tokens]
    tally: Counter = Counter()
    for toks in tokens:
        tally.update(set(toks))
    out: list[tuple[str | None, str]] = []
    for (v, f), toks, rel in zip(members, tokens, raw_relations, strict=True):
        fset = {canon.get(t, t) for t in rel}
        household = [t for t in toks if tally[t] >= 2] if len(members) >= 2 else []
        if household:
            pick = max(
                household,
                key=lambda t: (t in fset, tally[t], toks.index(t)),
            )
            out.append((pick, "T0"))
        else:
            out.append(_resolve_last_name(v, f, stop))
    return out


def iter_cache_households(electors: Path):
    """Yield (part, house, [(voter, relation, n), ...]) from an electors parquet.

    The parquet is the parse_searchable_rolls / parse_unsearchable_rolls elector schema
    (``elector_name, father_or_husband_name, house_no, filename, roll_section, deleted``).
    Active electors only: the mother roll plus the supplement's additions, minus the
    electors stamped deleted.
    """
    con = duckdb.connect()
    rel = con.execute(
        "SELECT filename, lower(trim(coalesce(house_no, ''))) AS house, "
        "       elector_name, coalesce(father_or_husband_name, '') AS relation, "
        "       count(*) AS n "
        f"FROM read_parquet('{electors}') "
        "WHERE coalesce(roll_section, 'main') IN ('main', 'addition') "
        "  AND NOT coalesce(deleted, false) AND elector_name IS NOT NULL "
        "GROUP BY 1, 2, 3, 4 ORDER BY 1, 2"
    )
    key: tuple[str, str] | None = None
    rows: list[tuple[str, str, int]] = []
    while batch := rel.fetchmany(100_000):
        for part, house, v, f, n in batch:
            v, f = to_ascii(v), to_ascii(f)
            if not v:
                continue
            if key != (part, house):
                if key is not None and rows:
                    yield key[0], key[1], rows
                key, rows = (part, house), []
            rows.append((v, f, n))
    if key is not None and rows:
        yield key[0], key[1], rows


def build_last_names_households(
    slug: str, electors: Path, out_dir: Path, *, extra_stop=frozenset()
) -> dict:
    """Household-tier surname resolution over an electors parquet -> last_names_<slug>.

    Also validates the household pick: on electors where the relation name shares a
    content token (the T1 evidence), the household pick agrees with T1 in
    ``agree`` of ``checked`` cases; the disagreements are the honest error bound.
    """
    stop = NULLS | extra_stop
    counts: Counter = Counter()
    tier_w: Counter = Counter()
    ladder: Counter = Counter()
    total = kept = checked = agree = 0
    for _part, house, rows in iter_cache_households(electors):
        # a blank house number is not a household; every row there falls through
        members = [(v, f) for v, f, n in rows for _ in range(n)] if house else []
        resolved = (
            resolve_household(members, stop)
            if house
            else [_resolve_last_name(v, f, stop) for v, f, n in rows for _ in range(n)]
        )
        flat = [(v, f) for v, f, n in rows for _ in range(n)]
        for (v, f), (ln, tier) in zip(flat, resolved, strict=True):
            total += 1
            tier_w[tier] += 1
            if ln is not None:
                counts[ln] += 1
                kept += 1
                # the evidence ladder: how the pick was corroborated and where it sits
                vt = v.split()
                hits = [i for i, t in enumerate(vt) if _same_spelling(t, ln)]
                where = (
                    "inherited"  # T3: taken from the relation name
                    if not hits
                    else "last"
                    if hits[-1] == len(vt) - 1
                    else "first"
                    if hits[0] == 0
                    else "middle"
                )
                fset = set(f.split()) if f else set()
                how = {
                    "T0": "household+relation" if ln in fset else "household",
                    "T1": "relation",
                }.get(tier, "position")
                ladder[(how, where)] += 1
            if tier == "T0":
                # the relation-shared token as an independent read of the same surname
                t1, t1_tier = _resolve_last_name(v, f, stop)
                if t1_tier == "T1":
                    checked += 1
                    agree += _same_spelling(t1, ln)
    out = out_dir / f"last_names_{slug}.csv.gz"
    rows_written = write_name_table(counts, out, header=("last_name", "n_times"))
    return {
        "slug": slug,
        "out": out,
        "surnames": rows_written,
        "total": total,
        "kept": kept,
        "tiers": dict(tier_w),
        "top": counts.most_common(30),
        "checked": checked,
        "agree": agree,
        "ladder": dict(ladder),
    }


@cli.command(name="lastnames-households")
@click.option("--electors", required=True, help="electors.parquet from a roll parser.")
@click.option("--lang", required=True, help="State slug for last_names_<slug>.csv.gz.")
@click.option("--out-dir", default=None)
@click.option("--extra-stop", default="", help="Comma-separated extra stop tokens.")
def lastnames_households(electors, lang, out_dir, extra_stop):
    """Resolve surnames with the household tier over a parsed roll (electors parquet)."""
    outdir = Path(out_dir) if out_dir else DEFAULT_OUT / "last_names"
    extra = frozenset(t.strip() for t in extra_stop.split(",") if t.strip())
    st = build_last_names_households(lang, Path(electors), outdir, extra_stop=extra)
    _report_last_names(st)
    click.echo("  evidence ladder (share of resolved electors):")
    kept = max(1, st["kept"])
    for how in ("household+relation", "household", "relation", "position"):
        row = "  ".join(
            f"{where} {100 * st['ladder'].get((how, where), 0) / kept:5.1f}%"
            for where in ("last", "first", "middle", "inherited")
        )
        click.echo(f"    {how:20s} {row}")
    t0 = st["tiers"].get("T0", 0)
    click.echo(
        f"  household tier T0: {100 * t0 / max(1, st['total']):.1f}% of electors; "
        f"agrees with the relation-shared token in {st['agree']:,}/{st['checked']:,} "
        f"({100 * st['agree'] / max(1, st['checked']):.1f}%) of the electors that have both"
    )


# ---------------------------------------------------------------------------
# Phase 3: merge the 34 last_names tables into surname-level state shares.
# ---------------------------------------------------------------------------

# v1's exact 31-column order (preserved for consumer compatibility); v2 appends the three
# states v1 omitted. ``set`` must equal ``set(FILE2STATE.values())`` (asserted in ln-prop).
V1_STATE_ORDER: tuple[str, ...] = (
    "Andaman and Nicobar Islands",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Dadra and Nagar Haveli",
    "Daman and Diu",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Jharkhand",
    "Jammu and Kashmir and Ladakh",
    "Karnataka",
    "Kerala",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Madhya Pradesh",
    "Nagaland",
    "Odisha",
    "Puducherry",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
)
V2_STATE_ORDER: tuple[str, ...] = V1_STATE_ORDER + (
    "Himachal Pradesh",
    "Tamil Nadu",
    "West Bengal",
)

# Dravidian + Odia scripts append an inherent trailing vowel when romanized (patil->patila,
# pradhan->pradhana) and split internal vowels (kamble->kambale). A variant whose weight is
# concentrated in these states is almost certainly such an artifact, not a distinct surname --
# so the merge is gated on this set (keeps Bengali ``saha`` out of Bihar ``sah``).
ARTIFACT_STATES: frozenset[str] = frozenset(
    {"Karnataka", "Odisha", "Telangana", "Andhra Pradesh"}
)


def _build_remap(
    freqs, art_frac, anchors, *, min_variant=200, art_min=0.6
) -> dict[str, str]:
    """Map Dravidian/Odia romanization variants -> their canonical high-frequency anchor.

    A variant is merged only when (a) its weight is concentrated in ARTIFACT_STATES
    (``art_frac >= art_min`` -- so real surnames elsewhere like Bengali ``saha`` are never
    touched), and (b) deleting ONE character yields a strictly-more-frequent anchor. The
    deletion-only rule captures the trailing/internal inherent-vowel artifact
    (``patila->patil``, ``kambale->kamble``, ``areddy->reddy``) while excluding the
    substitution merges that conflate distinct surnames (``rao->ram``, ``jena->jana``).
    Chains are resolved to a fixpoint.
    """
    remap: dict[str, str] = {}
    for v, f in freqs.items():
        if f < min_variant or len(v) <= 3 or art_frac.get(v, 0.0) < art_min:
            continue
        best, best_f = None, f  # target must be strictly more frequent than the variant
        for i in range(len(v)):
            cand = v[:i] + v[i + 1 :]
            if len(cand) >= 3 and cand in anchors and freqs[cand] > best_f:
                best, best_f = cand, freqs[cand]
        if best:
            remap[v] = best

    def resolve(x: str) -> str:
        seen: set[str] = set()
        while x in remap and x not in seen:
            seen.add(x)
            x = remap[x]
        return x

    return {v: resolve(v) for v in remap}


def _v2_default_out() -> Path:
    return PACKAGE_ROOT / "data" / "instate_unique_ln_state_prop_v2.csv.gz"


@cli.command(name="ln-prop")
@click.option("--in-dir", default=None, help="last_names_<slug>.csv.gz dir.")
@click.option("--out", "out_path", default=None, help="Output v2 csv.gz path.")
@click.option(
    "--anchor-min",
    default=20000,
    show_default=True,
    help="Min national frequency for a canonical anchor.",
)
@click.option(
    "--no-canon", is_flag=True, help="Skip KNN canonicalization of artifact variants."
)
@click.option(
    "--train-out",
    default=None,
    help="Also write canonicalized (last_name,state,n_times) GRU training data here.",
)
@click.option(
    "--min-total",
    default=3,
    show_default=True,
    help="Drop surnames with national total < this (denoise + shrink; v1 used 3).",
)
def ln_prop(in_dir, out_path, anchor_min, no_canon, train_out, min_total):
    """Merge 34 last-name tables into canonicalized, normalized state shares."""
    indir = Path(in_dir) if in_dir else DEFAULT_OUT / "last_names"
    out = Path(out_path) if out_path else _v2_default_out()
    assert set(FILE2STATE.values()) == set(V2_STATE_ORDER), "state name mismatch"
    slugs = [s for s in FILE2STATE if (indir / f"last_names_{s}.csv.gz").exists()]
    click.echo(f"[ln-prop] stacking {len(slugs)} state tables from {indir} ...")

    con = duckdb.connect()
    union = " UNION ALL ".join(
        f"SELECT '{FILE2STATE[s]}' AS state, last_name, n_times "
        f"FROM read_csv('{indir / f'last_names_{s}.csv.gz'}', header = true)"
        for s in slugs
    )
    con.execute(f"CREATE TABLE stacked AS {union}")

    art_list = ",".join(f"'{s}'" for s in ARTIFACT_STATES)
    rows = con.execute(
        "SELECT last_name, SUM(n_times) AS tot, "
        f"COALESCE(SUM(n_times) FILTER (WHERE state IN ({art_list})), 0) AS art "
        "FROM stacked GROUP BY last_name"
    ).fetchall()
    freqs = {ln: tot for ln, tot, _ in rows}
    art_frac = {ln: (art / tot if tot else 0.0) for ln, tot, art in rows}

    remap: dict[str, str] = {}
    if not no_canon:
        anchors = {ln for ln, f in freqs.items() if f >= anchor_min}
        remap = _build_remap(freqs, art_frac, anchors)
        wt = sum(freqs[v] for v in remap)
        click.echo(
            f"[canon] {len(remap):,} variants merged ({wt:,} weight); top examples:"
        )
        for v, t in sorted(remap.items(), key=lambda kv: -freqs[kv[0]])[:25]:
            click.echo(f"    {v} -> {t}  ({freqs[v]:,})")

    con.execute("CREATE TABLE remap(variant VARCHAR, canon VARCHAR)")
    if remap:
        con.executemany("INSERT INTO remap VALUES (?, ?)", list(remap.items()))
    con.execute(
        "CREATE TABLE norm AS SELECT canon AS last_name, state, SUM(n) AS n FROM ("
        "  SELECT COALESCE(r.canon, s.last_name) AS canon, s.state AS state, "
        "         s.n_times AS n "
        "  FROM stacked s LEFT JOIN remap r ON s.last_name = r.variant"
        ") GROUP BY canon, state"
    )
    if (
        train_out
    ):  # long (last_name,state,n_times) for the GRU, n>=3, same canonical space
        tp = Path(train_out)
        tp.parent.mkdir(parents=True, exist_ok=True)
        con.execute(
            "COPY (SELECT last_name, state, n AS n_times FROM norm "
            "WHERE regexp_full_match(last_name, '[a-z]+') AND length(last_name) > 2 "
            "AND n >= 3 ORDER BY last_name) "
            f"TO '{tp}' (FORMAT csv, HEADER, COMPRESSION gzip)"
        )
        click.echo(f"[ln-prop] GRU training rows -> {tp}")

    rel = con.execute(
        "SELECT last_name, state, n FROM norm "
        "WHERE regexp_full_match(last_name, '[a-z]+') AND length(last_name) > 2 "
        "ORDER BY last_name"
    )

    idx = {st: i for i, st in enumerate(V2_STATE_ORDER)}
    out.parent.mkdir(parents=True, exist_ok=True)
    parquet_out = out if out.suffix == ".parquet" else None
    if parquet_out is not None:
        out = out.with_suffix(".csv.gz")
    tmp = out.with_name(out.name + ".tmp")
    nrows = 0
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["last_name", *V2_STATE_ORDER, "total_n"])

        def flush(ln: str, vec: list[float]) -> int:
            tot = sum(vec)
            if tot < min_total:
                return 0
            w.writerow([ln, *(f"{x / tot:.10g}" for x in vec), int(tot)])
            return 1

        cur_ln: str | None = None
        vec = [0.0] * len(V2_STATE_ORDER)
        while batch := rel.fetchmany(100_000):
            for ln, state, n in batch:
                if ln != cur_ln:
                    if cur_ln is not None:
                        nrows += flush(cur_ln, vec)
                    cur_ln, vec = ln, [0.0] * len(V2_STATE_ORDER)
                vec[idx[state]] += n
        if cur_ln is not None:
            nrows += flush(cur_ln, vec)
    os.replace(tmp, out)
    click.echo(f"[ln-prop] {nrows:,} surnames x {len(V2_STATE_ORDER)} states -> {out}")
    if parquet_out is not None:
        # the packaged lookup table: same rows, typed (VARCHAR, 34 x DOUBLE, BIGINT)
        con.execute(
            f"COPY (SELECT * FROM read_csv('{out}', header = true)) "
            f"TO '{parquet_out}' (FORMAT parquet, COMPRESSION zstd)"
        )
        click.echo(f"[ln-prop] packaged parquet -> {parquet_out}")


# ---------------------------------------------------------------------------
# Phase 4: synthetic language labels from ranked state languages.
# Regenerate the table from v2 for the language BiLSTM and KNN lookup.
# ---------------------------------------------------------------------------

# Geometric decay over a state's 5 ranked most-spoken languages (reproduces the old weights;
# the shipped language model was trained on this scheme).
_LANG_RANK_COLS = (
    "most_spoken_lang",
    "second_most_spoken_lang",
    "third_most_spoken_lang",
    "fourth_most_spoken_lang",
    "fifth_most_spoken_lang",
)
_LANG_DECAY = (0.5, 0.25, 0.125, 0.0625, 0.03125)


def _load_constants_module():
    """Direct-load instate/constants.py (no package import -> no torch/Levenshtein)."""
    import importlib.util

    path = PACKAGE_ROOT / "constants.py"
    spec = importlib.util.spec_from_file_location("instate_constants", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm_lang(s: str) -> str:
    """Normalize a state_to_languages cell to a bare language name (drop parentheticals)."""
    return s.split("(")[0].strip().lower()


@cli.command(name="lang-prop")
@click.option(
    "--v2", "v2_path", default=None, help="v2 state-prop csv.gz (default bundled)."
)
@click.option(
    "--s2l", "s2l_path", default=None, help="state_to_languages.csv (default bundled)."
)
@click.option("--out", "out_path", default=None, help="Output lang_props_v2.csv.gz.")
def lang_prop(v2_path, s2l_path, out_path):
    """Merge v2 state distributions with state->language weights -> (last_name x 37 languages)."""
    import numpy as np
    import pandas as pd

    pkg = PACKAGE_ROOT / "data"
    v2p = Path(v2_path) if v2_path else _v2_default_out()
    s2lp = Path(s2l_path) if s2l_path else pkg / "state_to_languages.csv"
    # Build intermediate (gitignored, NOT bundled): the language BiLSTM trains on this.
    out = (
        Path(out_path)
        if out_path
        else Path(__file__).resolve().parents[1] / "data" / "lang_props_v2.csv.gz"
    )
    constants = _load_constants_module()
    languages = list(constants.LANGUAGES)
    lang_idx = {lng.lower(): i for i, lng in enumerate(languages)}

    # Build the (state x language) weight matrix W aligned to V2_STATE_ORDER / LANGUAGES.
    s2l = pd.read_csv(s2lp).set_index("state")
    W = np.zeros((len(V2_STATE_ORDER), len(languages)), dtype=np.float64)
    for si, state in enumerate(V2_STATE_ORDER):
        row = s2l.loc[constants.STATE_LANGUAGE_ALIASES.get(state, state)]
        for col, w in zip(_LANG_RANK_COLS, _LANG_DECAY, strict=True):
            val = row.get(col)
            if isinstance(val, str):
                li = lang_idx.get(_norm_lang(val))
                if li is not None:
                    W[si, li] += w

    # counts (N x 34) = state props * total_n; scores (N x 37) = counts @ W.
    df = pd.read_csv(v2p)
    names = df["last_name"].astype(str)
    counts = df[list(V2_STATE_ORDER)].to_numpy() * df["total_n"].to_numpy()[:, None]
    scores = counts @ W
    keep = scores.sum(axis=1) > 0
    out_df = pd.DataFrame(scores[keep].round(4), columns=languages)
    out_df.insert(0, "last_name", names[keep].to_numpy())
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, compression="gzip")
    click.echo(
        f"[lang-prop] {len(out_df):,} surnames x {len(languages)} languages -> {out}"
    )


def _load_word_map(corpus_csv) -> dict[str, str]:
    word_map: dict[str, str] = {}
    with gzip.open(corpus_csv, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                word_map[row[0]] = row[1]
    return word_map


if __name__ == "__main__":
    cli()
