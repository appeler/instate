"""Phase 1 for instate: per-state English voter-name count tables.

Aggregates a roll to ``(english_name, n_times)`` and writes ``<out_dir>/names_<state>.csv.gz``
(default out_dir is instate's ``data/``). Three romanization paths, one subcommand each:

  corpus   --state X            eroll corpus word-map (exact LLM transliterations)
  english  --roll F --lang X    roll already in English -> aggregate directly
  lstm     --roll F --lang X    indicate's trained Hindi/Punjabi LSTM (--script)

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
from eroll.states import STATES, StateConfig
from tqdm import tqdm

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data"  # instate/data

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


def _group2(roll_path, voter_col, father_col):
    """duckdb relation of ``(voter, father, n)`` grouped over both name columns."""
    con = duckdb.connect()
    return con.execute(
        f'SELECT "{voter_col}" AS v, "{father_col}" AS f, COUNT(*) AS n '
        f"FROM read_csv(?, header = true, all_varchar = true, ignore_errors = true) "
        f'WHERE "{voter_col}" IS NOT NULL AND "{voter_col}" <> \'\' '
        f'GROUP BY "{voter_col}", "{father_col}"',
        [str(roll_path)],
    )


def name_counts2_corpus(
    roll_path, *, voter_col, father_col, native_run, word_map, batch=100_000
):
    """Two-column: romanize voter (strict) + father/husband (best-effort) via the corpus."""
    rel = _group2(roll_path, voter_col, father_col)

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


def name_counts2_english(roll_path, *, voter_col, father_col, batch=100_000):
    """Two-column for an already-English roll."""
    rel = _group2(roll_path, voter_col, father_col)
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


def name_counts_via_lstm(
    roll_path, *, name_col, script, chunk=4000
) -> tuple[Counter, dict]:
    """Romanize via indicate's Hindi/Punjabi LSTM at the TOKEN level (memory-bounded).

    Transliterating millions of full names directly OOMs (beam search over huge batches),
    so we LSTM only the unique native TOKENS (~5x fewer, single words the model is built
    for), build a word-map, then substitute it onto every name -- the fast corpus path.
    """
    Model = {
        "hindi": "indicate.hindi2english:HindiToEnglish",
        "punjabi": "indicate.punjabi2english:PunjabiToEnglish",
    }[script]
    mod, cls = Model.split(":")
    Model = getattr(__import__(mod, fromlist=[cls]), cls)
    Model.BEAM_WIDTH = (
        1  # greedy: ~5x faster, ~1.5pt lower exact-match (fine for names)
    )
    lo, hi = LSTM_SCRIPTS[script]
    native_run = re.compile(f"[{lo}-{hi}]+")

    rel = _group_names(roll_path, name_col)
    rows = rel.fetchall()
    tokens = sorted({t for nm, _ in rows for t in native_run.findall(nm)})
    word_map: dict[str, str] = {}
    for i in tqdm(range(0, len(tokens), chunk), desc=f"{script} lstm tokens"):
        part = tokens[i : i + chunk]
        for tok, raw in zip(part, Model.transliterate_batch(part), strict=False):
            word_map[tok] = to_ascii(
                raw if isinstance(raw, str) else (raw[0] if raw else "")
            )

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
@click.option("--state", required=True, type=click.Choice(sorted(STATES)))
@click.option("--name-col", default="elector_name", show_default=True)
@click.option("--father-col", default="father_or_husband_name", show_default=True)
@click.option(
    "--roll", default=None, help="Roll path (default: the state's input_path)."
)
@click.option("--out-dir", default=None)
def corpus(state, name_col, father_col, roll, out_dir):
    """Romanize voter + father/husband via the state's eroll corpus word-map."""
    cfg: StateConfig = STATES[state]
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
@click.option("--out-dir", default=None)
def english(roll, lang, name_col, father_col, out_dir):
    """Roll already in English -> aggregate (voter, father/husband) directly."""
    click.echo(f"[{lang}] aggregating English roll {Path(roll).name} ...")
    counts, stats = name_counts2_english(
        roll, voter_col=name_col, father_col=father_col
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
# Last-name extraction (Phase 2): resolve a surname per (voter, father/husband)
# row of the names_<state> tables, weighted by n_times, then aggregate per state.
# ---------------------------------------------------------------------------

# Map each names_<slug>.csv.gz to its full state/UT name (the column headers used by
# instate's P(state|last_name) product). Covers all 34 tables -- the canonical slug set.
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
# Phase 3: merge the 34 last_names tables into P(state | last_name) (the v2 lookup).
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
    return (
        Path(__file__).resolve().parents[2]
        / "instate"
        / "data"
        / "instate_unique_ln_state_prop_v2.csv.gz"
    )


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
    """Merge the 34 last_names tables -> P(state | last_name) v2 (canonicalized, normalized)."""
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
