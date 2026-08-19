"""Build the state-by-language mother-tongue share table from Census 2011 C-16.

Reads the India-level C-16 file (mother tongue by state) and the Andhra
Pradesh state file (mother tongue by district), splits undivided Andhra
Pradesh into Telangana (the ten 2011 districts, codes 532-541) and residual
Andhra Pradesh, maps census area names to instate's electoral-roll state
labels, keeps every language whose mother-tongue share reaches the floor in
at least one state, pools the remainder into ``other``, and writes a typed
Parquet table plus a provenance manifest.

Output:
    src/instate/data/state_language_shares.parquet
    src/instate/data/state_language_shares.manifest.json

Source files are census downloads pinned by SHA-256 (see ``SOURCES``); the
script downloads them into ``--source-dir`` when absent. censusindia.gov.in
serves an incomplete certificate chain, so downloads skip TLS verification
and rely on the hash pin for integrity.

Run:
    uv run --python .venv/bin/python --with openpyxl \
        python model_training/build_state_language_shares.py

Known approximations, recorded in the manifest:
- Language-level grouping: constituent mother tongues (Marwari, Bhojpuri,
  Kumauni, ...) count under their census parent language (Hindi, ...).
- Telangana uses 2011 district boundaries; the seven Bhadrachalam-area
  mandals moved to Andhra Pradesh in 2014 (about 0.5% of Telangana's
  population, both sides Telugu-dominant) stay on the Telangana side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

INSTATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSTATE_ROOT / "src"))
from instate.constants import GT_KEYS  # noqa: E402
OUTPUT_PATH = INSTATE_ROOT / "src" / "instate" / "data" / "state_language_shares.parquet"
MANIFEST_PATH = OUTPUT_PATH.with_name("state_language_shares.manifest.json")

SHARE_FLOOR = 0.01
POOLED_LANGUAGE = "other"

SOURCES = {
    "DDW-C16-STMT-MDDS-0000.XLSX": {
        "url": (
            "https://censusindia.gov.in/nada/index.php/catalog/10191/"
            "download/13303/DDW-C16-STMT-MDDS-0000.XLSX"
        ),
        "catalog": "https://censusindia.gov.in/nada/index.php/catalog/10191",
        "sha256": "b800cd544d7196c9bbc3c88ec1814138b8b7c8d4ccb7a5a2263a84e4bbf28f83",
        "content": "C-16 population by mother tongue, India and states, 2011",
    },
    "DDW-C16-STMT-MDDS-2800.XLSX": {
        "url": (
            "https://censusindia.gov.in/nada/index.php/catalog/10193/"
            "download/13305/DDW-C16-STMT-MDDS-2800.XLSX"
        ),
        "catalog": "https://censusindia.gov.in/nada/index.php/catalog/10193",
        "sha256": "cd4be53ef2f022629805097e45d02aed2b62880d072231a9358407b346f6dcc4",
        "content": "C-16 population by mother tongue, Andhra Pradesh districts, 2011",
    },
}

COLUMNS = [
    "table",
    "state_code",
    "district",
    "subdistrict",
    "area",
    "mt_code",
    "mt_name",
    "total",
    "male",
    "female",
    "rural",
    "rural_m",
    "rural_f",
    "urban",
    "urban_m",
    "urban_f",
]

# Census 2011 area names -> instate electoral-roll state labels. Lakshadweep
# has no electoral-roll states entry and is dropped; undivided Andhra Pradesh
# is replaced by the Telangana split below.
AREA_TO_STATE = {
    "ANDAMAN & NICOBAR ISLANDS": "Andaman and Nicobar Islands",
    "ARUNACHAL PRADESH": "Arunachal Pradesh",
    "ASSAM": "Assam",
    "BIHAR": "Bihar",
    "CHANDIGARH": "Chandigarh",
    "CHHATTISGARH": "Chhattisgarh",
    "DADRA & NAGAR HAVELI": "Dadra and Nagar Haveli",
    "DAMAN & DIU": "Daman and Diu",
    "NCT OF DELHI": "Delhi",
    "GOA": "Goa",
    "GUJARAT": "Gujarat",
    "HARYANA": "Haryana",
    "HIMACHAL PRADESH": "Himachal Pradesh",
    "JAMMU & KASHMIR": "Jammu and Kashmir and Ladakh",
    "JHARKHAND": "Jharkhand",
    "KARNATAKA": "Karnataka",
    "KERALA": "Kerala",
    "MADHYA PRADESH": "Madhya Pradesh",
    "MAHARASHTRA": "Maharashtra",
    "MANIPUR": "Manipur",
    "MEGHALAYA": "Meghalaya",
    "MIZORAM": "Mizoram",
    "NAGALAND": "Nagaland",
    "ODISHA": "Odisha",
    "PUDUCHERRY": "Puducherry",
    "PUNJAB": "Punjab",
    "RAJASTHAN": "Rajasthan",
    "SIKKIM": "Sikkim",
    "TAMIL NADU": "Tamil Nadu",
    "TRIPURA": "Tripura",
    "UTTARAKHAND": "Uttarakhand",
    "UTTAR PRADESH": "Uttar Pradesh",
    "WEST BENGAL": "West Bengal",
}

# 2011 census district codes of the districts that formed Telangana in 2014.
TELANGANA_DISTRICTS = {
    "532": "Adilabad",
    "533": "Nizamabad",
    "534": "Karimnagar",
    "535": "Medak",
    "536": "Hyderabad",
    "537": "Rangareddy",
    "538": "Mahbubnagar",
    "539": "Nalgonda",
    "540": "Warangal",
    "541": "Khammam",
}

# Sum of language-level rows must reproduce the published 2011 populations
# exactly; any difference means the denominator is wrong.
CONTROL_TOTALS = {
    "INDIA": 1_210_854_977,
    "PUNJAB": 27_743_338,
    "KERALA": 33_406_061,
    "ANDHRA PRADESH": 84_580_777,
    "NCT OF DELHI": 16_787_941,
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_sources(source_dir: Path) -> dict[str, Path]:
    """Download any missing source file and verify every hash pin."""
    source_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, source in SOURCES.items():
        path = source_dir / name
        if not path.is_file():
            print(f"downloading {name} ...", flush=True)
            try:
                with (
                    urllib.request.urlopen(source["url"]) as response,  # noqa: S310
                    path.open("wb") as handle,
                ):
                    handle.write(response.read())
            except ssl.SSLError:
                # censusindia.gov.in serves an incomplete certificate chain;
                # the SHA-256 pin below is the integrity check either way.
                context = ssl._create_unverified_context()  # noqa: S323, SLF001
                with (
                    urllib.request.urlopen(  # noqa: S310
                        source["url"], context=context
                    ) as response,
                    path.open("wb") as handle,
                ):
                    handle.write(response.read())
        digest = sha256_file(path)
        if digest != source["sha256"]:
            raise SystemExit(
                f"{name}: SHA-256 {digest} does not match pin {source['sha256']}"
            )
        paths[name] = path
    return paths


def read_c16(path: Path) -> pd.DataFrame:
    """Read a C-16 statement file and normalize the language-level rows."""
    frame = pd.read_excel(
        path,
        header=None,
        skiprows=6,
        names=COLUMNS,
        dtype={"state_code": str, "district": str, "subdistrict": str, "mt_code": str},
    )
    frame = frame[frame.mt_code.notna()]
    # Codes ending in 000 are grouped language totals; the rows beneath them
    # are constituent mother tongues and would double-count.
    frame = frame[frame.mt_code.str.endswith("000")].copy()
    frame["language"] = (
        frame.mt_name.str.replace(r"^\d+\s*", "", regex=True).str.strip().str.lower()
    )
    frame["area"] = frame.area.str.strip()
    frame["total"] = frame.total.astype("int64")
    return frame


def state_language_counts(source_dir: Path) -> pd.DataFrame:
    """Return (state, language, population) for instate's 34 states."""
    paths = fetch_sources(source_dir)
    india = read_c16(paths["DDW-C16-STMT-MDDS-0000.XLSX"])

    state_rows = india[india.district == "000"]
    totals = state_rows.groupby("area")["total"].sum()
    for area, expected in CONTROL_TOTALS.items():
        actual = int(totals[area])
        if actual != expected:
            raise SystemExit(
                f"control total failed for {area}: {actual:,} != {expected:,}"
            )

    mapped = state_rows[state_rows.area.isin(AREA_TO_STATE)].copy()
    mapped["state"] = mapped.area.map(AREA_TO_STATE)

    andhra = read_c16(paths["DDW-C16-STMT-MDDS-2800.XLSX"])
    districts = andhra[(andhra.district != "000") & (andhra.subdistrict == "00000")]
    named = districts[["district", "area"]].drop_duplicates().set_index("district").area
    for code, name in TELANGANA_DISTRICTS.items():
        if named.get(code) != name:
            raise SystemExit(
                f"district {code} is {named.get(code)!r}, expected {name!r}"
            )
    is_telangana = districts.district.isin(TELANGANA_DISTRICTS)
    split_frames = []
    for state, subset in (
        ("Telangana", districts[is_telangana]),
        ("Andhra Pradesh", districts[~is_telangana]),
    ):
        grouped = subset.groupby("language", as_index=False)["total"].sum()
        grouped["state"] = state
        split_frames.append(grouped)
    split = pd.concat(split_frames, ignore_index=True)

    undivided = int(totals["ANDHRA PRADESH"])
    if int(split.total.sum()) != undivided:
        raise SystemExit("Telangana plus residual Andhra Pradesh != undivided total")

    counts = pd.concat(
        [mapped[["state", "language", "total"]], split[["state", "language", "total"]]],
        ignore_index=True,
    )
    # The artifact must cover exactly the package's electoral-roll states:
    # the census additionally has Lakshadweep and Chhattisgarh, which the
    # 2017 rolls data does not.
    counts = counts[counts.state.isin(GT_KEYS)]
    if set(counts.state.unique()) != set(GT_KEYS):
        missing = set(GT_KEYS) - set(counts.state.unique())
        raise SystemExit(f"states missing from census build: {sorted(missing)}")
    return counts.rename(columns={"total": "population"})


def build_shares(counts: pd.DataFrame) -> pd.DataFrame:
    """Apply the share floor, pool the tail into ``other``, and add shares."""
    state_totals = counts.groupby("state")["population"].transform("sum")
    counts = counts.assign(share=counts.population / state_totals)
    reaches_floor = (
        counts[counts.language != "others"]
        .groupby("language")["share"]
        .max()
        .pipe(lambda shares: set(shares[shares >= SHARE_FLOOR].index))
    )
    counts["language"] = counts.language.where(
        counts.language.isin(reaches_floor), POOLED_LANGUAGE
    )
    pooled = counts.groupby(["state", "language"], as_index=False)["population"].sum()

    states = sorted(pooled.state.unique())
    languages = sorted(reaches_floor) + [POOLED_LANGUAGE]
    grid = pd.MultiIndex.from_product(
        [states, languages], names=["state", "language"]
    ).to_frame(index=False)
    full = grid.merge(pooled, how="left", on=["state", "language"])
    full["population"] = full.population.fillna(0).astype("int64")
    state_totals = full.groupby("state")["population"].transform("sum")
    full["share"] = full.population / state_totals

    sums = full.groupby("state")["share"].sum()
    if not ((sums - 1.0).abs() < 1e-9).all():
        raise SystemExit("state shares do not sum to one")
    if len(full) != len(states) * len(languages):
        raise SystemExit("row contract failed: grid is not complete")
    return full


def write_artifact(shares: pd.DataFrame, source_dir: Path) -> None:
    """Write the typed Parquet table and its provenance manifest."""
    schema = pa.schema(
        [
            pa.field("state", pa.string(), nullable=False),
            pa.field("language", pa.string(), nullable=False),
            pa.field("population", pa.int64(), nullable=False),
            pa.field("share", pa.float64(), nullable=False),
        ]
    )
    table = pa.Table.from_pandas(shares, schema=schema, preserve_index=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, OUTPUT_PATH)

    manifest = {
        "schema_version": 1,
        "artifact": {
            "filename": OUTPUT_PATH.name,
            "sha256": sha256_file(OUTPUT_PATH),
            "rows": len(shares),
            "states": int(shares.state.nunique()),
            "languages": int(shares.language.nunique()),
            "row_definition": (
                "one (state, language) cell; share is the language's fraction of "
                "the state's 2011 mother-tongue population"
            ),
        },
        "sources": {
            name: {key: value for key, value in source.items()}
            for name, source in SOURCES.items()
        },
        "decisions": {
            "language_grouping": (
                "census language level: constituent mother tongues count under "
                "their census parent language"
            ),
            "share_floor": SHARE_FLOOR,
            "pooled_language": POOLED_LANGUAGE,
            "pooled_definition": (
                "languages below the floor in every state, plus the census "
                "'others' residual category"
            ),
            "telangana": (
                "2011 districts 532-541; the seven Bhadrachalam-area mandals "
                "transferred to Andhra Pradesh in 2014 remain on the Telangana "
                "side (about 0.5% of its population)"
            ),
            "excluded_areas": [
                "INDIA",
                "LAKSHADWEEP",
                "CHHATTISGARH (not in the electoral-roll state vocabulary)",
            ],
        },
        "builder": "model_training/build_state_language_shares.py",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Build the artifact and print the verification summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=INSTATE_ROOT / "model_training" / "data" / "census",
        help="Directory holding (or receiving) the pinned census downloads.",
    )
    args = parser.parse_args()

    counts = state_language_counts(args.source_dir)
    shares = build_shares(counts)
    write_artifact(shares, args.source_dir)

    print(f"states {shares.state.nunique()}  languages {shares.language.nunique()}")
    print(f"rows {len(shares)} -> {OUTPUT_PATH}")
    print(f"manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
