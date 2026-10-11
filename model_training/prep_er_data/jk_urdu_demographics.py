"""Recover labeled Urdu demographics using verified outlines and card geometry."""

import argparse
import hashlib
import io
import itertools
import json
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from fontTools.ttLib import TTLibError

from model_training.prep_er_data.english_glyphs import (
    ContourPen,
    ReferenceFonts,
    measure,
)
from model_training.prep_er_data.jk_urdu_nastaleeq import (
    _CONTROLS,
    _label_before,
    decode_line,
)


def sha256(path):
    """Return the digest of one input or completed checkpoint."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class NumericReference(ReferenceFonts):
    """Match numeric outlines to MuPDF's standard Type1 reference fonts."""

    def __init__(self):
        """Build a restricted digit/separator vocabulary from reference CFF fonts."""
        import pymupdf
        from fontTools.cffLib import CFFFontSet
        from fontTools.ttLib import TTFont

        self.sources, self.references, self.cache = [], [], {}
        names = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "slash",
            "hyphen",
            "space",
        ]
        for family in ("Times-Roman", "Helvetica", "Times-Bold", "Helvetica-Bold"):
            cff = CFFFontSet()
            cff.decompile(io.BytesIO(pymupdf.Font(family).buffer), TTFont())
            font = cff[0]
            for name, text in zip(names, "0123456789/- ", strict=True):
                glyph = font.CharStrings[name]
                raster = ContourPen(None, 1)
                glyph.draw(raster)
                self.references.append(
                    {
                        "text": text,
                        "width": glyph.width,
                        "bounds": measure(glyph, None, 1),
                        "raster": raster.raster(),
                    }
                )


def bottom_filter(boxes):
    """Select bottom line bands without decoding name glyphs."""
    starts = sorted({box[3] - 44 for box in boxes})
    ends = {box[3] - 44: box[3] for box in boxes}

    def include(origin):
        index = bisect_right(starts, origin[1]) - 1
        return index >= 0 and origin[1] <= ends[starts[index]]

    return include


def numeric_glyphs(page, fonts, include):
    """Decode numeric-field Latin fonts from their physical CFF/TrueType outlines."""
    spans = [
        s
        for s in page.get_texttrace()
        if s["font"].startswith(("Times", "Arial", "Helvetica"))
    ]
    mappings = fonts.page(page, {span["font"] for span in spans})
    standard = defaultdict(list)
    for xref, extension, _, name, *_ in page.get_fonts(full=True):
        name = name.split("+")[-1]
        standard[name].append(
            extension == "n/a"
            and name in {"TimesNewRomanPSMT", "TimesNewRomanPS-BoldMT", "Arial-BoldMT"}
            and fonts.document.xref_get_key(xref, "Encoding")
            == ("name", "/WinAnsiEncoding")
            and fonts.document.xref_get_key(xref, "ToUnicode")[0] == "null"
        )
    trusted = {name for name, choices in standard.items() if all(choices)}
    return [
        {
            "origin": char[2],
            "text": chr(char[0])
            if span["font"] in trusted and chr(char[0]) in "0123456789/- "
            else mappings.get(span["font"], {}).get(char[1], ("�",))[0],
            "method": "standard_winansi"
            if span["font"] in trusted
            else "outline_match",
        }
        for span in spans
        for char in span["chars"]
        if include(char[2])
    ]


def recover_card(box, glyphs, numbers, colon_key, labels, sex_reference, images=()):
    """Require all three fixed field labels before assigning numeric positions."""
    empty = {"house_no": None, "age": None, "sex_candidate": None}
    if not (175 <= box[2] - box[0] <= 185 and 61 <= box[3] - box[1] <= 100):
        return empty, "unsupported_geometry"
    card = glyphs.card(box)
    separators = [
        g for g in card if g["key"] == colon_key and g["origin"][1] > box[3] - 32
    ]
    fields = {}
    for separator in separators:
        label_card = [g for g in card if g["origin"][0] <= separator["origin"][0] + 40]
        label = _label_before(glyphs, label_card, separator, _CONTROLS, labels)
        if label is None:
            continue
        key = {
            "خانہ": "house_no",
            "خانہنمبر": "house_no",
            "عمر": "age",
            "جنس": "sex_candidate",
        }[label["text"]]
        if key in fields:
            return empty, "duplicate_label"
        label_glyphs = glyphs[label["start"] : label["end"]]
        fields[key] = (separator, max(g["origin"][0] for g in label_glyphs))
    if set(fields) != set(empty):
        return empty, "missing_label"
    output = dict(empty)
    for key in fields:
        separator, _ = fields[key]
        x, y = separator["origin"]
        left = max(
            (
                right + 0.1
                for other, right in fields.values()
                if other["origin"][0] < x and abs(other["origin"][1] - y) <= 6
            ),
            default=box[0],
        )
        if any(
            min(x, image[2]) > max(left, image[0])
            and min(y + 4, image[3]) > max(y - 11, image[1])
            for image in images
        ):
            continue
        if key == "sex_candidate":
            selected = [
                g
                for g in card
                if left <= g["origin"][0] < x - 0.01
                and y - 11 <= g["origin"][1] <= y + 4
            ]
            candidates, issue = decode_line(selected, sex_reference)
            if not issue and len(candidates) == 1 and candidates[0] in {"مرد", "عورت"}:
                output[key] = candidates[0]
        else:
            selected = sorted(
                (
                    g
                    for g in numbers
                    if left <= g["origin"][0] < x and abs(g["origin"][1] - y) <= 2
                ),
                key=lambda g: g["origin"][0],
            )
            value = "".join(g["text"] for g in selected).strip()
            selected_origins = {tuple(g["origin"]) for g in selected}
            if any(
                left <= g["origin"][0] < x
                and y - 11 <= g["origin"][1] <= y + 4
                and g["key"] != "space"
                and tuple(g["origin"]) not in selected_origins
                for g in card
            ):
                continue
            allowed = "0123456789" if key == "age" else "0123456789/-"
            if (
                value
                and any(c.isdigit() for c in value)
                and all(c in allowed for c in value)
            ):
                output[key] = value
    return output, None


_REFERENCE = _NUMERIC = _COLON = None


def initialize(reference_font, colon_catalog):
    """Initialize per-process reference caches."""
    from model_training.prep_er_data.urdu_geometry import LabelReferenceFont

    global _REFERENCE, _NUMERIC, _COLON
    _REFERENCE = LabelReferenceFont(reference_font)
    _NUMERIC = NumericReference()
    _COLON = json.loads(Path(colon_catalog).read_text())["development_colon_anchor"][
        "outline_key"
    ]


def recover_part(payload):
    """Recover one source PDF, checking its hash and keeping every input identity."""
    import pymupdf

    from model_training.prep_er_data.english_glyphs import DocumentFonts
    from model_training.prep_er_data.jk_urdu_nastaleeq import PdfOutlines

    rows, pdf_root, output = payload
    filename = rows[0]["filename"]
    path = Path(pdf_root) / filename
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if any(row["source_sha256"] != digest for row in rows):
        raise ValueError("PDF does not match inventory source hash")
    by_page = defaultdict(list)
    for row in rows:
        by_page[row["page"]].append(row)
    counts, records = Counter(), []
    started = time.monotonic()
    with pymupdf.open(path) as document:
        native = PdfOutlines(document, _REFERENCE)
        latin = DocumentFonts(document, _NUMERIC)
        for page_number, page_rows in by_page.items():
            page = document[page_number - 1]
            include = bottom_filter([row["bbox"] for row in page_rows])
            font_issue = None
            try:
                glyphs = native.page(page, include)
                numbers = numeric_glyphs(page, latin, include)
            except TTLibError:
                glyphs, numbers = None, []
                font_issue = "malformed_embedded_font"
            counts.update("numeric_glyph_" + g["method"] for g in numbers)
            images = [image["bbox"] for image in page.get_image_info()]
            for row in page_rows:
                if font_issue:
                    values = {"house_no": None, "age": None, "sex_candidate": None}
                    issue = font_issue
                else:
                    values, issue = recover_card(
                        row["bbox"],
                        glyphs,
                        numbers,
                        _COLON,
                        _REFERENCE,
                        _REFERENCE,
                        images,
                    )
                counts[issue or "verified_labels"] += 1
                counts.update(key for key, value in values.items() if value is not None)
                records.append({**row, **values, "demographic_issue": issue})
    fields = [
        pa.field(key, pa.string())
        for key in (
            "filename",
            "source_sha256",
            "entry_event_key",
            "event_key",
            "number",
            "id",
            "house_no",
            "age",
            "sex_candidate",
            "demographic_issue",
        )
    ]
    fields.extend(
        [
            pa.field("page", pa.int32()),
            pa.field("active", pa.bool_()),
            pa.field("assembly_eligible", pa.bool_()),
            pa.field("bbox", pa.list_(pa.float64())),
        ]
    )
    target = Path(output) / (filename + ".parquet")
    temporary = target.with_suffix(".parquet.tmp")
    pq.write_table(
        pa.Table.from_pylist(records, schema=pa.schema(fields)),
        temporary,
        compression="zstd",
    )
    temporary.replace(target)
    report = {
        "filename": filename,
        "source_sha256": digest,
        "rows": len(rows),
        "counts": dict(counts),
        "seconds": round(time.monotonic() - started, 3),
        "artifact_sha256": sha256(target),
    }
    target.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    """Run a pilot or full private demographic extraction with part checkpoints."""
    import duckdb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--pdf-root", type=Path, required=True)
    parser.add_argument("--reference-font", type=Path, required=True)
    parser.add_argument("--colon-catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parts", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    code = [
        Path(__file__),
        *[
            Path(__file__).with_name(name)
            for name in (
                "jk_urdu_nastaleeq.py",
                "english_glyphs.py",
                "urdu_geometry.py",
                "urdu_text.py",
                "hindi_glyphs.py",
                "jk_urdu_unicode.py",
            )
        ],
    ]
    inputs = [args.inventory, args.reference_font, args.colon_catalog, *code]
    if args.parts:
        inputs.append(args.parts)
    hashes = {str(path): sha256(path) for path in inputs}
    versions = {
        name: version(name)
        for name in (
            "PyMuPDF",
            "fonttools",
            "uharfbuzz",
            "numpy",
            "pyarrow",
            "duckdb",
        )
    }
    manifest = {
        "input_sha256": hashes,
        "versions": versions,
        "pdf_root": str(args.pdf_root.resolve()),
    }
    args.out.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = args.out / "run_manifest.json"
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume requires identical inputs, code and dependencies")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    connection = duckdb.connect()
    connection.execute("SET memory_limit='1GB'")
    connection.read_parquet(str(args.inventory)).create_view("inventory")
    query = (
        "SELECT filename,source_sha256,entry_event_key,event_key,number,id,page,"
        "active,assembly_eligible,bbox FROM inventory"
    )
    if args.parts:
        wanted = pa.table({"filename": args.parts.read_text().splitlines()})
        connection.register("wanted", wanted)
        query += " WHERE filename IN (SELECT filename FROM wanted)"
    query += " ORDER BY filename,page,entry_event_key"
    reader = connection.sql(query).to_arrow_reader(16384)
    rows = (row for batch in reader for row in batch.to_pylist())
    payloads = (
        (list(group), str(args.pdf_root), str(args.out))
        for _, group in itertools.groupby(rows, key=lambda row: row["filename"])
    )
    reports = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize,
        initargs=(args.reference_font, args.colon_catalog),
    ) as pool:
        for wave in itertools.batched(payloads, args.workers * 2):
            pending = []
            for payload in wave:
                target = args.out / (payload[0][0]["filename"] + ".parquet")
                report_path = target.with_suffix(".json")
                if args.resume and target.is_file() and report_path.is_file():
                    report = json.loads(report_path.read_text())
                    if sha256(target) != report["artifact_sha256"]:
                        raise ValueError("Corrupt demographic checkpoint")
                    reports.append(report)
                else:
                    pending.append(payload)
            reports.extend(pool.map(recover_part, pending))
            (args.out / "progress.json").write_text(
                json.dumps(
                    {
                        "parts": len(reports),
                        "rows": sum(p["rows"] for p in reports),
                        "counts": dict(
                            sum((Counter(p["counts"]) for p in reports), Counter())
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(len(reports), "parts completed", flush=True)  # noqa: T201
    if hashes != {str(path): sha256(path) for path in inputs}:
        raise ValueError("Inputs or code changed during the extraction")
    (args.out / "audit.json").write_text(
        json.dumps(
            {
                "parts": reports,
                "input_sha256": hashes,
                "versions": versions,
                "additional_paid_calls": 0,
                "policy": (
                    "Exact source identity; finite fixed labels; "
                    "bounded outline matches "
                    "or explicit unembedded WinAnsi numeric encoding; "
                    "missing/ambiguous fields abstain."
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
