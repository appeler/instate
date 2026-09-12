"""Decode embedded CFF outlines against local Latin reference fonts.

Character codes and subset glyph names are deliberately not used as mappings.
Matches require advance/bounds agreement and bidirectional contour agreement
within four pixels at 1,000 pixels/em. This is a rendering check, not OCR ground
truth. Unknown or ambiguous outlines remain replacement characters.
"""

import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

from fontTools.cffLib import CFFFontSet
from fontTools.pens.basePen import BasePen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import DecomposingRecordingPen, RecordingPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from model_training.prep_er_data.hindi_glyphs import glyph_groups

TOLERANCE = 4
CANVAS = (1500, 1500)


class ContourPen(BasePen):
    """Rasterize normalized contours without changing their origin or scale."""

    def __init__(self, glyph_set, scale):
        super().__init__(glyph_set)
        self.scale = scale
        self.paths = []

    def _moveTo(self, point):
        self.paths.append([point])

    def _lineTo(self, point):
        self.paths[-1].append(point)

    def _curveToOne(self, first, second, last):
        start = self._getCurrentPoint()
        for step in range(1, 65):
            t = step / 64
            self.paths[-1].append(
                tuple(
                    (1 - t) ** 3 * start[k]
                    + 3 * (1 - t) ** 2 * t * first[k]
                    + 3 * (1 - t) * t**2 * second[k]
                    + t**3 * last[k]
                    for k in (0, 1)
                )
            )

    def _closePath(self):
        self.paths[-1].append(self.paths[-1][0])

    def raster(self):
        image = Image.new("1", CANVAS)
        drawing = ImageDraw.Draw(image)
        for path in self.paths:
            points = [
                (round(x * self.scale + 250), round(1200 - y * self.scale))
                for x, y in path
            ]
            if any(not (0 <= x < 1500 and 0 <= y < 1500) for x, y in points):
                raise ValueError("Glyph extends outside the comparison canvas")
            drawing.line(points, fill=1, width=1)
        return image


def contours_agree(left, right):
    """Require every contour pixel to be near the other contour, in both directions."""
    bounds = ImageChops.lighter(left, right).getbbox()
    if bounds is None:
        return True
    x0, y0, x1, y1 = bounds
    crop = (x0 - TOLERANCE, y0 - TOLERANCE, x1 + TOLERANCE, y1 + TOLERANCE)
    left, right = left.crop(crop), right.crop(crop)
    dilation = ImageFilter.MaxFilter(2 * TOLERANCE + 1)
    return (
        ImageChops.subtract(left, right.filter(dilation)).getbbox() is None
        and ImageChops.subtract(right, left.filter(dilation)).getbbox() is None
    )


def measure(glyph, glyph_set, scale):
    pen = BoundsPen(glyph_set)
    glyph.draw(pen)
    return tuple(v * scale for v in pen.bounds) if pen.bounds else None


def close_metrics(width, bounds, reference):
    if abs(width - reference["width"]) > TOLERANCE:
        return False
    other = reference["bounds"]
    if bounds is None or other is None:
        return bounds is None and other is None
    return all(abs(a - b) <= TOLERANCE for a, b in zip(bounds, other, strict=True))


class ReferenceFonts:
    """Match outlines to unique ASCII characters defined by reference font cmaps."""

    def __init__(self, paths):
        self.sources, self.references, self.cache = [], [], {}
        for path in paths:
            path = Path(path)
            with TTFont(path) as font:
                scale = 1000 / font["head"].unitsPerEm
                glyph_set = font.getGlyphSet()
                for codepoint, name in (font.getBestCmap() or {}).items():
                    if not 32 <= codepoint < 127 or name == ".notdef":
                        continue
                    glyph = glyph_set[name]
                    # Rasterize now, while composite reference glyphs are available.
                    raster = ContourPen(glyph_set, scale)
                    glyph.draw(raster)
                    self.references.append(
                        {
                            "text": chr(codepoint),
                            "width": glyph.width * scale,
                            "bounds": measure(glyph, glyph_set, scale),
                            "raster": raster.raster(),
                        }
                    )
            self.sources.append(
                {"filename": path.name, "sha256": sha256(path.read_bytes())}
            )
        if not self.references:
            raise ValueError("Reference fonts contain no printable ASCII mappings")

    def decode(self, charstring, matrix, glyph_name):
        if glyph_name == ".notdef":
            return "�", "notdef"
        if matrix != [0.001, 0, 0, 0.001, 0, 0]:
            return "�", "unsupported_font_matrix"
        return self.match(charstring, None, 1)

    def decode_truetype(self, font, gid):
        if gid <= 0 or gid >= len(font.getGlyphOrder()):
            return "�", "notdef"
        glyph_set = font.getGlyphSet()
        return self.match(
            glyph_set[font.getGlyphName(gid)],
            glyph_set,
            1000 / font["head"].unitsPerEm,
        )

    def match(self, glyph, glyph_set, scale):
        pen = DecomposingRecordingPen(glyph_set) if glyph_set else RecordingPen()
        glyph.draw(pen)
        width = glyph.width * scale
        fingerprint = sha256(json.dumps([scale, width, pen.value]).encode())
        if fingerprint not in self.cache:
            bounds = measure(glyph, glyph_set, scale)
            candidates = [
                ref for ref in self.references if close_metrics(width, bounds, ref)
            ]
            if bounds is None:
                # Only an empty, positive-width glyph matching a cmap space is valid.
                meanings = {
                    ref["text"]
                    for ref in candidates
                    if ref["text"] == " " and width > 0
                }
            else:
                raster = ContourPen(glyph_set, scale)
                glyph.draw(raster)
                source = raster.raster()
                meanings = {
                    ref["text"]
                    for ref in candidates
                    if contours_agree(source, ref["raster"])
                }
            status = "outline_match" if len(meanings) == 1 else "unmapped_outline"
            if len(meanings) > 1:
                status = "ambiguous_outline"
            self.cache[fingerprint] = (
                next(iter(meanings)) if len(meanings) == 1 else "�",
                status,
            )
        return self.cache[fingerprint]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class DocumentFonts:
    """Resolve each page's subset glyph IDs; reject conflicting font resources."""

    def __init__(self, document, references):
        self.document, self.references = document, references
        self.fonts = {}

    def page(self, page):
        fonts = defaultdict(lambda: defaultdict(set))
        for xref, extension, _, name, *_ in page.get_fonts(full=True):
            if extension not in ("cff", "ttf"):
                continue
            if xref not in self.fonts:
                data = self.document.extract_font(xref)[3]
                if extension == "cff":
                    cff = CFFFontSet()
                    cff.decompile(io.BytesIO(data), None)
                    if len(cff) != 1:
                        raise ValueError("Expected one embedded CFF font")
                    font = cff[0]
                    self.fonts[xref] = {
                        gid: self.references.decode(
                            font.CharStrings[glyph], font.FontMatrix, glyph
                        )
                        for gid, glyph in enumerate(font.charset)
                    }
                else:
                    with TTFont(io.BytesIO(data)) as font:
                        self.fonts[xref] = {
                            gid: self.references.decode_truetype(font, gid)
                            for gid in range(len(font.getGlyphOrder()))
                        }
            for gid, value in self.fonts[xref].items():
                fonts[name.split("+")[-1]][gid].add(value)
        return {
            name: {
                gid: next(iter(values))
                if len(values) == 1
                else ("�", "ambiguous_font_resource")
                for gid, values in entries.items()
            }
            for name, entries in fonts.items()
        }


def extract_page(page, fonts):
    mappings = fonts.page(page)
    glyphs = []
    for span in page.get_texttrace():
        mapping = mappings.get(span["font"])
        for group in glyph_groups(span["chars"]):
            if mapping is not None:
                decoded, status = mapping.get(group["gid"], ("�", "missing_glyph"))
            elif span["font"] == "Times-Roman" and all(
                32 <= ord(char) < 127 for char in group["text"]
            ):
                decoded, status = group["text"], "standard_font_text"
            else:
                decoded, status = "�", "unsupported_font"
            glyphs.append(
                {
                    **group,
                    "decoded": decoded,
                    "status": status,
                    "font": span["font"],
                    "ordinal": len(glyphs),
                }
            )
    return glyphs
