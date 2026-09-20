"""Check Nastaleeq candidates against contours and complete word positions.

This decoder returns all candidates within a fixed geometric policy. Callers
must verify field boundaries and withhold ambiguous results. It does not infer
Unicode from source glyph numbers or treat OCR suggestions as ground truth.
"""

import math
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import pairwise, permutations
from types import SimpleNamespace

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from scipy.spatial import cKDTree

from model_training.prep_er_data.jk_urdu_unicode import contours, outline_key
from model_training.prep_er_data.urdu_text import ReferenceFont, combine, complete_text

# The observed source and reference fonts use 2048 units per em.
UNITS = 2048
BOUND = 32
STEP = 2
CURVE_ERROR = 1 / 64
MARK_ERROR = 2


class _RawGlyphSet:
    def __init__(self, table):
        self.table = table

    def __getitem__(self, name):
        return SimpleNamespace(
            draw=partial(self.table[name].draw, glyfTable=self.table)
        )


class CurvePen(BasePen):
    """Flatten quadratic outlines with a bounded L-infinity approximation."""

    def __init__(self, glyph_set=None):
        """Initialize a quadratic-outline collector."""
        super().__init__(glyph_set)
        self.skipMissingComponents = False
        self.paths = []

    def _moveTo(self, point):  # noqa: N802
        self.paths.append([np.asarray(point, dtype=float)])

    def _lineTo(self, point):  # noqa: N802
        self.paths[-1].append(np.asarray(point, dtype=float))

    def _qCurveToOne(self, control, end):  # noqa: N802
        def flatten(first, middle, last, depth=0):
            if depth > 24:
                raise ValueError("Curve subdivision exceeds the recovery limit")
            if np.max(np.abs(first - 2 * middle + last)) / 4 <= CURVE_ERROR:
                self.paths[-1].append(last)
                return
            left, right = (first + middle) / 2, (middle + last) / 2
            center = (left + right) / 2
            flatten(first, left, center, depth + 1)
            flatten(center, right, last, depth + 1)

        flatten(
            np.asarray(self._getCurrentPoint()),
            np.asarray(control, dtype=float),
            np.asarray(end, dtype=float),
        )

    def _curveToOne(self, *_):  # noqa: N802
        raise ValueError("Cubic outlines are unsupported")

    def _closePath(self):  # noqa: N802
        self.paths[-1].append(self.paths[-1][0])

    def _endPath(self):  # noqa: N802
        raise ValueError("Open outlines are unsupported")


@dataclass(eq=False)
class Contour:
    """A sampled contour, preserving its winding and position within the glyph."""

    points: np.ndarray
    tree: cKDTree
    sign: int
    bounds: np.ndarray


@dataclass(eq=False)
class Outline:
    """Physical glyph geometry independent of PDF font names and glyph numbers."""

    key: str
    points: list
    parts: list[Contour]
    center: np.ndarray
    bounds: np.ndarray
    mark_centers: np.ndarray


def read_outline(font, gid):
    """Read a nonempty TrueType glyph; reject unknown glyphs and unsupported units."""
    if "glyf" not in font or font["head"].unitsPerEm != UNITS:
        raise ValueError("Expected quadratic outlines with 2048 units per em")
    key = outline_key(font, gid)
    if key is None:
        return None
    pen = CurvePen(_RawGlyphSet(font["glyf"]))
    font["glyf"][font.getGlyphName(gid)].draw(pen, font["glyf"])
    paths = [np.asarray(path) for path in pen.paths]
    joined = np.concatenate(paths)
    low, high = joined.min(axis=0), joined.max(axis=0)
    center = (low + high) / 2
    parts = []
    for path in paths:
        samples = []
        for first, last in pairwise(path):
            count = max(1, math.ceil(np.max(np.abs(last - first)) / STEP))
            samples.append(np.linspace(first, last, count + 1))
        points = np.concatenate(samples) - center
        area = np.sum(path[:-1, 0] * path[1:, 1] - path[1:, 0] * path[:-1, 1])
        if abs(area) < 1e-8:
            raise ValueError("Degenerate contour")
        parts.append(
            Contour(
                points,
                cKDTree(points),
                int(np.sign(area)),
                np.concatenate([points.min(axis=0), points.max(axis=0)]),
            )
        )
    raw = contours(font, gid)
    centers = np.asarray(
        [
            (np.asarray(c)[:, :2].min(axis=0) + np.asarray(c)[:, :2].max(axis=0)) / 2
            for c in raw
        ]
    )
    return Outline(
        key, raw, parts, center, np.concatenate([low - center, high - center]), centers
    )


def contour_distance(first, second):
    """Return the symmetric sampled Hausdorff distance without moving contours."""
    if first.sign != second.sign:
        return math.inf
    return float(
        max(
            second.tree.query(first.points, p=np.inf)[0].max(),
            first.tree.query(second.points, p=np.inf)[0].max(),
        )
    )


def outline_distance(first, second):
    """Bound every contour under a one-to-one assignment and common glyph origin."""
    if len(first.parts) != len(second.parts) or len(first.parts) > 6:
        return math.inf
    if np.max(np.abs(first.bounds - second.bounds)) > BOUND + 2 * CURVE_ERROR:
        return math.inf
    costs = [
        [
            contour_distance(a, b)
            if np.max(np.abs(a.bounds - b.bounds)) <= BOUND + 2 * CURVE_ERROR
            else math.inf
            for b in second.parts
        ]
        for a in first.parts
    ]
    distance = min(
        max(costs[i][j] for i, j in enumerate(order))
        for order in permutations(range(len(second.parts)))
    )
    # Include both curve approximations and a conservative sampling allowance.
    return distance + STEP + 2 * CURVE_ERROR


def mark_variant(first, second):
    """Check fixed dot resizing while preserving relative contour centers."""
    if (
        not first.points
        or len(first.points) != len(second.points)
        or len(first.points) > 4
    ):
        return False
    for order in permutations(range(len(second.points))):
        delta = second.mark_centers[list(order)] - first.mark_centers
        if np.max(np.abs(delta - delta.mean(axis=0))) > MARK_ERROR:
            continue
        good = True
        for i, j in enumerate(order):
            a, b = np.asarray(first.points[i]), np.asarray(second.points[j])
            if len(a) != len(b):
                good = False
                break
            found = False
            for offset in range(len(b)):
                rotated = np.roll(b, offset, axis=0)
                if not np.array_equal(a[:, 2], rotated[:, 2]):
                    continue
                if any(
                    np.max(
                        np.abs(
                            a[:, :2]
                            - first.mark_centers[i]
                            - scale * (rotated[:, :2] - second.mark_centers[j])
                        )
                    )
                    <= MARK_ERROR
                    for scale in (1, 0.9)
                ):
                    found = True
                    break
            if not found:
                good = False
                break
        if good:
            return True
    return False


def word_bound(deltas, errors):
    """Minimize the common-translation bound, including each glyph's own error."""
    if (
        not len(deltas)
        or not np.isfinite(deltas).all()
        or not np.isfinite(errors).all()
    ):
        return math.inf
    errors = np.asarray(errors)
    allowance = BOUND - errors
    low = (deltas - allowance[:, None]).max(axis=0)
    high = (deltas + allowance[:, None]).min(axis=0)
    translation = (low + high) / 2
    return float(np.max(np.max(np.abs(deltas - translation), axis=1) + errors))


class PositionedReferenceFont:
    """Invert source outlines and check all candidate letters and word positions."""

    def __init__(self, path):
        """Load reference outlines and cache candidate matches."""
        self.reference = ReferenceFont(path)
        with TTFont(path) as font:
            self.outlines = {
                gid: outline
                for gid in self.reference.meanings
                if (outline := read_outline(font, gid)) is not None
            }
        self.marks = {
            gid
            for gid, meanings in self.reference.meanings.items()
            if all(all(index > 0 for _, index, _ in meaning) for meaning in meanings)
        }
        self._sources = {outline.key: outline for outline in self.outlines.values()}
        self._matches = {}
        self.shape = lru_cache(maxsize=16384)(self.shape)

    def source_outline(self, font, gid):
        """Resolve a glyph by its outline, allowing identically drawn PDF subsets."""
        if "glyf" not in font or font["head"].unitsPerEm != UNITS:
            raise ValueError("Expected quadratic outlines with 2048 units per em")
        key = outline_key(font, gid)
        if key is None:
            return None
        if key not in self._sources:
            self._sources[key] = read_outline(font, gid)
        return self._sources[key]

    def matches(self, source):
        """Retain every reference candidate under the same fixed matching policy."""
        if source.key not in self._matches:
            values = {}
            for gid, target in self.outlines.items():
                marked = gid in self.marks and mark_variant(source, target)
                if source.key == target.key:
                    error = 0
                elif marked:
                    error = 2 * MARK_ERROR
                else:
                    error = outline_distance(source, target)
                if error <= BOUND:
                    values[gid] = (error, marked)
            self._matches[source.key] = values
        return self._matches[source.key]

    def shape(self, text):
        """Shape Unicode into reference glyph identities and physical origins."""
        import uharfbuzz as hb

        buffer = hb.Buffer()
        buffer.add_str(text)
        buffer.guess_segment_properties()
        buffer.language = "ur"
        hb.shape(self.reference.font, buffer)
        x = y = 0
        result = []
        for info, position in zip(
            buffer.glyph_infos, buffer.glyph_positions, strict=True
        ):
            result.append(
                (info.codepoint, x + position.x_offset, y + position.y_offset)
            )
            x += position.x_advance
            y += position.y_advance
        return tuple(result)

    def decode(self, glyphs):
        """Return all word candidates within the bound; never choose an ambiguity."""
        glyphs = tuple(glyphs)
        if not glyphs or any(glyph is None for glyph, _, _ in glyphs):
            return ()
        choices = [self.matches(glyph) for glyph, _, _ in glyphs]
        values = {()}
        for candidates in reversed(choices):
            meanings = {
                meaning
                for gid in candidates
                for meaning in self.reference.meanings[gid]
            }
            values = combine(values, meanings)
            if not values:
                return ()
        texts = {text for value in values if (text := complete_text(value))}
        accepted = []
        for text in sorted(texts):
            shaped = self.shape(text)
            if len(shaped) != len(glyphs):
                continue
            deltas, errors = [], []
            for (source, x, y), candidates, (gid, rx, ry) in zip(
                glyphs, choices, shaped, strict=True
            ):
                if gid not in candidates:
                    break
                error, marked = candidates[gid]
                target = self.outlines[gid]
                a = source.mark_centers.mean(axis=0) if marked else source.center
                b = target.mark_centers.mean(axis=0) if marked else target.center
                deltas.append(np.asarray([x - rx, y - ry]) + a - b)
                errors.append(error)
            else:
                if word_bound(np.asarray(deltas), errors) <= BOUND:
                    accepted.append(text)
        return tuple(accepted)


def _cluster_contour(points, sign):
    return Contour(
        points,
        cKDTree(points),
        sign,
        np.concatenate([points.min(axis=0), points.max(axis=0)]),
    )


def _cluster_distance(first, second):
    if not first or len(first) != len(second) or len(first) > 6:
        return math.inf
    costs = [
        [
            contour_distance(a, b)
            if np.max(np.abs(a.bounds - b.bounds)) <= BOUND
            else math.inf
            for b in second
        ]
        for a in first
    ]
    return (
        min(
            max(costs[i][j] for i, j in enumerate(order))
            for order in permutations(range(len(second)))
        )
        + STEP
        + 2 * CURVE_ERROR
    )


class LabelReferenceFont(PositionedReferenceFont):
    """Match fixed printed labels whose integrated dot cluster was resized.

    The cluster uses only scales 1 or 0.9, with an eight-unit fitting bound.
    Contour count and winding, the letter body and the common word-position
    bound remain enforced. Use this extension only for known field labels;
    personal names retain the base decoder's candidate policy.
    """

    def __init__(self, path):
        """Load label outlines and precompute allowed mark clusters."""
        super().__init__(path)
        self.patterns = []
        self._classes = {}
        self._clusters = {}
        self._mixed = {}
        for gid in self.marks:
            if gid not in self.outlines:
                continue
            for p in self.outlines[gid].parts:
                center = (p.bounds[:2] + p.bounds[2:]) / 2
                for scale in (1, 0.9):
                    self.patterns.append(
                        _cluster_contour((p.points - center) * scale, p.sign)
                    )

    def classify(self, outline):
        """Split one outline into letter-body and mark contours."""
        if outline.key not in self._classes:
            marks = []
            bodies = []
            for p in outline.parts:
                center = (p.bounds[:2] + p.bounds[2:]) / 2
                normalized = _cluster_contour(p.points - center, p.sign)
                matched = any(
                    np.max(np.abs(normalized.bounds - q.bounds)) <= BOUND
                    and contour_distance(normalized, q) + STEP + 2 * CURVE_ERROR
                    <= BOUND
                    for q in self.patterns
                )
                (marks if matched else bodies).append(p)
            self._classes[outline.key] = (bodies, marks)
        return self._classes[outline.key]

    def cluster(self, outline, scale=1):
        """Return a centered mark cluster at the requested scale."""
        key = (outline.key, scale)
        if key not in self._clusters:
            _, marks = self.classify(outline)
            cloud = np.concatenate([p.points for p in marks])
            center = (cloud.min(axis=0) + cloud.max(axis=0)) / 2
            self._clusters[key] = (
                center,
                [_cluster_contour((p.points - center) * scale, p.sign) for p in marks],
            )
        return self._clusters[key]

    def mixed_distance(self, source, target):
        """Measure a match that permits the label's known mark scaling."""
        if len(source.parts) != len(target.parts):
            return math.inf
        a, am = self.classify(source)
        b, bm = self.classify(target)
        if not a or not am or len(a) != len(b) or len(am) != len(bm):
            return math.inf
        body = _cluster_distance(a, b)
        if body > BOUND:
            return math.inf
        ac, ap = self.cluster(source)
        for scale in (1, 0.9):
            bc, bp = self.cluster(target, scale)
            delta = float(np.max(np.abs(ac - bc)))
            fit = _cluster_distance(ap, bp)
            if fit <= 8 and delta + fit <= BOUND:
                return max(body, delta + fit)
        return math.inf

    def matches(self, source):
        """Return ordinary and label-specific matches for a source outline."""
        if source.key not in self._mixed:
            values = dict(super().matches(source))
            for gid, target in self.outlines.items():
                if gid in values and values[gid][0] == 0:
                    continue
                error = self.mixed_distance(source, target)
                if error <= BOUND and (gid not in values or error < values[gid][0]):
                    values[gid] = (error, False)
            self._mixed[source.key] = values
        return self._mixed[source.key]
