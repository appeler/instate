"""Regressions for conservative reuse of embedded Hindi glyph mappings."""

from types import SimpleNamespace

from model_training.prep_er_data.hindi_glyphs import (
    finalize_catalog,
    glyph_fingerprint,
    glyph_groups,
    recover_group,
    usable_unicode,
)


def test_outline_identity_ignores_gid_but_requires_exact_geometry_and_units():
    class Font(dict):
        def getGlyphOrder(self):
            return [".notdef", "first", "same", "different", "blank"]

        def getGlyphName(self, gid):
            return self.getGlyphOrder()[gid]

    def glyph(x):
        return SimpleNamespace(
            numberOfContours=1,
            getCoordinates=lambda _: ([(0, 0), (x, 10)], [1], [1, 1]),
        )

    font = Font(
        head=SimpleNamespace(unitsPerEm=2048),
        glyf={
            ".notdef": glyph(10),
            "first": glyph(10),
            "same": glyph(10),
            "different": glyph(11),
            "blank": SimpleNamespace(numberOfContours=0),
        },
    )
    fingerprint = glyph_fingerprint(font, 1)
    assert fingerprint == glyph_fingerprint(font, 2)
    assert fingerprint != glyph_fingerprint(font, 3)
    assert glyph_fingerprint(font, 0) is None
    assert glyph_fingerprint(font, 4) is None
    assert glyph_fingerprint(font, 5) is None
    font["head"].unitsPerEm = 1024
    assert fingerprint != glyph_fingerprint(font, 1)


def test_virtual_continuations_remain_attached_to_physical_glyph():
    chars = [
        (ord("क"), 200, (1, 2), (1, 2, 3, 4)),
        (ord("्"), -1, (1, 2), (1, 2, 3, 4)),
        (ord("ष"), 159, (3, 2), (3, 2, 4, 4)),
    ]
    groups = glyph_groups(chars)
    assert [(group["gid"], group["text"]) for group in groups] == [
        (200, "क्"),
        (159, "ष"),
    ]


def test_conflicting_outline_labels_cannot_supply_a_recovery():
    mapping, conflicts = finalize_catalog({"same": {"क्", "ख्"}, "unique": {"क्ष"}})
    assert mapping == {"unique": "क्ष"}
    assert conflicts == {"same": ["क्", "ख्"]}
    assert recover_group({"text": "�"}, "same", {"mapping": mapping}) == (
        "�",
        "unresolved",
    )


def test_replacement_and_legacy_latin_are_both_damage():
    assert not usable_unicode("�")
    assert not usable_unicode("à")
    assert usable_unicode("ख्")
    assert usable_unicode("2018")
    assert usable_unicode(" ")


def test_recovery_requires_exact_outline_evidence():
    catalog = {"mapping": {"verified-outline": "क्"}}
    assert recover_group({"text": "�"}, "verified-outline", catalog) == (
        "क्",
        "matched_outline",
    )
    assert recover_group({"text": "à"}, "different-outline", catalog) == (
        "�",
        "unresolved",
    )


def test_valid_source_mapping_is_preserved_even_if_catalog_disagrees():
    assert recover_group({"text": "ख्"}, "outline", {"mapping": {"outline": "क्"}}) == (
        "ख्",
        "source_mapping",
    )
