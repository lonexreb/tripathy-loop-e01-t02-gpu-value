"""Pure-Python checks for the loop's contracts (no torch needed)."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tripathy_arch3.forecaster import ForecasterPatch


def _spec_class():  # wsp_loop imports wsp, which only needs torch inside functions
    from tripathy_arch3.wsp_loop import BANDS, PATCHES, EditSpec

    return BANDS, PATCHES, EditSpec


def test_edit_spec_class_membership() -> None:
    _, _, spec_type = _spec_class()
    features = {"instrument_class": "marketplace_ask", "unchanged_fraction": 0.12}
    equals = spec_type(
        hypothesis="h", feature="instrument_class", op="==", value="marketplace_ask", promote="wide"
    )
    below = spec_type(
        hypothesis="h", feature="unchanged_fraction", op="<", value=0.5, promote="wide"
    )
    at_least = spec_type(
        hypothesis="h", feature="unchanged_fraction", op=">=", value=0.5, promote="narrow"
    )
    assert equals.matches(features)
    assert below.matches(features)
    assert not at_least.matches(features)
    assert spec_type(hypothesis="h", feature="all", op="==", value="", promote="medium").matches({})


def test_edit_spec_rejects_unknown_band_and_extra_fields() -> None:
    _, _, spec_type = _spec_class()
    with pytest.raises(ValidationError):
        spec_type(hypothesis="h", feature="all", op="==", value="", promote="huge")
    with pytest.raises(ValidationError):
        spec_type(hypothesis="h", feature="all", op="==", value="", promote="wide", layer=3)


def test_bands_map_to_valid_repricing_patches() -> None:
    bands, patches, _ = _spec_class()
    assert bands == {"narrow": Decimal("0.5"), "medium": Decimal("1"), "wide": Decimal("2")}
    for name, patch in patches.items():
        assert isinstance(patch, ForecasterPatch)
        assert (patch.method, patch.window_days, patch.width_scale) == (
            "repricing_interval",
            60,
            bands[name],
        )
    lookup = {(date(2026, 9, 1), "a"): patches["narrow"], (date(2026, 9, 1), "b"): patches["wide"]}
    assert len(set(lookup.values())) == 2  # patches are hashable, as _compile_block requires
