"""The nndct/brevitas probes must work without the optional qat extra."""

from __future__ import annotations

from cpsa_ml.qat import NndctProbe, brevitas_available, try_nndct


def test_try_nndct_never_raises() -> None:
    probe = try_nndct()
    assert isinstance(probe, NndctProbe)
    assert bool(probe) == probe.available
    assert probe.reason


def test_brevitas_available_is_boolean() -> None:
    assert isinstance(brevitas_available(), bool)
