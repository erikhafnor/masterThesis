"""Tests for fleet registry."""

from datetime import date

from ventilator_pdm.registry import (
    FLEET_REGISTRY,
    FLEET_SERIALS,
    KNOWN_FAILURES,
    PV_EVENTS,
    REG_TO_SERIAL,
    _cmms_to_telemetry,
)


def test_fleet_size():
    assert len(FLEET_REGISTRY) == 30
    assert len(FLEET_SERIALS) == 30


def test_telemetry_serial_format():
    for serial in FLEET_SERIALS:
        assert len(serial) == 15, f"Serial {serial} not 15 chars"
        assert serial.isdigit(), f"Serial {serial} not all digits"


def test_cmms_to_telemetry_mapping():
    # Spot-check known mappings
    assert _cmms_to_telemetry("0408810hul08204503") == "000000008204503"
    assert _cmms_to_telemetry("0408810hul08204514") == "000000008204514"
    assert _cmms_to_telemetry("08209508") == "000000008209508"
    assert _cmms_to_telemetry("08209509") == "000000008209509"


def test_reg_to_serial_coverage():
    expected_regs = {
        18814, 18815, 18816, 18817, 18818, 18819, 18820, 18821, 18822,
        18823, 18824, 18825, 18826, 18827, 18828, 18829, 18830, 18831,
        18832, 18833, 18834, 18835, 18836, 18837, 18838, 18839, 18840,
        18841, 18843, 18844,
    }
    assert set(REG_TO_SERIAL.keys()) == expected_regs


def test_known_failures():
    assert len(KNOWN_FAILURES) == 4
    failure_serials = {f["telemetry_serial"] for f in KNOWN_FAILURES}
    # All failure devices must be in fleet
    assert failure_serials <= FLEET_SERIALS

    # Check specific failures
    regs = {f["cmms_reg"] for f in KNOWN_FAILURES}
    assert regs == {18823, 18817, 18839, 18833}

    # Check dates are in observation period
    for f in KNOWN_FAILURES:
        assert f["date"] >= date(2025, 11, 17)


def test_known_failure_details():
    by_reg = {f["cmms_reg"]: f for f in KNOWN_FAILURES}

    assert by_reg[18823]["date"] == date(2025, 12, 15)
    assert by_reg[18823]["telemetry_serial"] == "000000008204514"

    assert by_reg[18817]["date"] == date(2025, 12, 18)
    assert by_reg[18817]["fault_code"] == "#259"

    assert by_reg[18839]["date"] == date(2026, 1, 9)
    assert by_reg[18833]["date"] == date(2026, 1, 26)


def test_pv_events():
    assert len(PV_EVENTS) > 0
    for pv in PV_EVENTS:
        assert pv["telemetry_serial"] in FLEET_SERIALS
        assert pv["date"].month == 12
        assert pv["date"].year == 2025


def test_no_duplicate_serials():
    serials = [info["cmms_serial"] for info in FLEET_REGISTRY.values()]
    assert len(serials) == len(set(serials))
