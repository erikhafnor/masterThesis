"""Fleet registry for Elisa 800 ventilators at Helse Stavanger.

Hardcoded, human-verified mapping from CMMS (maintenance system) to
telemetry serial numbers. The telemetry serial is the last 7 digits of
the CMMS serial, zero-padded to 15 characters.

Source: maintenance_log.csv (CMMS export).
"""

from __future__ import annotations

from datetime import date
from typing import TypedDict


class DeviceInfo(TypedDict):
    """One row of the fleet registry, linking CMMS to telemetry identifiers.

    Attributes:
        cmms_reg: CMMS registration number as used in the maintenance export.
        cmms_serial: Full CMMS serial string as it appears in the maintenance export.
    """

    cmms_reg: int
    cmms_serial: str


class FailureEvent(TypedDict):
    """One logged O2-sensor failure event from the maintenance record.

    Attributes:
        telemetry_serial: Zero-padded 15-character telemetry identifier for the device.
        cmms_reg: CMMS registration number of the affected device.
        date: Date the failure was logged in the maintenance system.
        description: Free-text failure summary as recorded in the maintenance log.
        fault_code: Elisa 800 fault code string (e.g. ``"#259"``), or ``None`` if
            the maintenance record does not include a fault code.
    """

    telemetry_serial: str
    cmms_reg: int
    date: date
    description: str
    fault_code: str | None


class PVEvent(TypedDict):
    """One scheduled preventive-maintenance event from the CMMS record.

    Attributes:
        telemetry_serial: Zero-padded 15-character telemetry identifier for the device.
        cmms_reg: CMMS registration number of the serviced device.
        date: Date the PV service was recorded in the CMMS.
    """

    telemetry_serial: str
    cmms_reg: int
    date: date


def _cmms_to_telemetry(cmms_serial: str) -> str:
    """Extract last 7 digits of CMMS serial, zero-pad to 15 chars."""
    digits = "".join(c for c in cmms_serial if c.isdigit())
    return digits[-7:].zfill(15)


# ── Fleet registry: telemetry_serial → device info ─────────────────────
# 30 verified Elisa 800 devices.
_RAW_FLEET: list[tuple[int, str]] = [
    (18814, "0408810hul08204503"),
    (18815, "0408810hul08204504"),
    (18816, "0408810hul08204505"),
    (18817, "0408810hul08204506"),
    (18818, "0408810hul08204509"),
    (18819, "0408810hul08204510"),
    (18820, "0408810hul08204511"),
    (18821, "0408810hul08204512"),
    (18822, "0408810hul08204513"),
    (18823, "0408810hul08204514"),
    (18824, "0408810hul08204515"),
    (18825, "0408810hul08204516"),
    (18826, "0408810hul08204517"),
    (18827, "0408810hul08204518"),
    (18828, "0408810hul08204519"),
    (18829, "0408810hul08204520"),
    (18830, "0408810hul08204521"),
    (18831, "0408810hul08204522"),
    (18832, "0408810hul08204523"),
    (18833, "0408810hul08204524"),
    (18834, "0408810hul08204525"),
    (18835, "0408810hul08204526"),
    (18836, "0408810hul08204527"),
    (18837, "0408810hul08204528"),
    (18838, "0408810hul08204529"),
    (18839, "0408810hul08204570"),
    (18840, "0408810hul08204571"),
    (18841, "0408810hul08204572"),
    (18843, "08209508"),
    (18844, "08209509"),
]

#: Mapping from telemetry serial to :class:`DeviceInfo` for the 30 verified Elisa 800 devices.
FLEET_REGISTRY: dict[str, DeviceInfo] = {
    _cmms_to_telemetry(cmms_serial): DeviceInfo(
        cmms_reg=reg, cmms_serial=cmms_serial
    )
    for reg, cmms_serial in _RAW_FLEET
}

#: Precomputed set of all 30 telemetry serials drawn from :data:`FLEET_REGISTRY`; use for O(1) membership checks in ingestion filters.
FLEET_SERIALS: set[str] = set(FLEET_REGISTRY.keys())

#: Inverse lookup from CMMS registration number to telemetry serial.
REG_TO_SERIAL: dict[int, str] = {
    info["cmms_reg"]: serial for serial, info in FLEET_REGISTRY.items()
}


# ── Known O2 sensor failures (within telemetry observation period) ──────
#: Logged O2-sensor failures observed within the telemetry collection window (Dec 2025 – Jan 2026).
KNOWN_FAILURES: list[FailureEvent] = [
    FailureEvent(
        telemetry_serial="000000008204514",
        cmms_reg=18823,
        date=date(2025, 12, 15),
        description="FiO2 drift: set 0.40, measured 0.36",
        fault_code=None,
    ),
    FailureEvent(
        telemetry_serial="000000008204506",
        cmms_reg=18817,
        date=date(2025, 12, 18),
        description="#259 sensor fault during system test",
        fault_code="#259",
    ),
    FailureEvent(
        telemetry_serial="000000008204570",
        cmms_reg=18839,
        date=date(2026, 1, 9),
        description="#259 O2 flow fault",
        fault_code="#259",
    ),
    FailureEvent(
        telemetry_serial="000000008204524",
        cmms_reg=18833,
        date=date(2026, 1, 26),
        description="#259 sensor fault",
        fault_code="#259",
    ),
]

#: Set of telemetry serials that appear in :data:`KNOWN_FAILURES`.
FAILURE_SERIALS: set[str] = {f["telemetry_serial"] for f in KNOWN_FAILURES}


# ── PV (scheduled maintenance) events Dec 2025 ─────────────────────────
#: Scheduled preventive-maintenance (PV) events recorded in Dec 2025, covering 28 of 30 fleet devices. Used as a negative control in `pdm.evaluation_pv`.
PV_EVENTS: list[PVEvent] = [
    PVEvent(telemetry_serial="000000008204504", cmms_reg=18815, date=date(2025, 12, 10)),
    PVEvent(telemetry_serial="000000008204505", cmms_reg=18816, date=date(2025, 12, 9)),
    PVEvent(telemetry_serial="000000008204506", cmms_reg=18817, date=date(2025, 12, 8)),
    PVEvent(telemetry_serial="000000008204509", cmms_reg=18818, date=date(2025, 12, 10)),
    PVEvent(telemetry_serial="000000008204510", cmms_reg=18819, date=date(2025, 12, 8)),
    PVEvent(telemetry_serial="000000008204511", cmms_reg=18820, date=date(2025, 12, 10)),
    PVEvent(telemetry_serial="000000008204512", cmms_reg=18821, date=date(2025, 12, 11)),
    PVEvent(telemetry_serial="000000008204514", cmms_reg=18823, date=date(2025, 12, 10)),
    PVEvent(telemetry_serial="000000008204515", cmms_reg=18824, date=date(2025, 12, 9)),
    PVEvent(telemetry_serial="000000008204516", cmms_reg=18825, date=date(2025, 12, 8)),
    PVEvent(telemetry_serial="000000008204517", cmms_reg=18826, date=date(2025, 12, 11)),
    PVEvent(telemetry_serial="000000008204518", cmms_reg=18827, date=date(2025, 12, 10)),
    PVEvent(telemetry_serial="000000008204519", cmms_reg=18828, date=date(2025, 12, 10)),
    PVEvent(telemetry_serial="000000008204520", cmms_reg=18829, date=date(2025, 12, 10)),
    PVEvent(telemetry_serial="000000008204521", cmms_reg=18830, date=date(2025, 12, 10)),
    PVEvent(telemetry_serial="000000008204522", cmms_reg=18831, date=date(2025, 12, 10)),
    PVEvent(telemetry_serial="000000008204523", cmms_reg=18832, date=date(2025, 12, 10)),
    PVEvent(telemetry_serial="000000008204524", cmms_reg=18833, date=date(2025, 12, 9)),
    PVEvent(telemetry_serial="000000008204525", cmms_reg=18834, date=date(2025, 12, 9)),
    PVEvent(telemetry_serial="000000008204526", cmms_reg=18835, date=date(2025, 12, 9)),
    PVEvent(telemetry_serial="000000008204527", cmms_reg=18836, date=date(2025, 12, 11)),
    PVEvent(telemetry_serial="000000008204528", cmms_reg=18837, date=date(2025, 12, 11)),
    PVEvent(telemetry_serial="000000008204529", cmms_reg=18838, date=date(2025, 12, 12)),
    PVEvent(telemetry_serial="000000008204570", cmms_reg=18839, date=date(2025, 12, 8)),
    PVEvent(telemetry_serial="000000008204571", cmms_reg=18840, date=date(2025, 12, 8)),
    PVEvent(telemetry_serial="000000008204572", cmms_reg=18841, date=date(2025, 12, 9)),
    PVEvent(telemetry_serial="000000008209508", cmms_reg=18843, date=date(2025, 12, 10)),
    PVEvent(telemetry_serial="000000008209509", cmms_reg=18844, date=date(2025, 12, 10)),
]
