"""Curated variable taxonomy for Elisa 800 ventilator telemetry.

Variables are organized into tiers by relevance to O2 sensor fault detection.
Variable IDs correspond to the ``variable_id`` column in QuestDB. Tiers are:

- **Tier 1** – O2 fault signature variables used as primary model features.
- **Tier 2** – General ventilation variables included as supporting features.
- **Tier 3** – Context / filtering variables (standby, mode, calibration) that
  are typically excluded from anomaly models but used for data gating.

Exported constants follow this naming pattern:

- ``TIER<N>_*`` – ordered lists of :class:`Variable` for each tier.
- ``*_IDS`` – corresponding ``set[int]`` of variable IDs for fast membership
  tests.
- ``FEATURE_IDS`` – the union of Tier 1 and Tier 2 IDs used as model inputs.
- ``FIO2_MEASURED_ID``, ``FIO2_SETTING_ID``, ``STANDBY_ID`` – scalar IDs for
  the three most-referenced individual signals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variable:
    """Immutable descriptor for a single Elisa 800 telemetry variable.

    Attributes:
        id: Numeric ``variable_id`` stored in QuestDB (matches the value in
            the ``variable_id`` column of the raw telemetry table).
        name: Snake-case signal name used as a column label after pivoting.
        unit: Physical unit string (e.g. ``"%"``, ``"kPa"``, ``"bool"``).
        tier: Tier classification (1 = O2 fault signature, 2 = general
            ventilation, 3 = context / filtering).
        description: Human-readable description of the signal.
    """

    id: int
    name: str
    unit: str
    tier: int
    description: str


# ── Tier 1: O2 fault signature (~14 variables) ─────────────────────────
#: Primary feature set for O2 sensor fault detection. These variables directly
#: reflect O2 delivery, supply pressure, sensor alarms, and FiO2 readings.
TIER1_O2_FAULT: list[Variable] = [
    Variable(635,  "fio2_measured",        "%",     1, "Measured FiO2"),
    Variable(2782, "fio2_setting",         "%",     1, "FiO2 setpoint"),
    Variable(2098, "o2_supply_pressure",   "kPa",   1, "O2 supply pressure"),
    Variable(2097, "air_supply_pressure",  "kPa",   1, "Air supply pressure"),
    Variable(4024, "o2_flow_rate",         "L/min", 1, "O2 flow rate"),
    Variable(4113, "air_flow_rate",        "L/min", 1, "Air flow rate"),
    Variable(1817, "o2_pressure_alarm",    "bool",  1, "O2 pressure alarm active"),
    Variable(2046, "flow_sensor_alarm",    "bool",  1, "Flow sensor alarm active"),
    Variable(7629, "o2_sensor_disconnect", "bool",  1, "O2 sensor disconnected"),
    Variable(8024, "circuit_o2",           "%",     1, "Circuit O2 concentration"),
    Variable(6501, "o2_replacement",       "bool",  1, "O2 sensor replacement needed"),
    Variable(8596, "low_fio2_alarm",       "bool",  1, "Low FiO2 alarm"),
    Variable(8597, "high_fio2_alarm",      "bool",  1, "High FiO2 alarm"),
    Variable(8625, "high_o2_supply_alarm", "bool",  1, "High O2 supply pressure alarm"),
]

# ── Tier 2: General ventilation (~12 variables) ────────────────────────
#: Supporting ventilation mechanics features — pressures, volumes, flow rates,
#: compliance, resistance, and PEEP. Included alongside Tier 1 as model inputs.
TIER2_VENTILATION: list[Variable] = [
    Variable(1414, "peak_airway_pressure",  "cmH2O",  2, "Peak inspiratory pressure"),
    Variable(1415, "mean_airway_pressure",  "cmH2O",  2, "Mean airway pressure"),
    Variable(2324, "insp_tidal_volume",     "mL",     2, "Inspiratory tidal volume"),
    Variable(2325, "exp_tidal_volume",      "mL",     2, "Expiratory tidal volume"),
    Variable(2326, "exp_minute_volume",     "L/min",  2, "Expiratory minute volume"),
    Variable(2092, "insp_minute_volume",    "L/min",  2, "Inspiratory minute volume"),
    Variable(1314, "spont_resp_rate",       "1/min",  2, "Spontaneous respiratory rate"),
    Variable(5945, "total_resp_rate",       "1/min",  2, "Total respiratory rate"),
    Variable(1310, "compliance",            "mL/cmH2O", 2, "Lung compliance"),
    Variable(1319, "resistance",            "cmH2O/(L/s)", 2, "Airway resistance (R)"),
    Variable(1761, "airway_resistance",     "cmH2O/(L/s)", 2, "Airway resistance (Raw)"),
    Variable(2776, "total_peep",            "cmH2O",  2, "Total PEEP"),
]

# ── Tier 3: Context / filtering ────────────────────────────────────────
#: Contextual / gating variables. Rows where ``standby_status`` is active or
#: ``calibration`` is in progress are typically excluded before model scoring.
TIER3_CONTEXT: list[Variable] = [
    Variable(1889, "standby_status", "bool", 3, "Device in standby mode"),
    Variable(584,  "vent_mode",      "enum", 3, "Ventilation mode"),
    Variable(386,  "calibration",    "bool", 3, "Calibration in progress"),
]

# ── Bitfield source variables (packed integers decoded during ingestion) ──
#: Variable IDs (801–804) that carry packed integer bitfields. ``parsing.py``
#: decodes these into the individual boolean ``bitfield_*`` columns at
#: ingestion time. They must be included in the QuestDB extraction query so
#: that the decoded boolean columns are populated in the wide-format DataFrame.
BITFIELD_SOURCE_IDS: set[int] = {801, 802, 803, 804}

# ── Bitfield columns (present as boolean columns in telemetry) ──────────
#: Ordered list of decoded boolean column names produced by ``parsing.py``
#: from the packed bitfield source variables. Each entry is ``True`` when the
#: corresponding subsystem self-test passes.
BITFIELDS: list[str] = [
    "bitfield_o2_flow_sensor_ok",
    "bitfield_o2_conc_sensor_ok",
    "bitfield_air_flow_sensor_ok",
    "bitfield_inspiratory_flow_sensor_ok",
    "bitfield_expiratory_flow_sensor_ok",
    "bitfield_airway_pressure_sensor_ok",
    "bitfield_barometric_pressure_ok",
    "bitfield_circuit_o2_cell_checked",
    "bitfield_flow_controls_ok",
    "bitfield_gas_supplies_ok",
    "bitfield_ac_mains_power_ok",
    "bitfield_battery_charge_ok",
    "bitfield_safety_valve_ok",
    "bitfield_exhalation_valve_ok",
    "bitfield_vent_circuit_leak_ok",
    "bitfield_manual_circuit_leak_ok",
    "bitfield_vent_delivery_ok",
]

# ── Convenience collections ─────────────────────────────────────────────
#: Flat list of all :class:`Variable` objects across all tiers (excludes
#: bitfield source IDs which have no ``Variable`` descriptors).
ALL_VARIABLES: list[Variable] = TIER1_O2_FAULT + TIER2_VENTILATION + TIER3_CONTEXT

#: Set of Tier 1 variable IDs for fast membership tests.
TIER1_IDS: set[int] = {v.id for v in TIER1_O2_FAULT}
#: Set of Tier 2 variable IDs for fast membership tests.
TIER2_IDS: set[int] = {v.id for v in TIER2_VENTILATION}
#: Set of Tier 3 variable IDs for fast membership tests.
TIER3_IDS: set[int] = {v.id for v in TIER3_CONTEXT}
#: Complete set of variable IDs to include in QuestDB extraction queries
#: (Tier 1 + Tier 2 + Tier 3 + bitfield source variables).
ALL_VAR_IDS: set[int] = TIER1_IDS | TIER2_IDS | TIER3_IDS | BITFIELD_SOURCE_IDS

#: Lookup from ``variable_id`` integer to its :class:`Variable` descriptor.
#: Only covers Tier 1–3 variables; bitfield source IDs are not present.
VAR_BY_ID: dict[int, Variable] = {v.id: v for v in ALL_VARIABLES}

#: IDs used as anomaly model input features — Tier 1 (O2 fault signature)
#: plus Tier 2 (general ventilation). Tier 3 context variables are excluded.
FEATURE_IDS: set[int] = TIER1_IDS | TIER2_IDS

#: ``variable_id`` for the measured FiO2 reading (signal ``fio2_measured``).
FIO2_MEASURED_ID = 635
#: ``variable_id`` for the clinician-set FiO2 target (signal ``fio2_setting``).
FIO2_SETTING_ID = 2782
#: ``variable_id`` for the device standby flag (signal ``standby_status``).
STANDBY_ID = 1889
