"""Match Medusa work orders to PdM alerts and classify feedback labels.

Join semantics: for each unmatched work order, the matcher looks up the
corresponding device serial via :data:`pdm.registry.REG_TO_SERIAL` and then
queries the alert store for any ``warning``-or-above alerts within
``MATCH_WINDOW_DAYS`` days of the work-order registration date (±7 days by
default).  If at least one alert is found, the nearest one is linked; if none
are found the event is classified as ``"unrelated"`` regardless of the work-
order type.  The resulting feedback label is written to the database and the
work order is marked as matched to prevent duplicate processing.
"""

from datetime import date as date_type, timedelta

from ventilator_pdm.service.database import Database
from ventilator_pdm.registry import REG_TO_SERIAL

MATCH_WINDOW_DAYS = 7


def _classify_work_order(work_order_type: str, fault_description: str) -> str:
    if work_order_type == "corrective":
        return "confirmed_fault"
    if work_order_type == "preventive":
        return "scheduled_pv"
    return "unrelated"


def match_work_orders(db: Database) -> list[dict]:
    """Match unmatched work orders to alerts and insert feedback labels.

    Fetches all work orders not yet matched from the database, then for
    each one resolves the device serial, finds nearby alerts within
    ``MATCH_WINDOW_DAYS``, classifies the event type, writes a feedback
    label, and marks the work order as matched.  Work orders whose
    registration number cannot be resolved in the fleet registry are
    silently skipped.

    Args:
        db: Open :class:`~pdm.service.database.Database` instance used for
            both reading work orders and alerts and writing feedback labels.

    Returns:
        List of result dicts, one per processed work order, each containing:

        - ``work_order_id`` (str): CMMS work-order identifier.
        - ``device_serial`` (str): Resolved device serial number.
        - ``event_type`` (str): One of ``"confirmed_fault"``,
            ``"scheduled_pv"``, or ``"unrelated"``.
        - ``matched_alert`` (bool): Whether a nearby alert was found and
            linked.
    """
    unmatched = db.get_unmatched_work_orders()
    results = []

    for wo in unmatched:
        device_serial = REG_TO_SERIAL.get(wo["device_reg"])
        if not device_serial:
            continue

        # registered_date is stored as an ISO 8601 string in SQLite; parse to date.
        reg_date_raw = wo["registered_date"]
        reg_date: date_type = (
            date_type.fromisoformat(reg_date_raw)
            if isinstance(reg_date_raw, str)
            else reg_date_raw
        )

        nearby_alerts = db.get_alerts(
            device_serial=device_serial,
            from_date=reg_date - timedelta(days=MATCH_WINDOW_DAYS),
            to_date=reg_date + timedelta(days=MATCH_WINDOW_DAYS),
            min_level="warning",
        )

        event_type = _classify_work_order(
            wo["work_order_type"], wo["fault_description"]
        )

        if not nearby_alerts:
            event_type = "unrelated"

        matched_alert_id = nearby_alerts[0]["id"] if nearby_alerts else None

        db.insert_feedback_label(
            device_serial=device_serial,
            event_date=reg_date,
            event_type=event_type,
            work_order_id=wo["work_order_id"],
            matched_alert_id=matched_alert_id,
            source="auto",
        )

        db.mark_work_order_matched(wo["work_order_id"])

        results.append({
            "work_order_id": wo["work_order_id"],
            "device_serial": device_serial,
            "event_type": event_type,
            "matched_alert": matched_alert_id is not None,
        })

    return results
