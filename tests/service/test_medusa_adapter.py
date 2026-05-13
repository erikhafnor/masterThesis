import pytest
from pathlib import Path
from datetime import date
from openpyxl import Workbook
from ventilator_pdm.service.medusa_adapter import parse_medusa_excel

COLUMNS = [
    "AO-nr.", "AO-type", "Reg. dato", "Ferdig",
    "Oppgave/feilbeskrivelse", "Teknisk beskrivelse", "Reg.nr.", "Serienr."
]


def _create_test_excel(path: Path, rows: list[list]) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.append(COLUMNS)
    for row in rows:
        ws.append(row)
    wb.save(path)
    return path


def test_parse_single_corrective_work_order(tmp_path):
    xlsx = _create_test_excel(tmp_path / "test.xlsx", [
        ["AO-001", "Korrektiv", "2026-01-09", "2026-01-10",
         "O2 sensor feil #259", "Byttet O2 sensor", 18839, "8204570"],
    ])
    records = parse_medusa_excel(xlsx)

    assert len(records) == 1
    r = records[0]
    assert r["work_order_id"] == "AO-001"
    assert r["work_order_type"] == "corrective"
    assert r["registered_date"] == date(2026, 1, 9)
    assert r["fault_description"] == "O2 sensor feil #259"
    assert r["device_reg"] == 18839


def test_parse_preventive_work_order(tmp_path):
    xlsx = _create_test_excel(tmp_path / "test.xlsx", [
        ["AO-002", "Forebyggende", "2025-12-08", "2025-12-08",
         "PV kontroll", "Forebyggende vedlikehold", 18817, "8204506"],
    ])
    records = parse_medusa_excel(xlsx)

    assert len(records) == 1
    assert records[0]["work_order_type"] == "preventive"


def test_parse_multiple_rows(tmp_path):
    xlsx = _create_test_excel(tmp_path / "test.xlsx", [
        ["AO-001", "Korrektiv", "2026-01-09", "2026-01-10",
         "Feil", "Beskrivelse", 18839, "8204570"],
        ["AO-002", "Forebyggende", "2025-12-08", "2025-12-08",
         "PV", "Kontroll", 18817, "8204506"],
    ])
    records = parse_medusa_excel(xlsx)
    assert len(records) == 2


def test_parse_maps_device_reg_to_serial(tmp_path):
    xlsx = _create_test_excel(tmp_path / "test.xlsx", [
        ["AO-001", "Korrektiv", "2026-01-09", "2026-01-10",
         "Feil", "Beskrivelse", 18839, "8204570"],
    ])
    records = parse_medusa_excel(xlsx)
    assert records[0]["device_serial"] is not None
    assert records[0]["device_serial"] == "000000008204570"


def test_parse_empty_excel(tmp_path):
    xlsx = _create_test_excel(tmp_path / "test.xlsx", [])
    records = parse_medusa_excel(xlsx)
    assert records == []


def test_parse_handles_date_objects(tmp_path):
    """Excel often stores dates as datetime objects, not strings."""
    from datetime import datetime
    xlsx = _create_test_excel(tmp_path / "test.xlsx", [
        ["AO-001", "Korrektiv", datetime(2026, 1, 9), datetime(2026, 1, 10),
         "Feil", "Beskrivelse", 18839, "8204570"],
    ])
    records = parse_medusa_excel(xlsx)
    assert records[0]["registered_date"] == date(2026, 1, 9)
