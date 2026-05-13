# run_server.py
import json
import logging
import logging.handlers
import sys

from .mllp_receiver import receive_hl7_messages
from .parsing import parse_hl7_string, hl7_timestamp_to_ns, safe_bitfield_key, ALL_BITFIELD_KEYS
from .questdb_writer import QuestDBWriter

def setup_logging():
    """Configure logging to both file (with rotation) and stderr."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Rotating file handler: 5 MB per file, keep 5 backups
    file_handler = logging.handlers.RotatingFileHandler(
        "hl7parser.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Also log to stderr so NSSM captures critical messages
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)

    # Stdout for INFO only (WARNING+ goes to stderr, avoid double-output)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stdout_handler.setFormatter(fmt)
    root.addHandler(stdout_handler)

logger = logging.getLogger(__name__)

def make_handler(writer):
    """Create the HL7 message handler closure with the given QuestDB writer."""

    def handle_hl7_message(hl7_message):
        logger.debug("Raw HL7 message received (%d bytes)", len(hl7_message))

        data = parse_hl7_string(hl7_message)
        if not data:
            logger.debug("Message filtered out (not Elisa 800)")
            return

        for row in data:
            ns_timestamp = hl7_timestamp_to_ns(row["timestamp"])
            fields = {
                "label": row["label"],
                "value": row["value"],
                "value_description": row.get("value_description", row["value"]),
                "unit": row["unit"],
                "unit_short": row.get("unit_short", ""),
                "timestamp_str": row["timestamp"],
            }
            # Prepare bitfield columns
            bitfield_values = {safe_bitfield_key(k): False for k in ALL_BITFIELD_KEYS}
            if row.get("bitfield_status"):
                bitfield = json.loads(row["bitfield_status"])
                for k, v in bitfield.items():
                    bitfield_values[safe_bitfield_key(k)] = v
            for k, v in bitfield_values.items():
                fields[f"bitfield_{k}"] = v

            writer.queue(
                measurement="pdm_medical_device",
                tags={
                    "variable_id": row["variable_id"],
                    "device_serial": row.get("device_serial", "unknown"),
                },
                fields=fields,
                timestamp=ns_timestamp,
            )

        sent = writer.flush()
        logger.info("Processed message: %d rows queued, %d sent to QuestDB", len(data), sent)

    return handle_hl7_message

if __name__ == "__main__":
    setup_logging()
    logger.info("Starting HL7 Parser service")

    writer = QuestDBWriter(host="localhost", port=9009)
    try:
        receive_hl7_messages(callback=make_handler(writer))
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        writer.flush()
        writer.close()
