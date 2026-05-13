"""Email alerting for PdM service. Uses stdlib smtplib."""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

from ventilator_pdm.service.config import SMTPConfig

logger = logging.getLogger(__name__)


def compose_alert_email(
    alerts: list[dict],
    base_url: str = "http://localhost:8000",
) -> tuple[str, str]:
    """Compose an alert digest email summarising anomalous devices.

    Iterates over alert records and formats each into a multi-line block
    with device registration number, alert level, fleet percentile, onset
    timestamp, top contributing features, and a deep-link URL.

    Args:
        alerts: List of alert dicts.  Each dict must contain keys
            ``device_serial``, ``alert_level``, and ``timestamp``, and
            may contain ``device_reg``, ``fleet_percentile``, and
            ``top_features``.
        base_url: Base URL of the PdM web UI used to build per-device
            deep-link URLs.

    Returns:
        A two-tuple ``(subject, body)`` where both elements are plain-text
        strings ready to be passed to :func:`send_email`.
    """
    n = len(alerts)
    device_word = "device" if n == 1 else "devices"
    subject = f"PdM Alert — {n} {device_word} requires attention"

    lines = []
    for a in alerts:
        reg = a.get("device_reg", "?")
        level = a["alert_level"].upper()
        ts = a["timestamp"]
        pct = a.get("fleet_percentile", 0)
        features = a.get("top_features", "")
        serial = a["device_serial"]

        lines.append(f"Device {reg} — {level} (fleet percentile: {pct:.0%})")
        lines.append(f"  Since: {ts}")
        if features:
            lines.append(f"  Top drivers: {features}")
        lines.append(f"  Details: {base_url}/device/{serial}")
        lines.append("")

    body = "\n".join(lines)
    return subject, body


def compose_error_email(job_name: str, error_message: str) -> tuple[str, str]:
    """Compose an error notification email for a failed scheduled job.

    Args:
        job_name: Human-readable name or identifier of the job that failed
            (e.g. ``"daily_inference"``).
        error_message: String representation of the exception or error
            detail to include in the email body.

    Returns:
        A two-tuple ``(subject, body)`` of plain-text strings.
    """
    subject = f"PdM Service Error — {job_name}"
    body = (
        f"Job: {job_name}\n"
        f"Time: {datetime.now().isoformat()}\n"
        f"Error: {error_message}\n"
    )
    return subject, body


def send_email(
    smtp_config: SMTPConfig,
    subject: str,
    body: str,
) -> bool:
    """Send a plain-text email via SMTP.

    Opens a new SMTP connection for each call, sends the message, and
    closes the connection.  If ``smtp_config.host`` is empty or
    ``smtp_config.recipients`` is empty the function logs a warning and
    returns ``False`` without attempting a connection.

    Args:
        smtp_config: SMTP connection and addressing parameters, including
            ``host``, ``port``, ``from_addr``, and ``recipients``.
        subject: Email subject line.
        body: Plain-text email body.

    Returns:
        ``True`` if the message was accepted by the SMTP server, ``False``
        if SMTP is not configured or if the send attempt raised an
        exception.
    """
    if not smtp_config.host or not smtp_config.recipients:
        logger.warning("SMTP not configured, skipping email")
        return False

    msg = MIMEMultipart()
    msg["From"] = smtp_config.from_addr
    msg["To"] = ", ".join(smtp_config.recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_config.host, smtp_config.port, timeout=30) as server:
            server.sendmail(
                smtp_config.from_addr,
                smtp_config.recipients,
                msg.as_string(),
            )
        logger.info("Alert email sent: %s", subject)
        return True
    except Exception:
        logger.exception("Failed to send email: %s", subject)
        return False
