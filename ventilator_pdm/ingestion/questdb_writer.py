import socket
import logging
import time

logger = logging.getLogger(__name__)


class QuestDBWriter:
    """Persistent ILP writer for QuestDB with reconnection and batching."""

    def __init__(self, host="localhost", port=9009, connect=True):
        self.host = host
        self.port = port
        self._sock = None
        self._batch = []
        if connect:
            self.connect()

    def connect(self):
        """Establish TCP connection to QuestDB ILP endpoint."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5)
            self._sock.connect((self.host, self.port))
            logger.info("Connected to QuestDB at %s:%d", self.host, self.port)
        except OSError as e:
            logger.error("Failed to connect to QuestDB at %s:%d: %s", self.host, self.port, e)
            self._sock = None

    def close(self):
        """Close the TCP connection."""
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _reconnect(self):
        """Close and re-establish the connection."""
        self.close()
        time.sleep(1)
        self.connect()

    @staticmethod
    def _sanitize_tag(value):
        """Sanitize a tag value for ILP: replace spaces with _, strip commas and = signs."""
        value = str(value).replace(" ", "_")
        value = value.replace(",", "").replace("=", "")
        return value

    def format_ilp_line(self, measurement, tags, fields, timestamp=None):
        """Format a single ILP line string.

        - String fields are double-quoted.
        - Boolean fields use t/f (QuestDB ILP format).
        - None and empty-string fields are skipped.
        """
        tag_str = ",".join(f"{k}={self._sanitize_tag(v)}" for k, v in tags.items())

        field_parts = []
        for k, v in fields.items():
            if v is None or v == "":
                continue
            if isinstance(v, bool):
                field_parts.append(f"{k}={'t' if v else 'f'}")
            elif isinstance(v, str):
                escaped = v.replace('"', '\\"')
                field_parts.append(f'{k}="{escaped}"')
            else:
                field_parts.append(f"{k}={v}")
        if not field_parts:
            raise ValueError("No valid fields to write")
        field_str = ",".join(field_parts)

        line = measurement
        if tag_str:
            line += f",{tag_str}"
        line += f" {field_str}"
        if timestamp:
            line += f" {timestamp}"
        line += "\n"
        return line

    def queue(self, measurement, tags, fields, timestamp=None):
        """Add an ILP line to the batch buffer."""
        line = self.format_ilp_line(measurement, tags, fields, timestamp)
        self._batch.append(line)

    def flush(self):
        """Send all queued ILP lines to QuestDB in one TCP write.

        Returns the number of lines successfully sent.
        """
        if not self._batch:
            return 0

        payload = "".join(self._batch)
        count = len(self._batch)

        for attempt in range(3):
            if self._sock is None:
                self.connect()  # No existing socket to close, just connect
            if self._sock is None:
                logger.warning("QuestDB not reachable (attempt %d/3)", attempt + 1)
                continue
            try:
                self._sock.sendall(payload.encode())
                logger.debug("Flushed %d ILP lines to QuestDB", count)
                self._batch.clear()
                return count
            except OSError as e:
                logger.warning("Send failed (attempt %d/3): %s", attempt + 1, e)
                if attempt < 2:
                    self._reconnect()

        logger.error("Failed to flush %d ILP lines after 3 attempts — dropping batch", count)
        self._batch.clear()
        return 0
