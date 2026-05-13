import socket
from datetime import datetime

MLLP_START = b'\x0b'
MLLP_END = b'\x1c\x0d'

def extract_message_control_id(hl7_message):
    lines = hl7_message.replace('\r', '\n').split('\n')
    for line in lines:
        if line.startswith('MSH'):
            fields = line.split('|')
            if len(fields) > 9:
                return fields[9]
    return ''

def build_ack(original_message, message_control_id):
    msh_fields = original_message.split('\r')[0].split('|')
    sending_app = msh_fields[2] if len(msh_fields) > 2 else ''
    sending_fac = msh_fields[3] if len(msh_fields) > 3 else ''
    receiving_app = msh_fields[4] if len(msh_fields) > 4 else ''
    receiving_fac = msh_fields[5] if len(msh_fields) > 5 else ''
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    # Set your application/facility here:
    my_app = "MTU_datalake"
    my_fac = "SUS-UiS"
    ack = (
        f"MSH|^~\\&|{my_app}|{my_fac}|{sending_app}|{sending_fac}|{timestamp}||ACK^R01|{message_control_id}|P|2.3\r"
        f"MSA|AA|{message_control_id}\r"
    )
    return ack

def send_ack(conn, ack_message):
    ack_bytes = MLLP_START + ack_message.encode('utf-8') + MLLP_END
    conn.sendall(ack_bytes)

def receive_hl7_messages(host='0.0.0.0', port=2554, callback=None):
    import logging
    logger = logging.getLogger(__name__)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        logger.info("Listening for HL7 MLLP on %s:%d", host, port)
        while True:
            try:
                conn, addr = s.accept()
                logger.info("Connection from %s", addr)
                with conn:
                    buffer = b''
                    while True:
                        data = conn.recv(4096)
                        if not data:
                            break
                        buffer += data
                        while MLLP_START in buffer and MLLP_END in buffer:
                            start = buffer.index(MLLP_START) + 1
                            end = buffer.index(MLLP_END)
                            hl7_message = buffer[start:end].decode('utf-8')
                            buffer = buffer[end+2:]
                            try:
                                if callback:
                                    callback(hl7_message)
                            except Exception:
                                logger.exception("Error in message callback")
                            # Always ACK even if callback failed, so Capsule
                            # doesn't stall waiting and re-sends the message
                            try:
                                msg_id = extract_message_control_id(hl7_message)
                                ack = build_ack(hl7_message, msg_id)
                                send_ack(conn, ack)
                            except Exception:
                                logger.exception("Error sending ACK")
            except Exception:
                logger.exception("Connection error, waiting for next connection")
