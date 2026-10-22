#!/usr/bin/env python3
"""Optional serial output for the LPC2148 Sign2Text LCD receiver."""

from __future__ import annotations

import argparse
import sys
import threading
import time


def normalize_message(message: str) -> str:
    """Keep printable ASCII, replacing unsupported code points with '?'.

    Trailing carriage returns and newlines are line terminators for the LPC2148
    protocol (CR ignored, LF ends a line) and are stripped here so the sender
    emits exactly one LF-terminated line.  Any remaining non-printable bytes,
    including embedded control characters, become '?'.
    """
    message = message.rstrip("\r\n")
    return "".join(ch if 32 <= ord(ch) <= 126 else "?" for ch in message)


class SerialTextSender:
    def __init__(self, port: str, baud: int = 9600, *, serial_module=None):
        if serial_module is None:
            try:
                import serial as serial_module
            except ImportError as exc:
                raise RuntimeError("pySerial is required; install it in the app environment") from exc
        self._serial = serial_module.Serial(port, baudrate=baud, timeout=0.1, write_timeout=0.5)
        self._last: str | None = None
        self._lock = threading.Lock()

    def send(self, message: str, *, force: bool = False) -> bool:
        value = normalize_message(message)
        with self._lock:
            if not force and value == self._last:
                return False
            self._serial.write(value.encode("ascii") + b"\n")
            self._serial.flush()
            self._last = value
            return True

    def close(self) -> None:
        self._serial.close()


def manual_mode(sender: SerialTextSender) -> None:
    print("Type text to display; Ctrl-D/Ctrl-C quits. Blank line clears the LCD.")
    try:
        for line in sys.stdin:
            sender.send(line.rstrip("\r\n"))
    except KeyboardInterrupt:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="serial port (for example /dev/ttyUSB0 or COM4)")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--manual", action="store_true", help="type messages interactively")
    args = parser.parse_args(argv)
    sender = None
    try:
        sender = SerialTextSender(args.port, args.baud)
        if args.manual:
            manual_mode(sender)
        else:
            print("Connected. Use --manual for interactive messages; Ctrl-C quits.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Serial sender: {exc}", file=sys.stderr)
        return 1
    finally:
        if sender is not None:
            sender.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
