"""Best-effort, queued serial delivery for the final visible Sign2Text text."""

from __future__ import annotations

from queue import Empty, Queue
import threading
import time
import sys


MAX_MESSAGE_BYTES = 128


def format_message(text: str) -> bytes:
    """Encode one UTF-8 message, bounded by bytes, and terminated by LF."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    encoded = text.replace("\r", " ").replace("\n", " ").encode("utf-8")
    encoded = encoded[:MAX_MESSAGE_BYTES]
    while True:
        try:
            encoded.decode("utf-8")
            break
        except UnicodeDecodeError as exc:
            encoded = encoded[:exc.start]
    return encoded + b"\n"


class ArduinoTextSender:
    """A one-item latest-value queue; camera code only calls offer().

    Serial access and reconnect delays happen on a worker thread. An unavailable
    port is reported to stderr and retried while retaining the latest text.
    """

    def __init__(self, port: str, baudrate: int = 115200, *, serial_factory=None,
                 boot_delay: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self._serial_factory = serial_factory
        self._boot_delay = max(0.0, float(boot_delay))
        self._queue: Queue[str | None] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._last_offered: str | None = None
        self._last_sent: str | None = None
        self._last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name="arduino-text-sender", daemon=True)
        self._thread.start()

    def offer(self, text: str) -> None:
        if text == self._last_offered:
            return
        self._last_offered = text
        try:
            self._queue.put_nowait(text)
        except Exception:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(text)
            except Exception:
                pass

    def _open(self):
        if self._serial_factory is None:
            try:
                import serial
            except Exception as exc:
                self._report(f"pyserial is unavailable: {exc}")
                return None
            factory = serial.Serial
        else:
            factory = self._serial_factory
        try:
            connection = factory(self.port, self.baudrate, timeout=0, write_timeout=0.1)
            # Opening the USB UART may reset the ESP32; allow its firmware to boot.
            if self._boot_delay:
                time.sleep(self._boot_delay)
            self._report(None)
            return connection
        except Exception as exc:
            self._report(f"cannot open {self.port}: {exc}")
            return None

    def _report(self, message: str | None) -> None:
        previous_error = getattr(self, "_last_error", None)
        if message == previous_error:
            return
        had_error = previous_error is not None
        self._last_error = message
        if message:
            print(f"[TFT serial] {message}; retrying in 2 seconds", file=sys.stderr, flush=True)
        elif had_error:
            print(f"[TFT serial] connected to {self.port} at {self.baudrate} baud",
                  file=sys.stderr, flush=True)

    def _run(self) -> None:
        connection = None
        retry_at = 0.0
        pending: str | None = None
        while not self._stop.is_set():
            try:
                newer = self._queue.get(timeout=0.1)
                if newer is None:
                    break
                pending = newer
            except Empty:
                pass
            if pending is None or pending == self._last_sent:
                continue
            if connection is None and time.monotonic() >= retry_at:
                connection = self._open()
                retry_at = time.monotonic() + 2.0
            if connection is None:
                continue
            try:
                connection.write(format_message(pending))
                self._last_sent = pending
                pending = None
            except Exception as exc:
                self._report(f"write to {self.port} failed: {exc}")
                try:
                    connection.close()
                except Exception:
                    pass
                connection = None
                retry_at = time.monotonic() + 2.0
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass
        self._thread.join(timeout=2.5)

