import sys
import types

import pytest

from tools.lpc2148_serial_sender import SerialTextSender, normalize_message


class FakePort:
    def __init__(self, *args, **kwargs):
        self.writes = []
        self.closed = False
    def write(self, data):
        self.writes.append(data)
    def flush(self):
        pass
    def close(self):
        self.closed = True


def build_sender(monkeypatch):
    fake = FakePort()
    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=lambda *a, **k: fake))
    return SerialTextSender("fake"), fake


@pytest.mark.parametrize("value", ["HELLO", "HELLO WORLD", "A B C", "X" * 32,
                                    "X" * 40, "", "HELLO\r\n"])
def test_line_protocol_and_consecutive_messages(monkeypatch, value):
    sender, port = build_sender(monkeypatch)
    assert sender.send(value)
    assert port.writes == [(value.rstrip("\r\n") + "\n").encode("ascii")]
    sender.close()
    assert port.closed


@pytest.mark.parametrize("length", [0, 1, 16, 17, 31, 32, 33, 80])
def test_lcd_boundary_lengths_are_sent_as_one_ascii_line(monkeypatch, length):
    sender, port = build_sender(monkeypatch)
    message = "A" * length
    assert sender.send(message)
    assert port.writes == [message.encode("ascii") + b"\n"]


def test_cr_lf_and_crlf_normalize_to_one_lf_line(monkeypatch):
    sender, port = build_sender(monkeypatch)
    for message in ("CR\r", "LF\n", "CRLF\r\n"):
        assert sender.send(message)
    assert port.writes == [b"CR\n", b"LF\n", b"CRLF\n"]


def test_duplicate_suppression(monkeypatch):
    sender, port = build_sender(monkeypatch)
    assert sender.send("HELLO")
    assert not sender.send("HELLO")
    assert sender.send("WORLD")
    assert port.writes == [b"HELLO\n", b"WORLD\n"]


def test_unsupported_characters_are_safely_replaced():
    assert normalize_message("hi\né") == "hi??"


def test_unavailable_port_error(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("port unavailable")
    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=fail))
    with pytest.raises(OSError, match="unavailable"):
        SerialTextSender("missing")


def test_write_failure_propagates_for_integration_worker_to_report(monkeypatch):
    class BrokenPort(FakePort):
        def write(self, data):
            raise OSError("disconnected")

    port = BrokenPort()
    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=lambda *a, **k: port))
    sender = SerialTextSender("fake")
    with pytest.raises(OSError, match="disconnected"):
        sender.send("HELLO")
