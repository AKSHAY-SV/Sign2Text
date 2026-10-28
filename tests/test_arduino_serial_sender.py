import unittest
from threading import Event
from unittest.mock import Mock

from tools.arduino_serial_sender import ArduinoTextSender, MAX_MESSAGE_BYTES, format_message


class ArduinoSerialSenderTests(unittest.TestCase):
    def test_newline_framing_and_utf8_byte_limit(self):
        self.assertEqual(format_message("HELLO"), b"HELLO\n")
        self.assertLessEqual(len(format_message("é" * 100)) - 1, MAX_MESSAGE_BYTES)
        self.assertTrue(format_message("é" * 100).endswith(b"\n"))

    def test_empty_message_is_a_newline(self):
        self.assertEqual(format_message(""), b"\n")
        self.assertEqual(format_message("x" * (MAX_MESSAGE_BYTES + 20)),
                         b"x" * MAX_MESSAGE_BYTES + b"\n")

    def test_embedded_line_breaks_are_flattened(self):
        self.assertEqual(format_message("one\ntwo\rthree"), b"one two three\n")

    def test_utf8_text_is_transmitted_without_loss(self):
        message = "HELLO café"
        self.assertEqual(format_message(message)[:-1].decode("utf-8"), message)

    def test_duplicate_suppression(self):
        sender = ArduinoTextSender.__new__(ArduinoTextSender)
        sender._last_offered = None
        sender._queue = __import__("queue").Queue(maxsize=1)
        sender.offer("same")
        sender.offer("same")
        self.assertEqual(sender._queue.qsize(), 1)
    def test_write_error_is_contained_and_connection_closed(self):
        attempted = Event()
        serial = Mock()
        def broken_write(_payload):
            attempted.set()
            raise OSError("disconnected")
        serial.write.side_effect = broken_write
        sender = ArduinoTextSender("test-port", serial_factory=Mock(return_value=serial),
                                   boot_delay=0)
        sender.offer("HELLO")
        self.assertTrue(attempted.wait(1))
        sender.close()
        serial.close.assert_called()

    def test_missing_port_is_nonfatal(self):
        attempted = Event()
        def unavailable(*_args, **_kwargs):
            attempted.set()
            raise OSError("no such device")
        factory = Mock(side_effect=unavailable)
        sender = ArduinoTextSender("missing", serial_factory=factory, boot_delay=0)
        sender.offer("HELLO")
        self.assertTrue(attempted.wait(1))
        sender.close()
        self.assertGreaterEqual(factory.call_count, 1)

    def test_missing_serial_dependency_stays_laptop_only(self):
        sender = ArduinoTextSender.__new__(ArduinoTextSender)
        sender._serial_factory = None
        with unittest.mock.patch.dict("sys.modules", {"serial": None}):
            self.assertIsNone(sender._open())


if __name__ == "__main__":
    unittest.main()
