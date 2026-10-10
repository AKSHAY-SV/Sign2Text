"""
Sign2Text - Serial Communication Module

Python serial communication for sending text to LPC2148 microcontroller
via UART.

Provides a simple, configurable interface for UART communication with
the LPC2148 evaluation board. The LPC2148 receives characters over UART
and displays them on its LCD.

Protocol:
    - Text messages sent with line ending (e.g., "HELLO\n", "YES\n")
    - Simple ASCII protocol, one line per message
    - No handshaking required for basic operation
    - Supported commands: CLEAR, BACKSPACE

Configuration is read from config/settings.py - do not hardcode ports.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import serial

from config.settings import (
    SERIAL_PORT,
    BAUD_RATE,
    SERIAL_TIMEOUT,
    SERIAL_WRITE_TIMEOUT,
    LINE_ENDING,
)


class SerialCommunicator:
    """Handles UART serial communication with LPC2148.

    Provides methods for sending text messages to the LPC2148 microcontroller
    via serial UART. The LPC2148 receives characters and displays them on its
    LCD screen.

    Attributes:
        _ser: Underlying pySerial Serial object.
        _port: Serial port name.
        _baud_rate: Baud rate for communication.
        _timeout: Read timeout in seconds.
        _write_timeout: Write timeout in seconds.
    """

    def __init__(
        self,
        port: str = SERIAL_PORT,
        baud_rate: int = BAUD_RATE,
        timeout: float = SERIAL_TIMEOUT,
        write_timeout: float = SERIAL_WRITE_TIMEOUT,
    ) -> None:
        """Initialize the serial communicator.

        Args:
            port: Serial port name (e.g., "/dev/ttyUSB0", "COM3").
            baud_rate: Baud rate for UART communication.
            timeout: Read timeout in seconds.
            write_timeout: Write timeout in seconds.
        """
        self._port = port
        self._baud_rate = baud_rate
        self._timeout = timeout
        self._write_timeout = write_timeout
        self._ser: Optional[serial.Serial] = None

    def _connect(self) -> None:
        """Establish the serial connection.

        Attempts to open the serial port. If it fails, prints a warning
        and continues without an active connection (self._ser remains None).
        """
        try:
            self._ser = serial.Serial(
                port=self._port,
                baudrate=self._baud_rate,
                timeout=self._timeout,
                write_timeout=self._write_timeout,
            )
            print(f"Serial connection opened on {self._port} at {self._baud_rate} baud")
        except serial.SerialException as e:
            print(f"Warning: Could not open serial port {self._port}: {e}")
            print("  Continuing without serial connection.")
            self._ser = None

    def is_connected(self) -> bool:
        """Check if the serial connection is open and active.

        Returns:
            True if connected, False otherwise.
        """
        return self._ser is not None and self._ser.is_open

    def send_text(self, text: str) -> bool:
        """Send a text message to the LPC2148.

        Adds the line ending character (default: newline) and sends
        the message over UART. Special commands are supported:
        - "CLEAR" clears the LPC2148 text buffer/LCD
        - "BACKSPACE" deletes the last character on LPC2148 LCD

        Args:
            text: The text message to send (without line ending).
                Special command names are also accepted.

        Returns:
            True if sent successfully, False if not connected.
        """
        if self._ser is None or not self._ser.is_open:
            print("Cannot send text: serial connection not open.")
            return False

        try:
            # Special command handling
            cmd = text.strip().upper()
            if cmd == "CLEAR":
                message = "CLEAR"
            elif cmd == "BACKSPACE":
                message = "BACKSPACE"
            else:
                message = f"{text}"

            # Add line ending and encode as ASCII
            full_message = f"{message}{LINE_ENDING}"
            self._ser.write(full_message.encode("ascii"))
            self._ser.flush()
            return True
        except serial.SerialException as e:
            print(f"Error sending text: {e}")
            return False
        except UnicodeEncodeError as e:
            print(f"Error encoding text: {e}")
            return False

    def send_line(self, text: str) -> bool:
        """Send a text line to the LPC2148 (alias for send_text).

        Args:
            text: The text message to send.

        Returns:
            True if sent successfully.
        """
        return self.send_text(text)

    def send_multiline(self, lines: list[str]) -> bool:
        """Send multiple lines of text to the LPC2148.

        Args:
            lines: List of text lines to send.

        Returns:
            True if all lines sent successfully.
        """
        if self._ser is None or not self._ser.is_open:
            return False

        try:
            for line in lines:
                self.send_text(line)
            return True
        except serial.SerialException as e:
            print(f"Error sending multi-line text: {e}")
            return False

    def read_line(self, timeout: float = None) -> Optional[str]:
        """Read a single line from the serial port.

        Args:
            timeout: Read timeout in seconds. Uses default if None.

        Returns:
            The line read (without line ending), or None if no data.
        """
        if self._ser is None or not self._ser.is_open:
            return None

        if timeout is None:
            timeout = self._timeout

        try:
            # Read until newline
            line = self._ser.readline(timeout=timeout)
            if line:
                # Decode and strip line ending
                return line.decode("ascii").strip()
            return None
        except serial.SerialException as e:
            print(f"Error reading from serial: {e}")
            return None
        except UnicodeDecodeError as e:
            print(f"Error decoding serial data: {e}")
            return None

    def close(self) -> None:
        """Close the serial connection."""
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
            print("Serial connection closed.")

    def __enter__(self) -> "SerialCommunicator":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()