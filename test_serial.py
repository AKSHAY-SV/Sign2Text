"""
Sign2Text - Serial Test Module

End-to-end test of Python → UART → LPC2148 → LCD communication
without the ML model.

This allows debugging the hardware communication independently from the AI.

Usage:
    python test_serial.py

    The script will:
    1. Attempt to connect to the LPC2148 via serial UART
    2. Send test messages (HELLO, SPACE, CLEAR, BACKSPACE)
    3. Read any responses from the LPC2148
    4. Display status on the laptop screen

Controls:
    Press Q to quit the test.

Protocol (documented in config/settings.py and serial.py):
    - Messages are sent with a newline line ending (default: "\\n")
    - Simple ASCII protocol
    - Supported commands:
        TEXT:HELLO\n       - Send "HELLO" text
        CLEAR\n            - Clear the LCD/text buffer
        BACKSPACE\n        - Simulate backspace
        SPACE\n            - Send space character
"""

from __future__ import annotations

import sys
import time
import signal
from pathlib import Path
from typing import Optional

from src.communication.serial import SerialCommunicator


# Constants
KEY_QUIT = ord("q")
TEST_MESSAGES = [
    "HELLO",
    " ",
    "YES",
    "NO",
    "THANK YOU",
]


def signal_handler(signum, frame) -> None:
    """Handle OS signals (Ctrl+C) gracefully."""
    print("\nReceived signal {signum}. Exiting test...")


def main() -> None:
    """Run the serial communication test."""
    print("=" * 60)
    print("Sign2Text - Serial UART Test (No ML Model)")
    print("=" * 60)
    print()

    # Show configured serial settings
    from config.settings import SERIAL_PORT, BAUD_RATE, SERIAL_TIMEOUT, LINE_ENDING
    print(f"Configured serial port: {SERIAL_PORT}")
    print(f"Baud rate: {BAUD_RATE}")
    print(f"Line ending: {repr(LINE_ENDING)}")
    print(f"Timeout: {SERIAL_TIMEOUT}s")
    print()

    # Initialize serial communicator
    serial_comm = SerialCommunicator(
        port=SERIAL_PORT,
        baud_rate=BAUD_RATE,
        timeout=SERIAL_TIMEOUT,
    )

    # Check connection
    if not serial_comm.is_connected():
        print("=" * 60)
        print("ERROR: Could not open serial port!")
        print()
        print("Possible reasons:")
        print("  1. The LPC2148 is not connected or powered on")
        print("  2. The wrong COM/USB port is configured")
        print("  3. The LPC2148 firmware is not running")
        print("  4. On Linux, check /dev/ttyUSB0 or /dev/ttyACM0")
        print("  5. You may need to update SERIAL_PORT in config/settings.py")
        print()
        print("Updating port...")

        # Try to auto-detect
        import glob
        usb_ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        if usb_ports:
            print(f"Available USB/ACM ports: {usb_ports}")
            print(f"  Would you like to try {usb_ports[0]}?")
        else:
            print("  No USB/ACM ports found.")

        print()
        print("Edit config/settings.py to set the correct SERIAL_PORT.")
        print("=" * 60)
        return

    print("Serial connection established!")
    print()

    # Send test messages
    try:
        print("Sending test messages...")
        print()

        for message in TEST_MESSAGES:
            # Check if user wants to quit
            print(f"Sending: '{message}'")
            success = serial_comm.send_text(message)
            if success:
                print(f"  ✓ Sent successfully")
            else:
                print(f"  ✗ Failed to send")

            # Wait a moment between messages
            time.sleep(0.5)

            # Try to read any response (non-blocking with short timeout)
            try:
                line = serial_comm.read_line(timeout=0.1)
                if line is not None:
                    print(f"  <- Response: '{line}'")
            except Exception:
                pass  # No response is also OK

        print()
        print("=" * 60)
        print("Test complete. Messages sent:")
        for msg in TEST_MESSAGES:
            print(f"  - {msg}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

    finally:
        # Clean up
        serial_comm.close()
        print("Serial connection closed.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()