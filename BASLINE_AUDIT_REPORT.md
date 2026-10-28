BASELINE REPORT — Sign2Text Arduino Connectivity Optimization
================================================================

Repository: ~/live-asl-translation
Date: 2026-10-28

1. TOOL VERSIONS
----------------
- OS: Linux akshaysv-ThinkPad-E16-Gen-1 7.0.0-34-generic (Ubuntu 24.04)
- Python: 3.12.3
- avr-gcc: 7.3.0
- GNU Make: 4.3
- AVRDude: available (invocation: avrdude -p <partno> ...)

2. HARDWARE AVAILABILITY
-----------------------
- Arduino Uno R3 (VID: 0x2341, PID: 0x0043) connected via USB
- Device: ttyACM0 (cdc_acm device)
- User groups: adm dialout dialout cdrom sudo dip video plugdev users lpadmin sambashare ollama render
- Arduino detected in dmesg: manufacturer "Arduino (www.arduino.cc)", device "ttyACM0"

3. FIRMWARE BUILD STATUS
------------------------
- Production firmware (sign2text_ili9341): builds successfully
  - Program: 2574 bytes (7.9% Full) of ATmega328P
  - Data: 701 bytes (34.2% Full)
- Diagnostic firmware (display_diagnostic): builds successfully
  - Program: 2334 bytes (7.1% Full) of ATmega328P
  - Data: 315 bytes (15.4% Full)
- Both built with: make -C embedded/arduino_uno_ili9341 all
- Build output: build/sign2text_ili9341.elf, build/firmware.hex, build/display_diagnostic.elf, build/display_diagnostic.hex

4. EXISTING FIRMWARE FILES
--------------------------
- embedded/arduino_uno_ili9341/main.c — Production main()
- embedded/arduino_uno_ili9341/display_diagnostic.c — Diagnostic main()
- embedded/arduino_uno_ili9341/config.h — Pin definitions, CPU freq, colors
- embedded/arduino_uno_ili9341/ili9341.c/h — ILI9341 driver
- embedded/arduino_uno_ili9341/spi.c/h — SPI driver
- embedded/arduino_uno_ili9341/uart.c/h — UART driver
- embedded/arduino_uno_ili9341/font.c/h — 5x7 font data
- embedded/arduino_uno_ili9341/Makefile — Build system

5. EXISTING SERIAL COMMUNICATION
--------------------------------
- Python sender: tools/arduino_serial_sender.py
  - ArduinoTextSender class with threading
  - format_message(): UTF-8 encode, bounded to 128 bytes, LF terminator
  - Boot delay: 2.0s default on port open (Uno reset)
  - Baud rate: 115200
  - Messages: newline-terminated, bounded at 128 payload bytes
- Application runner: tools/run_sign2text_integrated.py
  - Uses ArduinoTextSender from tools
  - Opens port, waits boot_delay (2s), then sends queued latest text
  - Handles reconnection
- Message format: UTF-8, LF-terminated, max 128 bytes

6. MAKEFILE TARGETS
-------------------
- all: firmware diagnostic (builds both)
- firmware: builds and verifies production HEX
- diagnostic: builds and verifies diagnostic HEX
- flash: flashes production HEX via AVRDude
- clean: removes build directory

7. SERIAL PORT
--------------
- Expected: /dev/ttyACM0
- Actual: /dev/ttyACM0 (when Arduino connected)
- User in: dialout group
- Device: cdc_acm (ACM - Abstract Control Model)

8. PYTHON VIRTUAL ENVIRONMENT
-----------------------------
- .venv-repair/ — main environment
- .venv-alphabet/ — alphabet recognizer environment
- Requirements: pyserial>=3.5,<4, opencv, tensorflow, mediapipe, etc.

9. EXISTING TESTS
-----------------
- tests/test_arduino_serial_sender.py — Arduino serial sender tests
- tests/test_integrated_helpers.py — Integrated helper tests
- Tests reference Arduino port /dev/ttyACM0

10. KNOWN LIMITATIONS
---------------------
- No physical Arduino connected at all times
- Serial port may be unavailable (device disconnects)
- Permissions may denied (not in dialout group on all systems)
- TFT display white-screen issue exists but is OUT OF SCOPE for this mission
- Python virtual environment may not be fully activated
- Some test dependencies may need .venv-repair/bin/python

11. EXISTING MODIFICATIONS (from git)
--------------------------------------
- README.md modified
- requirements.txt modified
- tests/test_integrated_helpers.py modified
- tools/run_sign2text_integrated.py modified
- Multiple lpc2148_keil files deleted (pre-existing, not in this mission scope)
- embedded/arduino_uno_ili9341/ — new directory (untracked, contains the Arduino firmware)
- tests/test_arduino_serial_sender.py — new file
- tools/arduino_serial_sender.py — new/modified file

12. MISSION SCOBJECTIVE
-----------------------
Optimize laptop-to-Arduino USB serial connectivity, firmware flashing, and communication.
Preserve existing firmware. Do NOT modify TFT display driver or ILI9341 behavior.
Focus on phases 1-12 as defined in the mission scope.