# Uno R3 and ILI9341 SPI text display

This directory contains pure AVR C for an ATmega328P. It uses avr-gcc,
avr-libc, the Uno hardware SPI peripheral and the ATmega UART. No Arduino
framework, C++ library, `.ino`, or framebuffer is used.

## Wiring

This firmware is configured for the supplied 240×320 ILI9341 module and the
existing wiring:

| TFT pin | Uno pin |
| --- | --- |
| CS | D10 |
| RESET/RST | D8 |
| DC | D9 |
| MOSI/SDI | D11 |
| SCK/SCL | D13 |
| MISO/SDO | D12, if present (not used by this driver) |
| GND | GND |

Connect VCC and LED/backlight only as required by the exact module. The module
power connections are reported as already made according to its requirements;
this project has not independently verified its onboard regulator or logic
level shifters. Keep D0/RX and D1/TX available for the Uno USB serial bridge.

## Build and upload

On Linux Mint/Ubuntu install the tools with:

```sh
sudo apt install gcc-avr avr-libc avrdude make
```

Install the Python serial dependency into the integrated application's
environment:

```sh
.venv-repair/bin/python -m pip install pyserial
```

Build from the repository root:

```sh
make -C embedded/arduino_uno_ili9341 clean all
```

Expected artifacts are `build/sign2text_ili9341.elf` and
`build/firmware.hex`. The `all` target checks that both are non-empty. Upload
only after confirming the connected target and port:

```sh
make -C embedded/arduino_uno_ili9341 flash PORT=/dev/ttyACM0
```

This command is provided for manual use; building does not flash the board.
The same Uno connected to USB is the target; no second Arduino is needed.
Find the current device with `ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null`.
If access is denied, check the user's existing `dialout` group membership;
this setup does not change system permissions.

If the panel stays blank, build and manually upload the display-only diagnostic:

```sh
make -C embedded/arduino_uno_ili9341 diagnostic
avrdude -p atmega328p -c arduino -P /dev/ttyACM0 -b 115200 -D \
  -U flash:w:embedded/arduino_uno_ili9341/build/display_diagnostic.hex:i
```

It initializes the TFT, clears it black, then displays `ILI9341 TEST` and
`HELLO WORLD` in large white text. This isolates TFT initialization and pixel
writes from Python and UART reception. Reflash the normal `build/firmware.hex`
to restore text reception.

Run the existing Sign2Text application with serial output enabled:

```sh
./run_sign2text_integrated.sh --arduino-port /dev/ttyACM0 --arduino-baud 115200
```

Omit both Arduino options for laptop-only use. Python sends the same committed
text shown in the laptop UI; the sender queue and serial writes run in a
background thread. Opening `/dev/ttyACM0` may reset the Uno; the sender allows
the bootloader time to finish before transmitting the queued latest message.

## Protocol and rendering

UART is configured for 115200 8N1 at 16 MHz using U2X and UBRR=16, which
produces about 117647 baud (+2.12%). Messages are newline terminated and
bounded at 128 payload bytes. An RX interrupt stores complete lines in a
bounded two-slot queue while the display is drawn. Oversized lines display a
notice; empty lines return to the waiting prompt. A small built-in 5×7 font
draws ASCII letters, digits and common punctuation in RGB565. Unsupported
glyphs show as `?`. Text wraps at the display edge and truncates at the bottom
with an ellipsis where space permits. Updates clear only the text area. The
default rotation is portrait (`ROTATION=0`, 240×320); landscape uses
`make -C embedded/arduino_uno_ili9341 clean all ROTATION=1` (320×240). Rotations
2 and 3 select the opposite portrait and landscape orientations.

SPI uses mode 0, MSB first, at 4 MHz for conservative module compatibility.
CS remains asserted across each command and its parameters; for drawing, it
stays asserted from CASET/PASET through RAMWR and the RGB565 pixel stream.

No physical display validation is claimed. Confirm the module's 5 V logic
compatibility from its documentation; use level shifting if the inputs are
3.3 V-only.
