# LPC2148 Sign2Text UART to LCD receiver

This directory contains a small receiver for the existing LPC2148 Keil project. Python performs camera processing and recognition; the board only receives an ASCII line over UART0 and displays up to 32 characters on its 16×2 LCD.

## What is verified here

The source implements UART0 at 9600 baud, 8N1, a 4-bit LCD interface, LF-delimited messages and safe truncation at 32 printable characters. The repository has no Keil project or lab-manual file; it now builds entirely from the Linux terminal with the GNU Arm Embedded GCC toolchain (see "GNU Arm GCC build" below). The mapping and clock information below was recorded in the previous version of this README as coming from a TMI Systems lab manual, but that manual is not present here and could not be independently checked. Verify all board connections and clock assumptions against the actual board and manual before building.

## Hardware assumptions to verify

| Signal | Expected LPC2148 pin | Requirement |
| --- | --- | --- |
| LCD RS | P0.2 | GPIO output |
| LCD EN | P0.3 | GPIO output |
| LCD D4–D7 | P0.4–P0.7 | GPIO outputs, 4-bit mode |
| LCD R/W | Ground | Write-only |
| UART0 TXD0 | P0.0 | Cross-connect to adapter RX |
| UART0 RXD0 | P0.1 | Cross-connect to adapter TX |
| Ground | Board GND | Common ground with USB-serial adapter |

The firmware selects the UART0 pin functions through `PINSEL0`. Confirm the evaluation-board connector actually routes these pins to the serial adapter and onboard LCD as expected, and check any level-shifting, jumpers and power requirements. Do not connect an RS-232 voltage-level port directly to MCU pins; use the board's intended interface or a suitable logic-level adapter. Expected serial format is 9600 baud, 8 data bits, no parity, one stop bit, no flow control.

## Clock and UART divisor

`UART0_PCLK_HZ` is required at compile time; there is no guessed default. `uart0_init()` does not alter `VPBDIV`, PLL or startup settings. Determine the project CPU clock (CCLK) from its oscillator and PLL setup and determine `VPBDIV` from startup code or the running target. On LPC2148, `VPBDIV=0` selects PCLK=CCLK/4, `VPBDIV=1` selects PCLK=CCLK, and `VPBDIV=2` selects PCLK=CCLK/2. (Verify the project's device documentation and actual register setup.)

The UART divisor is `round(PCLK / (16 × baud))`; this source calculates the nearest integer at compile time. At 9600 baud and a documented-example PCLK of 30 MHz, the divisor is 195 (`30,000,000 / (16 × 9,600) = 195.3125`), producing about 9601.6 baud. Treat 30 MHz as an example from the prior README's manual citation, not a confirmed setting for this project or board. Define the actual PCLK when building, for example `UART0_PCLK_HZ=30000000` only after verifying it. A mismatch in PCLK produces a corresponding baud-rate error.

## GNU Arm GCC build (Linux terminal, no Keil)

The firmware builds entirely from the terminal with the GNU Arm Embedded
GCC toolchain. Keil/ARMCC is not required.

### 1. Install dependencies (Linux Mint / Ubuntu)

```sh
sudo apt-get update
sudo apt-get install -y gcc-arm-none-eabi binutils-arm-none-eabi python3
```

`gcc-arm-none-eabi` provides `arm-none-eabi-gcc`; `binutils-arm-none-eabi`
provides `arm-none-eabi-objcopy`, `arm-none-eabi-size` and
`arm-none-eabi-readelf`. Python 3 is used by `build.sh` to validate the HEX.

### 2. Build

From this directory:

```sh
cd embedded/lpc2148_keil
UART0_PCLK_HZ=30000000 ./build.sh
```

`UART0_PCLK_HZ` is required and has no default: set it to the actual UART0
peripheral clock in Hz, determined from the board's CPU clock (CCLK) and
`VPBDIV` (`VPBDIV=0` → PCLK=CCLK/4, `VPBDIV=1` → PCLK=CCLK,
`VPBDIV=2` → PCLK=CCLK/2). Verify against the board schematic/lab manual;
30 MHz is an example only.

The build:
- checks that `arm-none-eabi-gcc`, `arm-none-eabi-objcopy`,
  `arm-none-eabi-size`, `arm-none-eabi-readelf` and `python3` are installed;
- compiles `main.c`, `uart.c`, `lcd.c` and assembles `startup.s`
  (`-mcpu=arm7tdmi -marm -std=c99 -ffreestanding -nostdlib -O2 -g
  -Wall -Wextra -Werror`);
- links with `lpc2148.ld` and emits a map file (`--cref`);
- produces `build/firmware.elf` and a flash-only Intel HEX
  `build/firmware.hex` (sections `.vectors`, `.text`, `.rodata` only — no
  RAM addresses, suitable for the LPC2148 bootloader and Flash Magic);
- validates that the HEX exists, is nonempty, and contains only flash
  addresses (0x00000000–0x0007FFFF);
- prints the ELF size report and section headers.

Clean generated artifacts without touching sources:

```sh
./build.sh clean
```

### 3. Generated outputs

| File | Meaning |
| --- | --- |
| `build/firmware.elf` | Linked ELF image |
| `build/firmware.hex` | Intel HEX for Flash Magic (flash addresses only) |
| `build/firmware.map` | Linker map with cross-reference |
| `build/firmware.sections.txt` | ELF section headers |

### 4. Programming with Flash Magic

1. Program `build/firmware.hex` into the LPC2148 with Flash Magic.
   Target address starts at 0x00000000 (reset vector).
2. Use the board's actual oscillator/PLL and ISP settings; Flash Magic
   baud/COM settings must match the hardware.
3. Close/release Flash Magic's serial connection after programming; do not
   let Flash Magic and Python hold the same serial port at once.
4. Connect the USB-serial adapter (UART0 TXD0=P0.0, RXD0=P0.1, common
   ground, logic levels) and test with the Python sender at 9600/8N1.

## Add to the lab's Keil project

1. Open the existing lab LPC2148 project and confirm the selected target is LPC2148.
2. Keep the lab project's startup code, oscillator/PLL setup, device header, memory map, linker configuration and compiler settings. Do not add a second startup or change clock setup to accommodate this receiver.
3. Add `main.c`, `lcd.c` and `uart.c` to a source group. Keep `lcd.h` and `uart.h` beside them (or add this directory to the C include paths).
4. Ensure the target's include paths provide the lab/device-pack `LPC214x.h` used by this code.
5. Add `UART0_PCLK_HZ=<actual PCLK in Hz>` to the target's C preprocessor definitions. Do not define it from an assumed CPU clock; account for the target's `VPBDIV` setting.
6. Build using the lab's installed Keil ARM compiler. Enable the target's Intel HEX output option and note the actual output path from the target settings/build log.

No project file or vendor startup/device files are supplied here, so exact Keil menu labels and HEX path depend on the lab project. A host compiler check is not a Keil build or evidence of ARMCC compatibility.

## Firmware protocol and display behavior

Send ASCII characters followed by LF (`\n`). CR is ignored, so LF and CRLF both terminate one message. Printable ASCII is displayed; other input bytes are ignored by the receiver. The first 32 printable bytes appear in positions 0–15 on row 1 and 16–31 on row 2. Further printable bytes are discarded until LF. Short lines are blank-filled; an empty LF line clears the display. Each completed line replaces the prior message, including when the new message is shorter. The startup display is `Sign2Text Ready`.

The LCD code uses software delay loops. Their timing depends on CPU clock and compiler optimization; confirm startup and command timing on the board and adjust the delay counts if needed. The firmware writes all 32 cells per received message, avoiding a repeated clear during polling.

## Test without the recognition model

Flash first using the lab's Flash Magic procedure and the generated HEX. After programming, close/release Flash Magic's serial connection; do not let Flash Magic and Python open the same serial port at once. Connect the USB-serial adapter and start the manual sender:

```sh
python3 tools/lpc2148_serial_sender.py --port /dev/ttyUSB0 --baud 9600 --manual
```

Replace `/dev/ttyUSB0` with the actual port. Type and submit these lines one at a time, checking the LCD after each:

```text
HELLO
HELLO WORLD
HI
[submit an empty line]
```

The last line clears both rows. Also check boundaries with 16, 17, 31, 32 and more than 32 ASCII characters. A serial terminal configured for 9600/8N1/no flow control can be used instead; send LF or CRLF line endings.

On Linux, inspect available serial devices with `ls /dev/ttyUSB* /dev/ttyACM*` (a no-match glob may print an error), or inspect kernel messages after plugging the adapter with `dmesg`. Access may require membership in the system's serial-device group or an administrator-granted permission; reconnect after group changes. Do not guess the device node.

## Integrated Sign2Text

From the repository root, enable optional serial output with the arguments supported by the integrated app:

```sh
python3 tools/run_sign2text_integrated.py --serial-port /dev/ttyUSB0 --serial-baud 9600
```

Without `--serial-port`, serial output is disabled. The UI queues text changes to a background serial worker so camera capture and inference do not wait for serial writes. The worker reports port/open/write errors and closes its sender at shutdown. Serial connection failure leaves recognition running.

Only committed/confirmed recognition candidates enter `TextAccumulator`; keyboard editing controls also apply to that same visible text. The application sends the complete current accumulated text whenever it changes, not just the newest character. The sender normalizes trailing CR/LF to one LF-terminated line, converts unsupported characters to `?`, and suppresses repeated identical payloads. `C`, Backspace and Space changes are transmitted too; clearing after a non-empty message sends a blank line. If the serial connection cannot be opened at startup, the app continues with LCD output disabled.

## Troubleshooting and lab checklist

1. Verify board identity, LCD pin wiring/R/W grounding, UART0 pin routing, adapter logic levels and common ground against the lab manual.
2. In Keil, inspect the startup/clock configuration and calculate actual PCLK from CCLK and `VPBDIV`; set the matching `UART0_PCLK_HZ` definition.
3. Add the three C files and include path/header to the existing project; build and confirm the HEX output path.
4. Program the HEX using the lab's Flash Magic settings and procedure. Close Flash Magic and release the port.
5. Test manual lines and different lengths; then run integrated Sign2Text with the same port.

| Symptom | Checks |
| --- | --- |
| Garbled text | Verify both ends are 9600/8N1/no flow control, actual PCLK and `UART0_PCLK_HZ` agree, and TX/RX are crossed with common ground. |
| Blank LCD | Verify LCD power/contrast, R/W ground, pin mapping, board routing, startup timing and that the existing project initializes GPIO as expected. |
| No UART reception | Verify UART0 pin mux, connector routing, adapter voltage levels, TX/RX cross-connect, common ground and LF termination. |
| Incorrect baud | Recalculate PCLK from CCLK and `VPBDIV`; check the Keil macro and any startup register writes. |
| Permission/open error | Check the actual Linux device node, close Flash Magic/terminal duplicates, and verify user access to the serial device. |

Without the physical evaluation board, its schematic/manual, and the lab's Keil installation, LCD electrical mapping, serial connector routing, clock, build compatibility, HEX generation, Flash Magic settings and runtime behavior remain unverified.
