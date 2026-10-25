#!/usr/bin/env bash
#
# build.sh - GNU Arm GCC build for the LPC2148 Sign2Text UART->LCD receiver.
#
# Builds the firmware entirely from the Linux terminal. No Keil, no ARMCC,
# no Keil project file is required.
#
# Usage:
#   UART0_PCLK_HZ=<actual UART0 peripheral clock in Hz> ./build.sh
#   ./build.sh clean
#
# UART0_PCLK_HZ is REQUIRED and has no assumed default. It must match the
# target's real peripheral clock, derived from the board's CCLK and VPBDIV
# (e.g. 30000000 for a 30 MHz PCLK). The 30 MHz figure is only a
# documentation example from the prior lab-manual citation; verify it
# against the board and the lab's Keil clock setup before building.
#
# Outputs (in build/):
#   firmware.elf  linked ELF image
#   firmware.hex  Intel HEX, flash sections only (.vectors .text .rodata)
#   firmware.map  linker map with cross-reference
#   firmware.sections.txt  ELF section headers
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
ELF="$BUILD_DIR/firmware.elf"
HEX="$BUILD_DIR/firmware.hex"
MAP="$BUILD_DIR/firmware.map"
SECTIONS="$BUILD_DIR/firmware.sections.txt"

SRCS=(main.c uart.c lcd.c)
OBJS=()

# --- Toolchain checks -------------------------------------------------------
for tool in arm-none-eabi-gcc arm-none-eabi-objcopy arm-none-eabi-size \
            arm-none-eabi-readelf python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required utility not found: $tool" >&2
        echo "install (Linux Mint/Ubuntu):" >&2
        echo "  sudo apt-get update && sudo apt-get install -y \\" >&2
        echo "    gcc-arm-none-eabi binutils-arm-none-eabi python3" >&2
        exit 1
    fi
done

# --- Clean target -----------------------------------------------------------
if [ "${1:-}" = "clean" ]; then
    echo "removing build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    echo "clean done (source files untouched)"
    exit 0
fi

# --- UART0 peripheral clock (required, no guessed default) ------------------
if [ -z "${UART0_PCLK_HZ:-}" ]; then
    echo "error: UART0_PCLK_HZ must be set to the actual UART0 peripheral clock in Hz" >&2
    echo "  example: UART0_PCLK_HZ=30000000 ./build.sh" >&2
    echo "  determine PCLK from the board's CCLK and VPBDIV and verify against the lab manual" >&2
    exit 1
fi

if ! [[ "$UART0_PCLK_HZ" =~ ^[0-9]+$ ]] || [ "$UART0_PCLK_HZ" -lt 1 ]; then
    echo "error: UART0_PCLK_HZ must be a positive integer (got '$UART0_PCLK_HZ')" >&2
    exit 1
fi

echo "=== LPC2148 Sign2Text receiver - GNU Arm GCC build ==="
echo "source dir : $SCRIPT_DIR"
echo "build dir  : $BUILD_DIR"
echo "PCLK_HZ    : $UART0_PCLK_HZ"

CFLAGS=(
    -mcpu=arm7tdmi
    -marm
    -std=c99
    -ffreestanding
    -nostdlib
    -fno-unwind-tables
    -fno-asynchronous-unwind-tables
    -O2
    -g
    -Wall
    -Wextra
    -Werror
    -I"$SCRIPT_DIR"
    -DUART0_PCLK_HZ="$UART0_PCLK_HZ"
)

LDFLAGS=(
    -mcpu=arm7tdmi
    -marm
    -nostdlib
    -ffreestanding
    -static
    -Wl,-Map,"$MAP"
    -Wl,--cref
    -T"$SCRIPT_DIR/lpc2148.ld"
)

mkdir -p "$BUILD_DIR"

# --- Compile C sources ------------------------------------------------------
for src in "${SRCS[@]}"; do
    obj="$BUILD_DIR/${src%.c}.o"
    echo "cc  $src -> $obj"
    arm-none-eabi-gcc "${CFLAGS[@]}" -c "$SCRIPT_DIR/$src" -o "$obj"
    OBJS+=("$obj")
done

# --- Assemble startup code --------------------------------------------------
startup_o="$BUILD_DIR/startup.o"
echo "as  startup.s -> $startup_o"
arm-none-eabi-gcc "${CFLAGS[@]}" -x assembler-with-cpp -c "$SCRIPT_DIR/startup.s" -o "$startup_o"
OBJS+=("$startup_o")

# --- Link -------------------------------------------------------------------
echo "ld  -> $ELF"
arm-none-eabi-gcc "${LDFLAGS[@]}" "${OBJS[@]}" -o "$ELF"

# --- Intel HEX (flash sections only, for Flash Magic) ----------------------
echo "objcopy -> $HEX"
arm-none-eabi-objcopy -O ihex \
    --only-section=.vectors \
    --only-section=.text \
    --only-section=.rodata \
    "$ELF" "$HEX"

# --- Validate HEX -----------------------------------------------------------
if [ ! -s "$HEX" ]; then
    echo "error: HEX file missing or empty: $HEX" >&2
    exit 1
fi

python3 - "$HEX" <<'PYEOF'
import sys

path = sys.argv[1]
FLASH_START = 0x00000000
FLASH_END   = 0x0007FFFF   # LPC2148: 512 KB flash (verify against datasheet)

with open(path, "r") as f:
    records = [line.strip() for line in f if line.startswith(":")]

if not records:
    raise SystemExit("error: HEX file contains no records")

nonflash = []
total_bytes = 0
for rec in records:
    if len(rec) < 11:
        continue
    reclen = int(rec[1:3], 16)
    addr = int(rec[3:7], 16)
    rtype = int(rec[7:9], 16)
    if rtype == 0:  # data record
        total_bytes += reclen
        end = addr + reclen - 1
        if not (FLASH_START <= addr and end <= FLASH_END):
            nonflash.append((addr, end))

if nonflash:
    print("error: HEX contains records outside flash:", nonflash, file=sys.stderr)
    sys.exit(1)

print(f"hex ok : {len(records)} records, {total_bytes} data bytes, "
      f"all within flash 0x{FLASH_START:08X}-0x{FLASH_END:08X}")
PYEOF

# --- Reports ----------------------------------------------------------------
echo
echo "--- size ---"
arm-none-eabi-size -B "$ELF"

echo
echo "--- sections ---"
arm-none-eabi-readelf -S "$ELF" > "$SECTIONS"
arm-none-eabi-readelf -S "$ELF"

hex_size=$(stat -c '%s' "$HEX")

echo
echo "=== BUILD OK ==="
echo "ELF  : $ELF"
echo "HEX  : $HEX ($hex_size bytes)"
echo "MAP  : $MAP"
echo "SECS : $SECTIONS"
echo
echo "Program $HEX into the LPC2148 with Flash Magic (verify the serial port,"
echo "baud, and oscillator/PLL settings against the board and lab manual)."
echo "Close Flash Magic's serial connection before opening the port from Python."
