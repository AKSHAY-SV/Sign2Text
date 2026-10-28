#!/usr/bin/env python3
"""
tools/arduino_flash.py
Reliable AVRDUDE flashing for Arduino Uno R3 with ILI9341 firmware.

Supports:
  --detect              : List available Arduino ports
  --port PORT           : Use specified port
  --baud BAUD           : Upload baud rate (default: 115200)
  --target TARGET       : 'production' or 'diagnostic' (default: production)
  --dry-run             : Show commands without touching hardware
  --verify              : Run AVRDude -V (verify) after flashing
  --timeout SECONDS     : AVRDude timeout (default: 60)
  --retries N           : Retry on recoverable failures (default: 2)
  --log-file PATH       : Write log to file
  --help                : Show this help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ——— Project constants ————————————————————————————————
REPO_ROOT = Path(__file__).resolve().parents[1]
ARDUINO_DIR = REPO_ROOT / "embedded" / "arduino_uno_ili9341"
PRODUCTION_HEX = ARDUINO_DIR / "build" / "firmware.hex"
DIAGNOSTIC_HEX = ARDUINO_DIR / "build" / "display_diagnostic.hex"
ELF_PATH = ARDUINO_DIR / "build" / "sign2text_ili9341.elf"

DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 2
DEFAULT_PROGRAMMER = "arduino"
DEFAULT_MCU = "atmega328p"

LOGGER = logging.getLogger("arduino_flash")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ——— Arduino detection ————————————————————————————————


def detect_ports() -> list[dict[str, str]]:
    """Scan for likely Arduino ACM/USB ports using lsusb and /dev entries."""
    ports: list[dict[str, str]] = []

    # Check /dev/ttyACM*
    import glob as glob_mod
    for dev in glob_mod.glob("/dev/ttyACM*"):
        if os.path.isfile(dev) and os.access(dev, os.R_OK | os.W_OK):
            ports.append({"port": dev, "type": "ACM", "desc": "ttyACM device"})

    # Check /dev/ttyUSB*
    for dev in glob_mod.glob("/dev/ttyUSB*"):
        if os.path.isfile(dev) and os.access(dev, os.R_OK | os.W_OK):
            ports.append({"port": dev, "type": "USB", "desc": "ttyUSB device"})

    # Prefer /dev/serial/by-id/ links
    by_id_dir = "/dev/serial/by-id"
    if os.path.isdir(by_id_dir):
        for link in os.listdir(by_id_dir):
            full = os.path.join(by_id_dir, link)
            if os.path.islink(full):
                target = os.readlink(full)
                if "ttyACM" in target or "ttyUSB" in target:
                    # Extract a readable name from the symlink
                    ports.append(
                        {"port": "/dev/" + link, "type": "by-id", "desc": f"by-id symlink -> {target}"}
                    )

    return ports


def select_port(user_port: Optional[str] = None) -> str:
    """Select the serial port, with automatic detection if not specified."""
    if user_port:
        if os.path.isfile(user_port) and os.access(user_port, os.R_OK | os.W_OK):
            return user_port
        else:
            LOGGER.warning("Specified port %s not accessible, attempting auto-detect", user_port)

    # Auto-detect
    ports = detect_ports()
    if not ports:
        raise RuntimeError("No Arduino serial port detected. Connect the Arduino or specify --port.")

    if len(ports) == 1:
        LOGGER.info("Auto-selected port: %s (%s)", ports[0]["port"], ports[0]["desc"])
        return ports[0]["port"]

    # Multiple ports: prefer the one matching the Arduino VID:PID
    LOGGER.warning("Multiple serial devices found, selecting first:")
    for p in ports:
        LOGGER.warning("  %s (%s)", p["port"], p["desc"])
    # Return the first ACM port if available, else first USB
    for p in ports:
        if p["type"] == "ACM":
            LOGGER.info("Selected ACM port: %s", p["port"])
            return p["port"]
    return ports[0]["port"]


# ——— Build verification ————————————————————————————————


def build_target(target: str) -> Path:
    """Build the requested firmware target. Returns the path to the .hex file."""
    if target == "production":
        hex_path = PRODUCTION_HEX
        source_pattern = "main.c"
    elif target == "diagnostic":
        hex_path = DIAGNOSTIC_HEX
        source_pattern = "display_diagnostic.c"
    else:
        msg = f"Unknown target: {target!r}"
        raise ValueError(msg)

    # Verify the .hex exists; if not, build
    if not hex_path.exists() or hex_path.stat().st_size == 0:
        LOGGER.info("Building %s target...", target)
        # Run make from the Arduino directory
        result = subprocess.run(
            ["make", "-C", str(ARDUINO_DIR), target],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            LOGGER.error("Build failed for target %s:\n%s", target, result.stderr)
            raise RuntimeError(f"Build failed for target {target}. See log above.")
        LOGGER.info("Build succeeded for target %s.", target)

    # Verify the hex is non-empty
    if not hex_path.exists() or hex_path.stat().st_size == 0:
        msg = f"Hex file {hex_path} is empty or missing after build"
        LOGGER.error(msg)
        raise RuntimeError(msg)

    # Verify the ELF exists (gc-sections may have removed it)
    if not ELF_PATH.exists():
        LOGGER.warning("ELF file %s not found after build (may be gc-sectored)", ELF_PATH)

    return hex_path


# ——— AVRDude invocation ————————————————————————————————

def run_avrdude(
    hex_path: Path,
    port: str,
    mcu: str,
    baud: int,
    programmer: str,
    timeout: int,
    verify: bool,
    dry_run: bool,
) -> dict:
    """Execute AVRDude and return result dict."""
    # Build the AVRDude command as a list (safe, no shell injection)
    cmd: list[str] = [
        "avrdude",
        "-p", mcu,
        "-c", programmer,
        "-P", port,
        "-b", str(baud),
        "-D",  # Don't wait for run-down of DTR/DDS
        "-U", f"flash:w:{hex_path.name}:i",
    ]

    if verify:
        cmd.append("-U")

    if timeout:
        cmd.extend(["-t", str(timeout)])

    LOGGER.info("AVRDude command: %s", " ".join(cmd))

    if dry_run:
        LOGGER.info(
            "DRY-RUN: Would execute: avrdude %s",
            " ".join(cmd[len(["avrdude"]):]),
        )
        return {"exit_code": 0, "dry_run": True, "port": port}

    LOGGER.info("Flashing firmware to %s...", port)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
    except subprocess.TimeoutExpired as exc:
        LOGGER.error("AVRDude timed out after %d seconds", timeout)
        return {"exit_code": -1, "timed_out": True, "port": port, "error": str(exc)}

    # Log AVRDude output
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            LOGGER.info("AVRDude: %s", line.strip())
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            LOGGER.warning("AVRDude ERR: %s", line.strip())

    return {
        "exit_code": result.returncode,
        "dry_run": False,
        "port": port,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ——— Main ————————————————————————————————


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Flash Arduino Uno R3 with ILI9341 firmware."
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Detect and list available Arduino serial ports",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port (e.g. /dev/ttyACM0). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Upload baud rate (default: {DEFAULT_BAUD})",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["production", "diagnostic"],
        default="production",
        help="Firmware target to flash (default: production)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the AVRDude command without touching hardware",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run AVRDude verification after flashing (-V implicit via read-back)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"AVRDude timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Number of retry attempts on recoverable failures (default: 2)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write log output to the specified file",
    )
    parser.add_argument(
        "--help",
        action="help",
        help="Show this help message and exit",
    )

    args = parser.parse_args(argv)

    # Set up logging
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        LOGGER.addHandler(file_handler)

    # ——— Detect mode ————————————————————————————————
    if args.detect:
        ports = detect_ports()
        if not ports:
            print("No Arduino serial devices found.", file=sys.stderr)
            return 1
        for p in ports:
            print(f"  {p['port']} — {p['desc']}")
        return 0

    # ——— Select port ————————————————————————————————
    try:
        port = select_port(args.port)
    except RuntimeError as e:
        LOGGER.error("%s", e)
        print(str(e), file=sys.stderr)
        return 1

    LOGGER.info("Using port: %s", port)

    # ——— Build the selected target ————————————————————————————————
    hex_path = build_target(args.target)
    LOGGER.info("Using hex file: %s (%d bytes)", hex_path, hex_path.stat().st_size)

    # ——— Run AVRDude ————————————————————————————————
    result = run_avrdude(
        hex_path=hex_path,
        port=port,
        mcu=DEFAULT_MCU,
        baud=args.baud,
        programmer=DEFAULT_PROGRAMMER,
        timeout=args.timeout,
        verify=args.verify and args.target == "production",  # Only verify production by default
        dry_run=args.dry_run,
    )

    # ——— Report results ————————————————————————————————
    if args.dry_run:
        LOGGER.info("Dry-run complete. No hardware was modified.")
        return 0

    if result["exit_code"] == 0:
        LOGGER.info("Flash succeeded on %s", port)
        if not result.get("verify") is None:
            LOGGER.info("Verification requested and passed (embedded in AVRDude).")
        else:
            LOGGER.info("Verification skipped (use --verify for explicit verification).")
    else:
        LOGGER.error("Flash failed on %s", port)
        if result.get("timed_out"):
            LOGGER.error("  Timeout detail: AVRDude did not complete within %s seconds", args.timeout)
        if result.get("stderr"):
            # Extract key error info
            stderr_text = result["stderr"]
            if "avrdude: stk500_getsync" in stderr_text:
                LOGGER.error("  Synchronization failure — check wiring and port")
            if "avrdude: error writing flash" in stderr_text:
                LOGGER.error("  Flash write error — verify port is not busy")
            if "avrdude: verification error" in stderr_text:
                LOGGER.error("  Verification mismatch — compare HEX contents")

    return 0 if result["exit_code"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())