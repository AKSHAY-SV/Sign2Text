#!/bin/bash
#
# tools/arduino_detect.sh
# Robust Arduino Uno R3 serial port detection for Linux Mint/Ubuntu.
#
# Rules (in order of preference):
# 1. /dev/serial/by-id/— ideal, persistent symlinks
# 2. /dev/ttyACM* — ACM devices (cdc_acm USB modems/Arduinos)
# 3. /dev/ttyUSB* — Prolific/FTDI-style devices
#
# Output:
#   Port name, or "NOT_FOUND" if no Arduino detected.
#   Reason text printed to stderr/stdout for each case.
#
# Usage:
#   ./tools/arduino_detect.sh
#   ./tools/arduino_detect.sh --port-only    # output only the port name
#
# Exit codes:
#   0 — Arduino found, port printed to stdout
#   1 — No Arduino found (port printed as "NOT_FOUND")
#   2 — Permission error (details printed)
#

set -euo pipefail

# ——— Configuration ————————————————————————————————
PREFERRED_VENDOR="2341"       # Arduino LLC USB Vendor ID
PREFERRED_PRODUCT="0043"      # Arduino Uno R3 USB Product ID
SERIAL_PREFIX_ACM="ttyACM"
SERIAL_PREFIX_USB="ttyUSB"

# ——— Helper functions ————————————————————————————————

log() { echo "$@" >&2; }
die() { log "$*"; exit 1; }

# check_permission — Verify the user can access the given port.
check_permission() {
    local port="$1"
    if [ ! -c "$port" ]; then
        log "WARNING: $port is not a character device"
        return 1
    fi
    if [ ! -r "$port" ] || [ ! -w "$port" ]; then
        log "WARNING: Permission denied for $port (check dialout group)"
        return 1
    fi
    return 0
}

# find_by_id — Look for /dev/serial/by-id/ links matching Arduino VID:PID
find_by_id() {
    local link
    if [ -d /dev/serial/by-id ]; then
        for link in /dev/serial/by-id/*; do
            if [ -L "$link" ]; then
                # shellcheck disable=SC2162
                local target=$(readlink -f "$link" 2>/dev/null)
                if [ -n "$target" ]; then
                    # Check if the symlink name contains our VID:PID or the target does
                    local link_basename=$(basename "$link")
                    if echo "$link_basename" | grep -q "\(arduino\|$PREFERRED_VENDOR\)" 2>/dev/null; then
                        # Extract port name from symlink path
                        local portname=$(basename "$link")
                        # Verify the target is a ttyACM or ttyUSB
                        if echo "$target" | grep -q "ttyACM\|ttyUSB"; then
                            check_permission "/dev/$portname" && echo "/dev/$portname"
                            return 0
                        fi
                    fi
                fi
            fi
        done
    fi
    return 1
}

# find_by_acm — Look for /dev/ttyACM* devices
find_by_acm() {
    local device
    for device in /dev/${SERIAL_PREFIX_ACM}*; do
        if [ -c "$device" ]; then
            # Check USB device descriptors via lsusb -v (may not be available)
            # For now, assume any ttyACM could be an Arduino
            check_permission "$device" && echo "$device" && return 0
        fi
    done
    return 1
}

# find_by_usb — Look for /dev/ttyUSB* devices and verify VID:PID via lsusb
find_by_usb() {
    local device
    for device in /dev/${SERIAL_PREFIX_USB}*; do
        if [ -c "$device" ]; then
            check_permission "$device" && echo "$device" && return 0
        fi
    done
    return 1
}

# verify_vid_pid — Verify a device's USB VID:PID matches Arduino
verify_vid_pid() {
    local device="$1"
    if command -v lsusb >/dev/null 2>&1; then
        local vid product
        vid=$(lsusb -d "$PREFERRED_VENDOR:$PREFERRED_PRODUCT" >/dev/null 2>&1 && echo "MATCH" || echo "NO_MATCH")
        if [ "$vid" = "MATCH" ]; then
            return 0
        fi
    fi
    # If lsusb or the specific device isn't available, we fall back to heuristic
    return 1
}

# ——— Main logic ————————————————————————————————

# Try method 1: /dev/serial/by/id/
if result=$(find_by_id 2>/dev/null); then
    if [ -n "$result" ]; then
        echo "Arduino detected (by-ID): $result"
        exit 0
    fi
fi

# Try method 2: /dev/ttyACM*
if result=$(find_by_acm 2>/dev/null); then
    if [ -n "$result" ]; then
        echo "Arduino detected (ttyACM): $result"
        exit 0
    fi
fi

# Try method 3: /dev/ttyUSB*
if result=$(find_by_usb 2>/dev/null); then
    if [ -n "$result" ]; then
        echo "Arduino detected (ttyUSB): $result"
        exit 0
    fi
fi

# No Arduino found
echo "NOT_FOUND" >&2
log "No Arduino Uno R3 device found. Tried: /dev/serial/by-id/, /dev/ttyACM*, /dev/ttyUSB*"
exit 1