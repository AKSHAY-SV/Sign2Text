# Sign2Text → ESP32 + 2.4" TFT SPI Display Migration Plan

## Phase 1 — Audit the existing project (Complete)

### Project Structure Inspection
- **Repository**: `~/live-asl-translation` (git tracked)
- **Python pipeline**: `launch_demo.py`, `tools/run_sign2text_integrated.py`
  - Uses MediaPipe Holistic for pose/hand/face landmark detection
  - Uses TensorFlow/Keras model (`models/best_model200.keras`, 271 output classes)
  - Runs inference every `prediction_interval` frames after 150-frame buffer
  - Outputs recognized text/label to UI and optionally over serial

- **Arduino firmware** (`embedded/arduino_uno_ili9341/`):
  - AVR C for ATmega328P
  - ILI9341 240x320 SPI TFT display driver
  - UART receive for text messages
  - 5x7 font renderer
  - Build with `make -C embedded/arduino_uno_ili9341 all`
  - Artifacts: `build/sign2text_ili9341.elf`, `build/firmware.hex`

- **LPC2148 firmware** (`embedded/lpc2148_keil/`): Original firmware (being retired per git diff)

- **Serial protocol** (`tools/arduino_serial_sender.py`):
  - `ArduinoTextSender` class with threading
  - `format_message()`: UTF-8 encode, bounded to 128 bytes, LF terminator
  - Boot delay: 2.0s default on port open (Uno reset)
  - Baud rate: 115200
  - Messages: newline-terminated, bounded at 128 payload bytes
  - Handles reconnection, blocking is avoided in main loop

- **Python serial sender integration** (`tools/run_sign2text_integrated.py`):
  - Uses `ArduinoTextSender` when `--arduino-port` specified
  - Sends confirmed text signs over serial
  - Handles port unavailable and permission errors

- **Tests** (`tests/`):
  - `test_arduino_serial_sender.py` — Arduino serial sender unit tests
  - `test_integrated_helpers.py` — Integrated helper tests
  - Other test files for word/alphabet recognizers

### Identified Components

| Component | Status |
|-----------|--------|
| Python recognition pipeline | **Preserve** — keep on laptop |
| TensorFlow/Keras model (`best_model200.keras`) | **Preserve** — model assets |
| MediaPipe landmark extraction | **Preserve** — laptop-side |
| ILI9341 Arduino firmware | **Replace** — migrate to ESP32 |
| AVR ATmega328P firmware | **Retire** — no longer needed |
| LPC2148 firmware | **Retire** — already being removed |
| `tools/arduino_serial_sender.py` | **Preserve protocol logic**, adapt for ESP32 USB CDC |
| `tools/run_sign2text_integrated.py` | **Preserve**, update serial config |
| `tests/test_arduino_serial_sender.py` | **Preserve and adapt** |
| `embedded/arduino_uno_ili9341/` | **Archive** after migration |
| `embedded/lpc2148_keil/` | **Already being retired** |

## Phase 2 — Protect existing work (Complete)

- Git repository at `~/live-asl-translation` with full history
- No uncommitted changes (`git diff HEAD --stat` shows only expected modifications)
- Original project preserved in Git history
- Python models and environment preserved (`.venv`, `.venv-repair`, `.venv-alphabet`)

## Phase 3 — Choose the ESP32 firmware environment

- **Preferred**: ESP-IDF (native ESP32 framework)
- **Not using**: Arduino IDE/CLI/Arduino-specific flashing workflow
- **ESP-IDF status**: Not currently installed — needs installation
- **Action**: Install ESP-IDF or use ESP32 Arduino core if preferred, but mission specifies native ESP-IDF

## Phase 4 — Identify the display controller

- **Confirmed controller**: **ILI9341**
  - Evidence: Existing Arduino firmware explicitly drives ILI9341
  - Init sequence in `ili9341.c` matches ILI9341 datasheet
  - Register `0x3A` pixel format set to `0x55` (RGB565)
  - `madctl` register for rotation selection
  - Display marking `2.4" TFT SPI 240X320 V1.3` with `SBT240-W61 V01` is a common ILI9341 module
  - SPI interface with CS, DC, RESET, SCK, MOSI, MISO pins

- **Touch controller**: Not confirmed from markings. Separate SD-card signals (`SD_SCK`, `SD_MISO`, `SD_MOSI`, `SD_CS`) are present but not for the main display.

- **Action**: Use ESP-IDF ILI9341 driver. If controller turns out to be different, will adapt.

## Phase 5 — Create a configurable pin map

### ESP32 GPIO pin selection (safe, available pins)

Based on ESP32 technical reference, avoiding boot-strapping pins:
- **GPIO6-11**: Boot-strapping/ JTAG - **AVOID**
- **GPIO0, GPIO2**: Boot-strapping - **AVOID** during boot
- **GPIO6** can be input-only with pull-up
- **Safe GPIO pins**: GPIO13, GPIO14, GPIO15, GPIO16, GPIO17, GPIO18, GPIO19, GPIO21, GPIO22, GPIO23, GPIO25, GPIO26, GPIO27, GPIO32, GPIO33

### TFT SPI pin assignments (ILI9341)

| Signal | ESP32 GPIO | Notes |
|--------|-----------|-------|
| SCLK (SCK) | GPIO18 | SPI2 HSPICLK |
| MOSI (SDA/SDI) | GPIO23 | SPI2 MOSI |
| MISO | GPIO19 | SPI2 MISO (optional, not needed for write-only) |
| CS (Chip Select) | GPIO5 | Free GPIO |
| DC (Data/Command) | GPIO17 | Free GPIO |
| RESET | GPIO16 | Free GPIO |
| BL/Backlight | GPIO4 | Free GPIO (PWM capable) |

**SD-card interface pins** (`SD_SCK`, `SD_MISO`, `SD_MOSI`, `SD_CS`): **NOT connected** to main display SPI. These are separate and not used.

### Touch controller pins (not implemented initially)

Touch controller not confirmed. Pins from board marking:
- `T_IRQ` - touch interrupt (not used initially)
- No touch SPI pins connected to main display SPI

### Power requirements

- TFT module: 3.3V logic, VCC may accept 3.3V or 5V depending on onboard regulator
- ESP32 GPIO: 3.3V only — never connect to 5V
- Backlight: May require separate power; GPIO4 can control transistor or PWM dimming

## Phase 6 — Implement the smallest working display test

### ESP-IDF ILI9341 driver selection

Use ESP-IDF's default ILI9341 driver configuration. The ESP-IDF framework includes `ili9341` driver in `components/ili9341` or via `display` component.

### Minimal firmware test tasks

1. Initialize ESP32
2. Initialize ILI9341 using ESP-IDF driver
3. Clear screen to black
4. Display:
   - `ESP32 TFT TEST`
   - `Sign2Text Ready`
   - Resolution: `240 x 320`
5. Draw simple shapes and render sample text
6. Use portrait rotation (240x320)
7. Report initialization errors over serial

### Build and flash commands (once ESP-IDF installed)

```bash
# Set up ESP-IDF path
. $HOME/esp-idf/export.sh

# Navigate to firmware directory
cd ~/live-asl-translation/embedded/esp32_tft

# Build
idf.py build

# Flash
idf.py flash

# Monitor serial
idf.py monitor
```

## Phase 7 — Implement the serial text receiver

### USB Serial protocol (ESP32)

- **Baud rate**: 115200 (compatible with existing Python sender)
- **Message format**: Newline-terminated UTF-8 strings
- **Maximum message length**: 128 bytes payload (matches existing protocol)
- **Invalid bytes**: Replace/ignore without crashing
- **Multiple messages**: Buffer incoming data, process line by line
- **Non-blocking**: Use FreeRTOS task or timer for serial reading
- **Display update**: Only refresh changed portions of screen

### Protocol design (compatible with existing)

```
[Text payload - UTF-8]\n
```

Maximum 128 bytes per message. Newline (`\n`) terminates the message. The ESP32 reads lines from USB serial CDC and displays them.

### Handling partial input

- Buffer incoming bytes until `\n` found
- If buffer exceeds 128 bytes, discard and start new message
- Invalid UTF-8 sequences: replace with `?` or strip
- Empty messages (just `\n`): show "Waiting for text"

### Display update strategy

- Do not refresh entire screen for every character
- Clear only the text area (bottom portion)
- Scroll or wrap text as needed
- Use double-buffering if possible

## Phase 8 — Integrate with the existing Python application

### Updates to `tools/run_sign2text_integrated.py`

- Add `--esp32-port` and `--esp32-baud` arguments (defaults to current `--arduino-port`/`--arduino-baud`)
- When ESP32 port specified, use `ArduinoTextSender` with that port
- Keep `--arduino-port` option for backwards compatibility (laptop-only mode when omitted)
- Allow serial port and baud rate to be configured without hardcoding
- Handle unavailable ports and permission errors with clear messages
- Preserve laptop-screen output option (when no serial port specified)

### Python serial configuration

```python
sender = ArduinoTextSender(arduino_port, arduino_baud)
```
or
```python
sender = ArduinoTextSender(esp32_port, esp32_baud)
```

Same `ArduinoTextSender` class can be used; just change the port name.

### Model inference preservation

- Keep all model inference on laptop (TensorFlow/Keras)
- Send only recognized text and protocol metadata
- Do not port TensorFlow or recognition models to ESP32

## Phase 9 — Remove obsolete Arduino/LPC2148-specific workflow

### Files to archive/remove (after migration confirmed working)

| File/Directory | Action |
|----------------|--------|
| `embedded/arduino_uno_ili9341/` | Archive after ESP32 firmware verified working |
| `embedded/lpc2148_keil/` | Already being retired per git diff |
| `tools/arduino_flash.py` | Archive (AVRDUDE for ATmega328P) |
| `tools/arduino_serial_sender.py` | **Preserve** — adapt for ESP32 USB CDC |
| `tests/test_arduino_serial_sender.py` | **Preserve and adapt** |
| Documentation referencing Arduino/LPC2148 | Update or archive |

### Preserve these files

- `launch_demo.py` — Main Python application
- `models/` — Model assets (`best_model200.keras`, `index_to_gloss_200.json`)
- `tools/` — Core logic, adapted for ESP32
- `tests/` — Tests, adapted for new protocol
- `README.md` — Update with ESP32 information
- `BASLINE_AUDIT_REPORT.md` — Keep as historical reference

## Phase 10 — Testing and acceptance criteria

### Tests to run

1. **Existing Python test suite**: `python -m pytest tests/` (adapt for new serial port)
2. **New serial protocol tests**: Test ESP32 serial receiver with fake data
3. **Firmware build**: `idf.py build` must succeed
4. **Firmware size**: Report BIN/ELF artifact sizes
5. **HEX/ELF artifacts**: Appropriate for ESP32 and ESP-IDF build system
6. **Serial receiver tests**: Using fake serial source if hardware not connected
7. **Minimal text-sender test**: Send `HELLO WORLD` to ESP32 over USB serial
8. **Full Sign2Text-to-display integration**: When hardware available

### Acceptance criteria checklist

- [ ] Code reviewed for all migrated components
- [ ] Build successful (`idf.py build`)
- [ ] Host-side tests passed
- [ ] Firmware flashed successfully (`idf.py flash`)
- [ ] Display physically tested (initialization, text rendering)
- [ ] End-to-end Sign2Text recognition and display verified
- [ ] Obsolete files removed after confirmation
- [ ] Original work preserved in Git history or backup

### Reporting distinctions

- Code reviewed ✓
- Build successful ✓ (or error details)
- Host-side tests passed ✓ (or error details)
- Firmware flashed successfully ✓ (or error details)
- Display physically tested ✗ (until hardware available)
- End-to-end Sign2Text recognition and display verified ✗ (until hardware available)

---

**Next step**: Install ESP-IDF and create the ESP32 project structure.