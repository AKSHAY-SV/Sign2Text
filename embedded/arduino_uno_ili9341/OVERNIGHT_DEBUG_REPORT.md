OVERNIGHT DEBUG REPORT - Arduino Uno R3 + ILI9341 TFT
 =====================================================

 Repository: ~/live-asl-translation
 Date: 2026-10-28

 HARDWARE
 ----------
 - Arduino Uno R3 / ATmega328P / 16 MHz
 - ILI9341 240x320 SPI TFT
 - Wiring: D13=SCK, D11=MOSI, D12=MISO, D10=CS, D9=DC, D8=RST
 - Linux Mint host
 - AVR-GCC, Make, AVRDUDE

 PROBLEM SYMPTOM
 ---------------
 - Diagnostic firmware compiles, flashes, and verifies successfully
 - Screen initially displays "ILI9341 TEST" and "HELLO WORLD"
 - After approximately 2 seconds, screen becomes white
 - Previous watchdog and delay changes did not resolve the issue

 FILES INSPECTED
 ---------------
 1. embedded/arduino_uno_ili9341/main.c        - Production firmware entry point
 2. embedded/arduino_uno_ili9341/config.h      - Pin definitions, CPU freq, colors
 3. embedded/arduino_uno_ili9341/ili9341.c      - ILI9341 driver (init, clear, text)
 4. embedded/arduino_uno_ili9341/ili9341.h      - ILI9341 function declarations
 5. embedded/arduino_uno_ili9341/spi.c         - SPI driver (init, transfer)
 6. embedded/arduino_uno_ili9341/spi.h         - SPI function declarations
 7. embedded/arduino_uno_ili9341/uart.c        - UART driver (init, ISR, take_message)
 8. embedded/arduino_uno_ili9341/uart.h        - UART function declarations
 9. embedded/arduino_uno_ili9341/font.c        - 5x7 font data (letters/digits)
10. embedded/arduino_uno_ili9341/font.h       - Font function declarations
11. embedded/arduino_uno_ili9341/display_diagnostic.c - Minimal diagnostic firmware

 PRODUCTION FIRMWARE (main.c) ANALYSIS
 --------------------------------------
 - main() initializes SPI, ILI9341, UART
 - Calls sei() early (enables ALL interrupts including UART RX Complete)
 - UART ISR (USART_RX_vect) can asynchronously call ili9341_show_message()
 - ili9341_show_message() fills bottom screen area with black and draws new text
 - Infinite loop checks uart_take_message() and redraws display on receipt
 - Key concern:sei() enables interrupts; UART RXCIE0 can fire at any time

 DIAGNOSTIC FIRMWARE (display_diagnostic.c) ANALYSIS
 -----------------------------------------------------
 - diagnostic_early_init() in .init3 section:
   * Captures MCUSR reset cause
   * Clears MCUSR
   * Disables watchdog (wdt_disable())
   * Sets TFT_CS, TFT_DC, TFT_RESET as outputs, driven high
 - main():
   * Calls startup_led_pattern() based on reset cause
   * Calls wdt_disable() explicitly
   * Calls spi_init() - SPI mode 0, F_CPU/4 = 4 MHz
   * Calls ili9341_init() - full ILI9341 init sequence
   * Calls ili9341_clear(COLOR_BLACK) then draws "ILI9341 TEST" and "HELLO WORLD"
   * Loops with: __asm__ __volatile__("nop") - NO UART, NO interrupts (no sei())
   * Includes pin toggle (PB5/D13) in modified version to verify MCU is running

 KEY OBSERVATIONS
 ----------------
 1. The diagnostic DOES NOT call sei() - global interrupts are never enabled
 2. The diagnostic DOES NOT initialize UART - no RXCIE0, no RX interrupts
 3. The diagnostic DOES disable the watchdog in early init and again in main()
 4. The problem occurs in the diagnostic, which has no UART and no interrupts
 5. Initial drawing is correct - "ILI9341 TEST" and "HELLO WORLD" display properly
 6. After ~2 seconds, screen becomes white despite no subsequent code execution
 7. Previous watchdog and delay changes did not resolve the issue (per problem statement)

 BUILD RESULTS
 -------------
 - Production firmware (sign2text_ili9341): 2574 bytes (7.9% Full)
 - Diagnostic display_diagnostic: 2282 bytes (7.0% Full) [original]
 - Diagnostic display_diagnostic: 2334 bytes (7.1% Full) [with pin-toggle]
 - Both compile with -Wall -Wextra -Werror and pass

 RESET CAUSE DIAGNOSTICS
 -----------------------
 - MCUSR capture in diagnostic_early_init() records reset reason
 - WDRF: Watchdog Reset Flag
 - BORF: Brown-out Reset Flag
 - EXTRF: External Reset Flag
 - In diagnostic, WDT is disabled early; BORF/EXTRF unlikely without external cause

 INVESTIGATED HYPOTHESES (and why they likely don't apply)
 ----------------------------------------------------------
 1. Watchdog timer reset: - WDT disabled in diagnostic early init and main()
    - Problem statement says "previous watchdog changes did not resolve issue"
    - Eliminates WDT as primary cause

 2. UART ISR corrupting display: - Diagnostic has NO UART init, no sei()
    - Problem occurs in diagnostic too; eliminates async UART as cause

 3. SPI peripheral corruption: - SPI enabled in diagnostic but no transfers after init
    - CS driven high; no CS toggle in NOP loop
    - SPI mode 0, 4 MHz clock
    - Cannot fully rule out without hardware observation

 4. TFT auto-sleep/power-saving: - ILI9341 sleep is software-controlled (0x01/0x11)
    - No hardware auto-sleep timer found in standard datasheet
    - Possible but not confirmed

 5. Brown-out reset: - MCU on 16 MHz from USB power
    - No BORF flag set in diagnostics (would be visible on reboot)
    - Unlikely without power supply anomaly

 6. MCU code crash/lockup: - Diagnostic just has NOP loop after init
    - No code path that could modify display RAM
    - Cannot rule out without hardware observation

 7. ILI9341 init sequence issue: - Init sequence sends: Software reset, Sleep Out,
    Power controls, Gamma, Pixel format, MADCTL, 0x13, 0x29
    - Initial drawing is correct, so init sequence works
    - Issue occurs after ~2 seconds of idle, not immediately

 8. CS pin floating/toggling: - diagnostic_early_init() drives CS high
    - spi_init() also sets CS as output
    - No SPI transfers in idle loop; CS should remain high
    - Possible if pin floats due to driver issue

9. Electrical/logic level issue: - Arduino Uno 5V -> TFT module (possibly 3.3V)
    - README warns: "Confirm the module's 5 V logic compatibility from its documentation;
      use level shifting if the inputs are 3.3 V-only"
    - Possible contributor but unlikely to cause exact ~2s timeout

 EXPERIMENTS PLANNED / IN PROGRESS
 -----------------------------------
 Experiment 1: Build diagnostic with pin-toggling on PB5 (D13)
   - Purpose: Verify MCU is still running (LED toggle) while observing screen
   - Result: Modified diagnostic built (2334 bytes); cannot test on hardware
   - Conclusion: Still theoretical without hardware access

 Experiment 2: Build diagnostic without text drawing (just init + clear)
   - Purpose: Determine if the issue is in the drawing or happens regardless
   - File: display_diag_part1.c (created, then removed - needed cleanup)
   - Result: Partially tested via code analysis only

 Experiment 3: Modify ILI9341 init sequence (add/remove delays, change command order)
   - Purpose: See if init sequence changes the ~2s timeout
   - Result: Code changes possible; cannot test on hardware

 Experiment 4: Modify SPI settings (clock rate, mode)
   - Purpose: See if SPI timing changes the ~2s timeout
   - Result: Code changes possible; cannot test on hardware

 Experiment 5: Disable SPI after init in diagnostic
   - Purpose: See if keeping SPI enabled causes the issue
   - Result: Code change possible; SPCR reset would be needed

 EXPERIMENT RESULTS (to be filled in after hardware testing)
 ----------------------------------------------------------
 [PENDING - awaiting Arduino connection]

 ROOT CAUSE STATUS
 ----------------
 - Not yet determined definitively
 - Code analysis eliminates: WDT, UART ISR (no UART in diagnostic)
 - Most likely: hardware-level issue or TFT module behavior not visible in code
 - Next step: Flash diagnostic to hardware and observe screen for 60+ seconds

 HARDWARE ACCESS STATUS
 ----------------------
 - Arduino Uno R3 not currently connected to this Linux Mint host
 - USB device (VID: 0x2341, PID: 0x0043) was last seen on ttyACM0
 - dmesg shows repeated connect/disconnect cycles
 - Port /dev/ttyACM0 not currently available
 - Cannot flash or verify firmware on hardware at this time

 REMAINING BLOCKERS
 ------------------
 - No physical access to Arduino + ILI9341 setup
 - Cannot observe screen behavior in real time
 - Cannot measure MCU registers or SPI activity
 - Cannot perform controlled electrical experiments

 NEXT ACTIONS (when hardware becomes available)
 ---------------------------------------------
 1. Connect Arduino Uno R3 with ILI9341 TFT
 2. Discover current serial port (ls /dev/ttyACM* /dev/ttyUSB*)
 3. Flash the display_diagnostic.hex to the Arduino
 4. Observe the screen for at least 60 seconds
 5. Check if PB5 (D13/onboard LED) toggles (verify MCU is running)
 6. Record: does screen go white after ~2s? Does PB5 keep toggling?
 7. If PB5 stops toggling: MCU is resetting after ~2s
 8. If PB5 keeps toggling: Display is changing without MCU reset
 9. Try flashing the production firmware and observe same behavior
10. If stuck, try: adding level shifters, checking power supply, replacing TFT module

 COMMANDS TO REPRODUCE FINAL TEST (when hardware available)
 ---------------------------------------------------------
 # Discover port
 ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null

 # Build diagnostic
 make -C embedded/arduino_uno_ili9341 diagnostic

 # Flash diagnostic (replace PORT with actual port)
 avrdude -p atmega328p -c arduino -P /dev/ttyACM0 -b 115200 -D \
   -U flash:w:embedded/arduino_uno_ili9341/build/display_diagnostic.hex:i

 # Verify flash
 avrdude -p atmega328p -c arduino -P /dev/ttyACM0 -b 115200 -D \
   -U flash:v:embedded/arduino_uno_ili9341/build/display_diagnostic.hex:i

 # Observe screen for 60+ seconds
 # Note: PB5 toggle indicates MCU is running;
 #   screen state indicates display health

 # Build and flash production firmware for comparison
 make -C embedded/arduino_uno_ili9341 firmware
 avrdude -p atmega328p -c arduino -P /dev/ttyACM0 -b 115200 -D \
   -U flash:w:build/firmware.hex:i