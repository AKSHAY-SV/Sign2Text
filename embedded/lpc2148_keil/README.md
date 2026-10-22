# Sign2Text LPC2148 LCD receiver

Sources target the TMI Systems board mapping given in the lab manual: LCD RS=P0.2, EN=P0.3, D4..D7=P0.4..P0.7, with LCD R/W grounded. UART0 uses P0.0 TXD0 and P0.1 RXD0 via PINSEL0. Confirm physical serial routing and any board connection requirements in the lab; no connector, jumper, reset or ISP setting is inferred here.

## Clock reference and Keil setup

The lab manual's UART0 example sets `VPBDIV = 2`, states `PCLK = 30 MHz`, and loads `U0DLL = 195` for 9600 baud. On LPC2148, `VPBDIV = 2` selects PCLK = CCLK/2, so that stated PCLK corresponds to a 60 MHz CPU clock (CCLK). The baud divisor is `PCLK/(16 × baud) = 30,000,000/(16 × 9,600) = 195.3125`; the manual uses integer divisor 195, giving approximately 9601.6 baud. `uart.c` now explicitly writes `VPBDIV=2` so it does not depend on startup state and programs DLL=195, DLM=0. This is consistent only when the running CCLK is 60 MHz; confirm the lab project's oscillator/PLL setup before use. The 30 MHz PCLK and divisor are documented manual values; the actual board clock configuration remains a physical/project check.

No Keil µVision project (`.uvprojx`/`.uvproj`), startup code, LPC214x device header, scatter file, or Keil/ARMCC toolchain is available in this workspace; only host GCC is present, which is not a substitute for the ARM target compiler. A reliable Keil project cannot be generated here without inventing target and linker settings. In the lab's existing LPC2148 project, add `main.c`, `lcd.c`, and `uart.c`; add this directory to its include paths; include the supplied project/device-pack `LPC214x.h`; retain the verified LPC2148 device selection, startup code, C library/compiler conventions, memory map/scatter settings and oscillator/PLL setup. Confirm that the project clock yields 60 MHz CCLK, then ensure `uart0_init()`'s `VPBDIV=2` yields the documented 30 MHz PCLK. Enable Intel HEX output using the target's existing Keil configuration. The exact HEX location is defined by that project and cannot be stated until it is opened in the lab. No Keil build or HEX is claimed here.

Build the target and generate Intel HEX through the existing Keil workflow, then program using the lab's Flash Magic procedure. Flash programming and runtime UART communication are separate; the same port generally cannot be open in Flash Magic and the Python sender at once. Confirm oscillator, PCLK, connector routing, Flash Magic COM/settings, ISP switches/jumpers and reset sequence physically or in the manual.

## Protocol and runtime

Python sends the complete accumulated ASCII text followed by LF. CR is ignored. Empty text is a blank LF line and clears the LCD contents (spaces are written across both lines). Only the first 32 printable characters display; excess input is discarded safely through LF. The startup screen is `Sign2Text Ready`.

The GPIO delay loops are conservative software delays and may vary with compiler optimization and clock. Check LCD timing on the board and adjust if required.

## Host commands

```sh
python3 tools/lpc2148_serial_sender.py --port PORT --baud 9600 --manual
python3 -m pytest tests/test_lpc2148_serial_sender.py
python3 tools/run_sign2text_integrated.py --serial-port PORT --serial-baud 9600
```

Serial output is optional and handled by a background thread. Serial initialization failure leaves camera recognition running. Host tests do not establish physical UART or LCD operation.

## Lab checklist

1. Verify board identity, physical LCD and UART0 connections, serial routing and the lab project's oscillator/PLL settings (CCLK=60 MHz expected).
2. Add the three C files and include the project's LPC214x header; retain its startup and linker configuration. Confirm `VPBDIV=2` produces PCLK=30 MHz.
3. Build and confirm Intel HEX output in the Keil target's configured location.
4. Program with the lab's existing Flash Magic procedure, then release the programming connection.
5. Connect the Python UART path as documented for the board and test `HELLO`, longer text, and blank-line clear.
6. Record physical results; neither firmware nor display has been hardware-verified in this environment.
