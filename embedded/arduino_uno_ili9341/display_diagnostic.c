/* Minimal diagnostic for ILI9341 TFT.
 * Initializes MCU, SPI, and TFT once.
 * Draws ILI9341 TEST and HELLO WORLD in white on black.
 * Loops with pin toggling to verify MCU activity.
 * No UART, no interrupts, WDT disabled.
 */

#include "config.h"
#include "ili9341.h"
#include "spi.h"
#include <avr/io.h>
#include <avr/wdt.h>
#include <util/delay.h>

/* Retain the reset flags across startup and prevent optimization away. */
volatile uint8_t diagnostic_reset_cause
    __attribute__((section(".noinit"), used));

void diagnostic_early_init(void)
    __attribute__((naked, used, section(".init3")));

void diagnostic_early_init(void) {
    diagnostic_reset_cause = MCUSR;
    MCUSR = 0;
    wdt_disable();

    /* Keep TFT selected high while D13 is used for the startup LED pattern. */
    TFT_CS_DDR |= (1 << TFT_CS_BIT);
    TFT_CS_PORT |= (1 << TFT_CS_BIT);
    TFT_DC_DDR |= (1 << TFT_DC_BIT);
    TFT_RESET_DDR |= (1 << TFT_RESET_BIT);
    TFT_RESET_PORT |= (1 << TFT_RESET_BIT);
}

/* LED pattern on startup based on reset cause. */
void startup_led_pattern(uint8_t reset_cause);

/* LED pattern on startup based on reset cause. */
void startup_led_pattern(uint8_t reset_cause) {
    uint8_t pulses = 1;
    if (reset_cause & (1 << WDRF)) pulses = 4;
    else if (reset_cause & (1 << BORF)) pulses = 3;
    else if (reset_cause & (1 << EXTRF)) pulses = 2;

    DDRB |= (1 << PB5); /* D13 onboard LED; also SPI SCK, before SPI starts. */
    PORTB &= (uint8_t)~(1 << PB5);
    while (pulses--) {
        PORTB |= (1 << PB5);
        _delay_ms(80);
        PORTB &= (uint8_t)~(1 << PB5);
        _delay_ms(120);
    }
}

int main(void) {
    startup_led_pattern(diagnostic_reset_cause);

    /* Disable watchdog explicitly. */
    wdt_disable();

    /* Initialize SPI (hardware mode 0, F_CPU/4 = 4 MHz). */
    spi_init();

    /* Initialize TFT once. */
    ili9341_init();

    /* Clear to black background. */
    ili9341_clear(COLOR_BLACK);

    /* Draw the required messages. */
    ili9341_draw_text(12, 48, "ILI9341 TEST", COLOR_WHITE, COLOR_BLACK, 3);
    ili9341_draw_text(12, 96, "HELLO WORLD", COLOR_WHITE, COLOR_BLACK, 3);

    /* Loop forever - toggle PB5 (D13/SPI SCK) to verify MCU is running.
     * PB5 is also the SPI SCK pin, but we set it as output directly,
     * so it won't interfere with the SPI peripheral in master mode
     * when no transfers are active (SCK will idle low). */
    for (;;) {
        PORTB ^= (1 << PB5);
        _delay_ms(500);
    }
}