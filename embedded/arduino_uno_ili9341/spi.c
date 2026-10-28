#include "config.h"
#include "spi.h"
#include <avr/io.h>

void spi_init(void) {
    DDRB |= (1 << PB3) | (1 << PB5) | (1 << PB2);
    DDRB &= (uint8_t)~(1 << PB4);
    SPCR = (1 << SPE) | (1 << MSTR); /* mode 0, MSB first, F_CPU/4 = 4 MHz */
    SPSR = 0;
}

uint8_t spi_transfer(uint8_t value) {
    SPDR = value;
    while (!(SPSR & (1 << SPIF))) { }
    return SPDR;
}
