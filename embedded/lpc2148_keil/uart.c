#include "LPC214x.h"
#include "uart.h"

/* Define UART0_PCLK_HZ in the Keil target from the project's clock setup.
   This driver deliberately does not change VPBDIV or the startup clock. */
#define UART0_BAUD 9600u

#ifndef UART0_PCLK_HZ
#error Define UART0_PCLK_HZ to the actual UART0 peripheral clock in Hz
#endif

/* Rounded integer divisor: PCLK / (16 * baud), nearest integer. */
#define UART0_DIV ((UART0_PCLK_HZ + (8UL * UART0_BAUD)) / (16UL * UART0_BAUD))
#if (UART0_DIV < 1) || (UART0_DIV > 65535)
#error UART0 baud divisor is outside the LPC2148 16-bit divisor range
#endif

void uart0_init(void) {
    PINSEL0 = (PINSEL0 & ~0x0000000Fu) | 0x00000005u; /* P0.0 TXD0, P0.1 RXD0 */
    U0FCR = 0x07u; /* enable FIFO and reset RX/TX FIFOs */
    U0LCR = 0x83; /* 8 data bits, no parity, 1 stop, divisor access */
    U0DLM = (unsigned int)((UART0_DIV >> 8) & 0xFFu);
    U0DLL = (unsigned int)(UART0_DIV & 0xFFu);
    U0LCR = 0x03;
}
int uart0_getchar_nonblocking(void) {
    if ((U0LSR & 0x01u) == 0u) return -1;
    return (int)(U0RBR & 0xFFu);
}
