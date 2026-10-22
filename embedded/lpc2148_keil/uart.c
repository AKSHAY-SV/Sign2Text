#include "LPC214x.h"
#include "uart.h"

/* Lab manual reference: VPBDIV=2, PCLK=30 MHz, UART0 DLL=195 at 9600 baud.
   VPBDIV=2 means PCLK=CCLK/2; CCLK must therefore be 60 MHz. Set VPBDIV
   here rather than relying on startup code. Confirm the board's actual CCLK. */
#define UART0_BAUD 9600u
#define UART0_PCLK_HZ 30000000u
#define UART0_DIV 195u

#if ((UART0_PCLK_HZ + (8u * UART0_BAUD)) / (16u * UART0_BAUD)) != UART0_DIV
#error UART0 divisor does not agree with the documented 30 MHz PCLK
#endif

void uart0_init(void) {
    VPBDIV = 0x02u; /* PCLK = CCLK/2; expected CCLK=60 MHz for 30 MHz PCLK. */
    PINSEL0 = (PINSEL0 & ~0x0000000Fu) | 0x00000005u; /* P0.0 TXD0, P0.1 RXD0 */
    U0LCR = 0x83; /* 8 data bits, no parity, 1 stop, divisor access */
    U0DLM = (UART0_DIV >> 8) & 0xFFu;
    U0DLL = UART0_DIV & 0xFFu;
    U0LCR = 0x03;
    U0FCR = 0x07;
}
int uart0_getchar_nonblocking(void) {
    if ((U0LSR & 0x01u) == 0u) return -1;
    return (int)(U0RBR & 0xFFu);
}
