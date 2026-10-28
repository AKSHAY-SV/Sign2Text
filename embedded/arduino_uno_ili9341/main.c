#include "config.h"
#include "ili9341.h"
#include "spi.h"
#include "uart.h"
#include <avr/interrupt.h>

int main(void) {
    char message[UART_RX_MESSAGE_CAPACITY];
    bool oversized;
    spi_init();
    ili9341_init();
    uart_init();
    sei();
    ili9341_show_message("");
    for (;;) {
        if (uart_take_message(message, sizeof(message), &oversized)) {
            if (oversized) ili9341_show_message("MESSAGE TOO LONG");
            else ili9341_show_message(message);
        }
    }
}
