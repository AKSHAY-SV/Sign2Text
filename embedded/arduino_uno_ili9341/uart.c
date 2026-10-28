#include "uart.h"
#include <avr/interrupt.h>
#include <avr/io.h>
#include <util/atomic.h>

#define MESSAGE_SLOTS 2u
static volatile char slots[MESSAGE_SLOTS][UART_RX_MESSAGE_CAPACITY];
static volatile uint8_t slot_lengths[MESSAGE_SLOTS];
static volatile uint8_t slot_bad[MESSAGE_SLOTS];
static volatile uint8_t head, tail;
static char rx[UART_RX_MESSAGE_CAPACITY];
static uint8_t rx_length, rx_bad;

void uart_init(void) {
    /* U2X=1, UBRR=16 at 16 MHz => 117647 baud (+2.12% vs 115200). */
    UCSR0A = (1 << U2X0);
    UBRR0 = (uint16_t)((F_CPU + 4UL * UART_BAUD) / (8UL * UART_BAUD) - 1UL);
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00); /* 8N1 */
    UCSR0B = (1 << RXEN0) | (1 << RXCIE0);
}

ISR(USART_RX_vect) {
    char c = (char)UDR0;
    if (c == '\r') return;
    if (c == '\n') {
        uint8_t next = (uint8_t)((head + 1u) % MESSAGE_SLOTS);
        if (next != tail) {
            uint8_t i;
            for (i = 0; i < rx_length; ++i) slots[head][i] = rx[i];
            slots[head][rx_length] = '\0';
            slot_lengths[head] = rx_length;
            slot_bad[head] = rx_bad;
            head = next;
        }
        rx_length = 0;
        rx_bad = 0;
    } else if (rx_length < UART_RX_MESSAGE_CAPACITY - 1u) {
        rx[rx_length++] = c;
    } else {
        rx_bad = 1;
    }
}

bool uart_take_message(char *out, uint8_t capacity, bool *oversized) {
    uint8_t n, i, bad;
    if (!capacity) return false;
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
        if (head == tail) return false;
        n = slot_lengths[tail];
        bad = slot_bad[tail];
        if (n >= capacity) n = (uint8_t)(capacity - 1u);
        for (i = 0; i < n; ++i) out[i] = slots[tail][i];
        out[n] = '\0';
        tail = (uint8_t)((tail + 1u) % MESSAGE_SLOTS);
    }
    if (oversized) *oversized = bad != 0;
    return true;
}
