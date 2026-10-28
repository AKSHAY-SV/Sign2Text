#ifndef UART_H
#define UART_H
#include <stdbool.h>
#include <stdint.h>
#include "config.h"
bool uart_take_message(char *out, uint8_t capacity, bool *oversized);
void uart_init(void);
#endif
