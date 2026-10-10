/*
 * LPC2148 UART Module Implementation
 *
 * UART initialization and receive functions for communication with
 * the laptop host running Sign2Text.
 *
 * Protocol: Receives ASCII characters with newline line ending.
 * Characters are buffered and can be parsed as commands.
 */

#include "uart.h"
#include <LPC214x.h>  // LPC2148 peripheral registers
#include <stdio.h>
#include <string.h>

// =============================================================================
// LOCAL DEFINITIONS AND TYPEDEFS
// =============================================================================

// UART buffer size for receive buffer
#define UART_RX_BUFFER_SIZE 64

// Maximum line length for uart_read_line
#define UART_MAX_LINE_LENGTH 128

// =============================================================================
// LOCAL VARIABLES
// =============================================================================

// Circular receive buffer
static struct {
    char data[UART_RX_BUFFER_SIZE];
    volatile uint8_t head;   // Next position to read
    volatile uint8_t tail;   // Next position to write
    volatile uint8_t count;  // Number of bytes in buffer
} uart_rx_buffer;

// Line buffer for reading complete lines
static char uart_line_buffer[UART_MAX_LINE_LENGTH];
static volatile uint8_t line_pos = 0;
static bool line_complete = false;

// =============================================================================
// UART INITIALIZATION
// =============================================================================

void uart_init(void)
{
    // -------------------------------------------------------
    // 1. Configure GPIO pins for UART function
    // -------------------------------------------------------
    // NOTE: The exact PINSEL configuration depends on which UART
    // peripheral is selected (UART0 or UART1) and the specific
    // LPC2148 revision. Update the PINSEL registers below.

    if (LPC2148_UART_PERIPHERAL == 0) {
        // UART0: P0.0 = TX, P0.1 = RX
        // PINSEL0 bits 3:0 for P0.0, 7:4 for P0.1
        * ((volatile unsigned int *) (0xE002C030)) = 0x00000050;
        // Bit 1:0 = 01 -> P0.0 is UART0 TXD0
        // Bit 3:2 = 01 -> P0.1 is UART0 RXD0
    }
    else {
        // UART1: P0.8 = TX, P0.9 = RX
        // PINSEL1 bits 3:0 for P0.8, 7:4 for P0.9
        * ((volatile unsigned int *) (0xE002C034)) = 0x00005000;
    }

    // -------------------------------------------------------
    // 2. Configure UART peripheral
    // -------------------------------------------------------

    // Power up UART0 (set bit 4 of PCONP register)
    * ((volatile unsigned int *) (0xE01FC000)) |= (1 << 4);

    // Reset UART
    * ((volatile unsigned int *) (0xE002C000)) = 1 << 1;  // UTRST

    // Configure baud rate
    // Baud rate formula: PCLK / (16 * DIVM) where DIVM = DLM:DLL+1
    // For LPC2148, PCLK is typically 60 MHz or as configured in your system
    // Here we use a simplified approach assuming 60 MHz PCLK

    // DLM (Divisor Latch High byte) at address 0xE002C008
    // DLL (Divisor Latch Low byte) at address 0xE002C000
    // For 9600 baud @ 60 MHz: DLL = floor(60000000 / (16 * 9600)) = floor(390.625) = 390 = 0x0FA
    // DLM = 0 (for values < 256)

    uint32_t dll_value = 390;  // 60MHz / (16 * 9600) ≈ 390.625

    // Disable divisors first (set DLAB=0 after setting baud)
    // Actually: set DLAB=1, set baud, then set DLAB=0

    // Set DLAB=1 (Divisor Latch Access Bit)
    * ((volatile unsigned int *) (0xE002C000)) |= (1 << 7);

    // Set DLL
    * ((volatile unsigned int *) (0xE002C000)) = dll_value & 0xFF;

    // Set DLM
    * ((volatile unsigned int *) (0xE002C008)) = (dll_value >> 8) & 0xFF;

    // Set DLAB=0 (back to normal operation)
    * ((volatile unsigned int *) (0xE002C000)) &= ~(1 << 7);

    // -------------------------------------------------------
    // 3. Configure Line Control
    // -------------------------------------------------------
    // 8 data bits, 1 stop bit, no parity
    * ((volatile unsigned int *) (0xE002C00C)) = (1 << 3);  // 8 bits

    // -------------------------------------------------------
    // 4. Enable UART Transmitter and Receiver
    // -------------------------------------------------------
    * ((volatile unsigned int *) (0xE002C00C)) |= (1 << 4) | (1 << 3);  // Enable TX and RX

    // -------------------------------------------------------
    // 5. Initialize receive buffer
    // -------------------------------------------------------
    uart_rx_buffer.head = 0;
    uart_rx_buffer.tail = 0;
    uart_rx_buffer.count = 0;

    line_pos = 0;
    line_complete = false;

    // Clear line buffer
    memset(uart_line_buffer, 0, UART_MAX_LINE_LENGTH);
}

/**
 * @internal
 * @brief Put a character into the UART receive buffer (ISR context)
 */
static void uart_rx_putc(char c)
{
    // Add character to circular buffer
    uint8_t next_head = (uart_rx_buffer.head + 1) % UART_RX_BUFFER_SIZE;

    // If buffer is full, overwrite oldest character
    if (next_head != uart_rx_buffer.tail) {
        uart_rx_buffer.data[uart_rx_buffer.head] = c;
        uart_rx_buffer.head = next_head;
        uart_rx_buffer.count++;
    }
}

// =============================================================================
// UART PUT FUNCTIONS (Transmit)
// =============================================================================

void uart_putc(char c)
{
    // Wait until THR (Transmit Holding Register) is empty
    while ((* ((volatile unsigned int *) (0xE002C014))) & (1 << 3)) {
        // Wait - bit 3 of ULSR is Transmit Holding Register Empty
    }

    // Write character to THR
    * ((volatile unsigned int *) (0xE002C014)) = (unsigned char) c;
}

void uart_puts(const char *str)
{
    if (str == NULL) {
        return;
    }

    while (*str) {
        uart_putc(*str);
        str++;
    }
}

// =============================================================================
// UART GET FUNCTIONS (Receive)
// =============================================================================

bool uart_available(void)
{
    return uart_rx_buffer.count > 0;
}

char uart_getc(void)
{
    if (uart_rx_buffer.count == 0) {
        return 0;  // No character available
    }

    // Get character from buffer
    char c = uart_rx_buffer.data[uart_rx_buffer.tail];

    // Advance tail
    uart_rx_buffer.tail = (uart_rx_buffer.tail + 1) % UART_RX_BUFFER_SIZE;
    uart_rx_buffer.count--;

    return c;
}

int uart_read_line(char *buffer, int max_len)
{
    if (buffer == NULL || max_len <= 0) {
        return 0;
    }

    // If a complete line is already available from previous call
    if (line_complete) {
        // Copy line buffer to output
        int len = strlen(uart_line_buffer);
        int copy_len = (len < max_len - 1) ? len : max_len - 1;

        if (copy_len > 0) {
            memcpy(buffer, uart_line_buffer, copy_len);
            buffer[copy_len] = '\0';
        }

        // Reset line state
        line_complete = false;
        line_pos = 0;
        memset(uart_line_buffer, 0, UART_MAX_LINE_LENGTH);

        return copy_len;
    }

    // Read characters until newline or buffer full
    int chars_read = 0;

    while (chars_read < max_len - 1) {
        // Check if a character is available in the receive buffer
        if (uart_rx_buffer.count > 0) {
            char c = uart_getc();

            if (c == '\n' || c == '\r') {
                // End of line
                uart_line_buffer[chars_read] = '\0';
                line_pos = 0;
                line_complete = true;
                return chars_read;
            }
            else {
                // Add to line buffer
                uart_line_buffer[chars_read] = c;
                chars_read++;
            }
        }
        else {
            // No character available, wait a moment and try again
            // In a real RTOS, we'd wait on an event flag
            // For now, just yield/small delay
            for (volatile int i = 0; i < 10; i++) {
                // busy wait
            }
        }
    }

    // Buffer full without newline - truncate
    uart_line_buffer[chars_read] = '\0';
    line_pos = 0;
    line_complete = true;

    int len = strlen(uart_line_buffer);
    return len;
}

bool uart_parse_command(const char *line)
{
    if (line == NULL) {
        return false;
    }

    // Trim whitespace
    char trimmed[UART_MAX_LINE_LENGTH];
    int start = 0;
    int end = (int)strlen(line) - 1;

    // Find start of non-whitespace
    while (start <= end && (line[start] == ' ' || line[start] == '\t' || line[start] == '\n')) {
        start++;
    }

    // Find end of non-whitespace
    while (end >= start && (line[end] == ' ' || line[end] == '\t' || line[end] == '\r')) {
        end--;
    }

    if (start > end) {
        return false;  // Empty line
    }

    // Copy trimmed string
    int i;
    for (i = 0; start + i <= end; i++, end--) {
        // This is getting complicated - let's just use a simpler approach
    }

    // Simpler: just copy from start to end
    int trim_len = end - start + 1;
    memset(trimmed, 0, UART_MAX_LINE_LENGTH);
    for (i = 0; i < trim_len; i++) {
        trimmed[i] = line[start + i];
    }

    // Parse command
    if (strcmp(trimmed, "CLEAR") == 0) {
        lcd_clear();
        return true;
    }
    else if (strcmp(trimmed, "BACKSPACE") == 0) {
        lcd_backspace();
        return true;
    }
    else {
        // Display arbitrary text on LCD
        lcd_puts(trimmed);
        return true;
    }

    return false;
}

// =============================================================================
// LCD MODULE (Placeholders - update GPIO pins for your board)
// =============================================================================

void lcd_init(void)
{
    // -------------------------------------------------------
    // 1. Configure GPIO pins for LCD control
    // -------------------------------------------------------
    // NOTE: The exact GPIO configuration depends on your board.
    // This is a placeholder - update the PINSEL/IODIR registers
    // based on your LPC2148 Evaluation Board schematic.

    // Example for HD44780 4-bit mode:
    // - RS on P0.16
    // - Enable on P0.17
    // - Data pins D4-D7 on P0.18-P0.21
    // Set as output
    // IO0DIR bit 16 = 1 (RS output)
    // IO0DIR bit 17 = 1 (EN output)
    // IO0DIR bits 18-21 = 1 (Data outputs)

    // For now, just print a message
    // TODO: Replace with actual GPIO configuration
    (void)LCD_RS_PIN;
    (void)LCD_RS_PORT;
    (void)LCD_EN_PIN;
    (void)LCD_EN_PORT;
    (void)LCD_D4_PIN;
    (void)LCD_D4_PORT;
    (void)LCD_D5_PIN;
    (void)LCD_D5_PORT;
    (void)LCD_D6_PIN;
    (void)LCD_D6_PORT;
    (void)LCD_D7_PIN;
    (void)LCD_D7_PORT;

    // -------------------------------------------------------
    // 2. HD44780 initialization sequence
    // -------------------------------------------------------

    // Wait for LCD to power up (>= 30ms)
    for (volatile int i = 0; i < 5000; i++) {
        // busy wait
    }

    // Function Set: 8-bit mode, 2 lines, 5x7 font
    // (First part - must send 3 times)
    lcd_putc(0x30);
    for (volatile int i = 0; i < 5000; i++) {
        // busy wait
    }

    lcd_putc(0x30);
    for (volatile int i = 0; i < 5000; i++) {
        // busy wait
    }

    lcd_putc(0x30);
    for (volatile int i = 0; i < 2000; i++) {
        // busy wait
    }

    // Function Set: 4-bit mode, 2 lines, 5x7 font
    lcd_putc(0x20);
    for (volatile int i = 0; i < 2000; i++) {
        // busy wait
    }

    // Function Set final
    // DL=0 (4-bit mode), N=1 (2 lines), F=0 (5x7 font)
    lcd_putc(0x28);
    for (volatile int i = 0; i < 2000; i++) {
        // busy wait
    }

    // Display On/Off Control: Display on, Cursor off, Blink off
    lcd_putc(0x0C);
    for (volatile int i = 0; i < 2000; i++) {
        // busy wait
    }

    // Clear Display
    lcd_putc(0x01);
    for (volatile int i = 0; i < 2000; i++) {
        // busy wait (clear takes ~1.5ms)
    }

    // Entry Mode Set: Increment cursor, no display shift
    lcd_putc(0x06);
    for (volatile int i = 0; i < 2000; i++) {
        // busy wait
    }
}

void lcd_putc(char c)
{
    // -------------------------------------------------------
    // 4-bit data write to LCD
    // -------------------------------------------------------

    // Set data pins (D4-D7)
    // P0.18 = D4, P0.19 = D5, P0.20 = D6, P0.21 = D7

    // RS = 0 for command, 1 for data
    // P0.16 = RS

    if (c & 0x01) {
        // Set D4
        // IO0SET1 bit 2 (or IO0SET0 bit depending on mapping)
        // TODO: Replace with actual GPIO set
        (void)c;
    }

    if (c & 0x02) {
        // Set D5
        // TODO: Replace with actual GPIO set
        (void)c;
    }

    if (c & 0x04) {
        // Set D6
        // TODO: Replace with actual GPIO set
        (void)c;
    }

    if (c & 0x08) {
        // Set D7
        // TODO: Replace with actual GPIO set
        (void)c;
    }

    // Pulse Enable
    // P0.17 = EN
    // Set EN high
    // TODO: IO0SET1 bit 17 = 1
    // Delay minimum 45ns
    for (volatile int i = 0; i < 5; i++) {
        // busy wait
    }

    // Set EN low
    // TODO: IO0CLR1 bit 17 = 1
    for (volatile int i = 0; i < 5; i++) {
        // busy wait
    }
}

void lcd_puts(const char *str)
{
    if (str == NULL) {
        return;
    }

    while (*str) {
        lcd_putc(*str);
        str++;
    }
}

void lcd_clear(void)
{
    lcd_putc(0x01);
    // Clear takes ~1.5ms
    for (volatile int i = 0; i < 5000; i++) {
        // busy wait
    }
}

void lcd_set_cursor(uint8_t col, uint8_t row)
{
    // Set DDRAM address
    // DDRAM address = 0x80 + row * 0x40 + col
    uint8_t addr = 0x80 | (row * 0x40 | col);

    lcd_putc(addr);
}

void lcd_next_line(void)
{
    // Move to next line
    // If at last line, wrap to first line or scroll
    static uint8_t current_row = 0;

    current_row++;
    if (current_row >= LCD_ROWS) {
        current_row = 0;
        // Could implement scroll here
        lcd_clear();
    }

    lcd_set_cursor(0, current_row);
}

// TODO: Implement lcd_print with basic formatting

// TODO: Implement lcd_backspace function