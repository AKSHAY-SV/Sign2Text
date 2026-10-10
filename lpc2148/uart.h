/*
 * LPC2148 UART Module
 *
 * UART initialization and receive functions for communication with
 * the laptop host running Sign2Text.
 *
 * Protocol: Receives ASCII characters with newline line ending.
 * Characters are buffered and can be parsed as commands.
 *
 * NOTE: GPIO pin definitions below are placeholders.
 * Update them based on your LPC2148 Evaluation Board schematic.
 */

#ifndef UART_H
#define UART_H

#include <stdint.h>
#include <stdbool.h>

// =============================================================================
// UART CONFIGURATION - UPDATE THESE FOR YOUR BOARD
// =============================================================================

// UART peripheral to use: 0 or 1
#define LPC2148_UART_PERIPHERAL 0

// Baud rate for communication with laptop
#define LPC2148_UART_BAUD 9600

// UART TX pin - Transmits data to laptop
// Common defaults: P0.0 for UART0, P0.8 for UART1
// Update based on your board schematic!
#define LPC2148_UART_TX_PIN 0   // P0.0
#define LPC2148_UART_TX_PORT 0  // Port 0

// UART RX pin - Receives data from laptop
// Common defaults: P0.1 for UART0, P0.9 for UART1
// Update based on your board schematic!
#define LPC2148_UART_RX_PIN 1   // P0.1
#define LPC2148_UART_RX_PORT 0  // Port 0

// =============================================================================
// LCD CONFIGURATION - UPDATE THESE FOR YOUR BOARD
// =============================================================================

// LCD controller type
#define LCD_TYPE_HD44780 1
#define LCD_TYPE_PCA9535 2

// Default LCD type - update based on your board
#ifndef LCD_TYPE
#define LCD_TYPE LCD_TYPE_HD44780
#endif

// LCD pins - HD44780 4-bit mode interface
// These are placeholders - update based on your board schematic!

// Register Select (RS) - Selects command vs data register
#define LCD_RS_PIN 16
#define LCD_RS_PORT 0

// Enable (EN) - Pulse to latch data
#define LCD_EN_PIN 17
#define LCD_EN_PORT 0

// Data pins (4-bit mode: D4, D5, D6, D7)
#define LCD_D4_PIN 18
#define LCD_D4_PORT 0
#define LCD_D5_PIN 19
#define LCD_D5_PORT 0
#define LCD_D6_PIN 20
#define LCD_D6_PORT 0
#define LCD_D7_PIN 21
#define LCD_D7_PORT 0

// LCD dimensions
#define LCD_COLS 16    // Number of columns (characters per line)
#define LCD_ROWS 2     // Number of rows

// =============================================================================
// PUBLIC FUNCTION DECLARATIONS
// =============================================================================

/**
 * @brief Initialize the UART peripheral for communication with laptop.
 *
 * Sets up the UART baud rate, enables the UART, and configures
 * the GPIO pins for TX and RX.
 *
 * Must be called before any other UART functions.
 */
void uart_init(void);

/**
 * @brief Send a single character over UART to the laptop.
 *
 * @param c Character to send (ASCII)
 */
void uart_putc(char c);

/**
 * @brief Send a null-terminated string over UART to the laptop.
 *
 * @param str String to send (must be null-terminated)
 */
void uart_puts(const char *str);

/**
 * @brief Check if a character is available in the UART receive buffer.
 *
 * @return true if a character is available, false otherwise
 */
bool uart_available(void);

/**
 * @brief Read a single character from the UART receive buffer.
 *
 * @return The received character (ASCII), or 0 if no character available
 */
char uart_getc(void);

/**
 * @brief Read a line of text from the UART receive buffer.
 *         Reading stops at newline character or buffer full.
 *
 * @param[out] buffer Buffer to store the received line
 * @param[in]  max_len Maximum number of characters to read (including null terminator)
 * @return Number of characters read (excluding null terminator), or 0 if no data
 */
int uart_read_line(char *buffer, int max_len);

/**
 * @brief Parse a received line of text and execute the corresponding command.
 *
 * Supported commands:
 *   - "CLEAR"   - Clear the LCD display
 *   - "BACKSPACE" - Simulate backspace (remove last character)
 *   - Any other text - Display on LCD
 *
 * @param line Null-terminated string received from UART
 * @return true if a command was recognized and executed, false otherwise
 */
bool uart_parse_command(const char *line);

/**
 * @brief Initialize the LCD display.
 *
 * Configures the LCD control pins and initializes the HD44780
 * controller (or other LCD type) for operation.
 *
 * Must be called once before using LCD display functions.
 */
void lcd_init(void);

/**
 * @brief Write a single character to the LCD display.
 *
 * @param c Character to display (ASCII)
 */
void lcd_putc(char c);

/**
 * @brief Write a null-terminated string to the LCD display.
 *
 * @param str String to display (must be null-terminated)
 */
void lcd_puts(const char *str);

/**
 * @brief Clear the LCD display and move cursor to home position.
 */
void lcd_clear(void);

/**
 * @brief Set LCD cursor position.
 *
 * @param col Column (0 to LCD_COLS-1)
 * @param row Row (0 to LCD_ROWS-1)
 */
void lcd_set_cursor(uint8_t col, uint8_t row);

/**
 * @brief Move LCD cursor to the next line.
 *        If at the last line, scrolls or clears depending on implementation.
 */
void lcd_next_line(void);

/**
 * @brief Print formatted text to the LCD (simple version).
 *
 * Writes characters directly to LCD; does not support full printf formatting.
 *
 * @param format Format string (limited support)
 * @param ... Arguments (limited support)
 */
void lcd_print(const char *format, ...);

#endif /* UART_H */