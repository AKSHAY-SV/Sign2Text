/*
 * LPC2148 LCD Module Header
 *
 * LCD display functions for HD44780-compatible character LCDs.
 * Pin definitions are placeholders - update based on your board.
 */

#ifndef LCD_H
#define LCD_H

#include <stdint.h>
#include <stdbool.h>

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

// LCD dimensions (must match uart.h definitions)
#define LCD_COLS 16    // Number of columns (characters per line)
#define LCD_ROWS 2     // Number of rows

// LCD GPIO pin definitions (placeholders - update for your board)
// RS (Register Select) - distinguishes commands from data
#define LCD_RS_PIN 16
#define LCD_RS_PORT 0

// Enable (EN) - latches data on rising edge
#define LCD_EN_PIN 17
#define LCD_EN_PORT 0

// Data pins (4-bit mode)
#define LCD_D4_PIN 18
#define LCD_D4_PORT 0
#define LCD_D5_PIN 19
#define LCD_D5_PORT 0
#define LCD_D6_PIN 20
#define LCD_D6_PORT 0
#define LCD_D7_PIN 21
#define LCD_D7_PORT 0

// =============================================================================
// PUBLIC FUNCTION DECLARATIONS
// =============================================================================

/**
 * @brief Initialize the LCD display.
 *
 * Must be called once before using any other LCD functions.
 * Configures GPIO pins and runs the HD44780 initialization sequence.
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
 * @param col Column position (0 to LCD_COLS-1)
 * @param row Row position (0 to LCD_ROWS-1)
 */
void lcd_set_cursor(uint8_t col, uint8_t row);

/**
 * @brief Move LCD cursor to the next line.
 *        If at the last line, wraps or clears (implementation-dependent).
 */
void lcd_next_line(void);

/**
 * @brief Print formatted text to the LCD.
 *
 * Simple version that writes characters directly.
 * Does not support full printf formatting.
 *
 * @param format Format string
 */
void lcd_print(const char *format, ...);

#endif /* LCD_H */