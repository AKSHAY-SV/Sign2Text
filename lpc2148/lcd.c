/*
 * LPC2148 LCD Module Implementation
 *
 * HD44780-compatible character LCD display driver.
 * Pin definitions are placeholders - update based on your board.
 *
 * NOTE: This is a minimal implementation. Full functionality
 * (including custom characters, scrolling, etc.) can be added later.
 */

#include "lcd.h"
#include <LPC214x.h>  // LPC2148 peripheral registers
#include <stdio.h>
#include <string.h>

// =============================================================================
// LOCAL CONSTANTS AND DEFINITIONS
// =============================================================================

// LCD commands
#define LCD_CLEAR_DISPLAY 0x01
#define LCD_RETURN_HOME 0x02
#define LCD_ENTRY_MODE_SET 0x04
#define LCD_DISPLAY_CONTROL 0x08
#define LCD_CURSOR_SHIFT 0x10
#define LCD_FUNCTION_SET 0x20
#define LCD_SET_CG_ADDR 0x40
#define LCD_SET_DD_ADDR 0x80

// Entry mode flags
#define LCD_ENTRY_RIGHT 0x00
#define LCD_ENTRY_LEFT 0x02
#define LCD_SHIFT_INCREMENT 0x00
#define LCD_SHIFT_DECREMENT 0x01

// Display control flags
#define LCD_DISPLAY_ON 0x04
#define LCD_CURSOR_ON 0x02
#define LCD_BLINK_ON 0x01

// Function set flags
#define LCD_8BIT_MODE 0x10
#define LCD_4BIT_MODE 0x00
#define LCD_2LINE 0x08
#define LCD_1LINE 0x00
#define LCD_5x10_DOTS 0x04
#define LCD_5x7_DOTS 0x00

// =============================================================================
// LOCAL VARIABLES
// =============================================================================

// Track current cursor position
static uint8_t current_col = 0;
static uint8_t current_row = 0;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 @brief Delay for a specified number of milliseconds (busy wait).
 @note In a real application, use a proper timer-based delay.
 */
static void lcd_delay_ms(uint32_t ms)
{
    for (volatile uint32_t i = 0; i < ms * 5000; i++) {
        // busy wait - approximate only
    }
}

/**
 @brief Delay for a specified number of microseconds (busy wait).
 */
static void lcd_delay_us(uint32_t us)
{
    for (volatile uint32_t i = 0; i < us * 5; i++) {
        // busy wait - approximate only
    }
}

/**
 @brief Write a nibble (4 bits) to the LCD data pins.
 @param nibble The 4-bit value to write (only lower 4 bits used)
 */
static void lcd_write_nibble(uint8_t nibble)
{
    // Set data pins D4-D7
    // Clear data pins first
    // TODO: Replace with actual GPIO clear operations

    // Set individual data bits
    if (nibble & 0x01) {
        // Set D4
        // TODO: IO0SET0 bit 18 or IO0SET1 bit 2
    }
    if (nibble & 0x02) {
        // Set D5
        // TODO: Replace with actual GPIO set
    }
    if (nibble & 0x04) {
        // Set D6
        // TODO: Replace with actual GPIO set
    }
    if (nibble & 0x08) {
        // Set D7
        // TODO: Replace with actual GPIO set
    }

    // Pulse Enable pin high-to-low
    // TODO: Set EN high, delay, set EN low
    // Typical minimum pulse width: 45ns
    lcd_delay_us(1);
}

/**
 @brief Send a command to the LCD.
 @param cmd LCD command byte
 */
static void lcd_send_command(uint8_t cmd)
{
    // RS = 0 for command
    // TODO: Set LCD_RS_PIN low

    // Upper nibble
    lcd_write_nibble(cmd >> 4);

    // Lower nibble
    lcd_write_nibble(cmd & 0x0F);

    // Enable pulse - wait for command to execute
    // Clear display takes ~1.5ms, others ~40us
    lcd_delay_ms(2);
}

/**
 @brief Send data (character) to the LCD.
 @param c Character to send (ASCII)
 */
static void lcd_send_data(uint8_t c)
{
    // RS = 1 for data
    // TODO: Set LCD_RS_PIN high

    // Upper nibble
    lcd_write_nibble(c >> 4);

    // Lower nibble
    lcd_write_nibble(c & 0x0F);

    // Enable pulse
    lcd_delay_us(50);
}

// =============================================================================
// PUBLIC FUNCTION IMPLEMENTATIONS
// =============================================================================

void lcd_init(void)
{
    // -------------------------------------------------------
    // 1. Configure GPIO pins for LCD
    // -------------------------------------------------------
    // Set LCD control pins as outputs
    // TODO: Replace with actual GPIO direction configuration
    // Based on your board:
    // - LCD_RS_PIN on LCD_RS_PORT -> output
    // - LCD_EN_PIN on LCD_EN_PORT -> output
    // - LCD_D4_PIN-LCD_D7_PIN on LCD_D4_PORT-LCD_D7_PORT -> output

    // For now, just initialize the hardware structure
    // TODO: Actual GPIO setup needed

    // -------------------------------------------------------
    // 2. HD44780 initialization sequence
    // -------------------------------------------------------

    // Wait for LCD to power up (>= 30ms)
    lcd_delay_ms(50);

    // Function Set: 8-bit mode, 2 lines, 5x7 font
    // (First part - must send 3 times during power-up)
    lcd_send_command(0x30);
    lcd_delay_ms(5);

    lcd_send_command(0x30);
    lcd_delay_ms(1);

    lcd_send_command(0x30);
    lcd_delay_ms(1);

    // Function Set: 4-bit mode, 2 lines, 5x7 font
    lcd_send_command(0x20);
    lcd_delay_ms(1);

    // Function Set final: DL=0 (4-bit), N=1 (2 lines), F=0 (5x7 dots)
    lcd_send_command(LCD_FUNCTION_SET | LCD_4BIT_MODE | LCD_2LINE | LCD_5x7_DOTS);
    lcd_delay_ms(1);

    // Display On/Off Control: Display on, Cursor off, Blink off
    lcd_send_command(LCD_DISPLAY_CONTROL | LCD_DISPLAY_ON);
    lcd_delay_ms(1);

    // Clear Display
    lcd_send_command(LCD_CLEAR_DISPLAY);
    lcd_delay_ms(2);  // Clear takes ~1.5ms

    // Entry Mode Set: Increment cursor, no display shift
    lcd_send_command(LCD_ENTRY_MODE_SET | LCD_ENTRY_LEFT);
    lcd_delay_ms(1);

    // Set cursor to home position
    lcd_set_cursor(0, 0);
}

void lcd_putc(char c)
{
    // Don't update cursor if at end of line and we're about to wrap
    if (c == '\n') {
        lcd_next_line();
        return;
    }

    // Send character to LCD
    lcd_send_data((uint8_t)c);

    // Update cursor position
    current_col++;

    // If cursor exceeds columns, move to next line
    if (current_col >= LCD_COLS) {
        lcd_next_line();
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
    lcd_send_command(LCD_CLEAR_DISPLAY);
    lcd_delay_ms(2);  // Clear takes ~1.5ms

    // Reset cursor position
    current_col = 0;
    current_row = 0;
}

void lcd_set_cursor(uint8_t col, uint8_t row)
{
    // Validate column and row
    if (col >= LCD_COLS) {
        col = LCD_COLS - 1;
    }
    if (row >= LCD_ROWS) {
        row = LCD_ROWS - 1;
    }

    // Set DDRAM address: 0x80 + row * 0x40 + col
    uint8_t addr = 0x80 | (row * 0x40 | col);

    lcd_send_command(addr);
    current_col = col;
    current_row = row;
}

void lcd_next_line(void)
{
    // Move cursor to the beginning of the next line
    // If at the last line, wrap to first line

    if (current_row < LCD_ROWS - 1) {
        current_row++;
        lcd_set_cursor(0, current_row);
    }
    else {
        // At last line - wrap to first line or clear
        // Simple implementation: clear and go to first line
        lcd_clear();
    }
}

// TODO: Implement lcd_print with basic sprintf-like formatting