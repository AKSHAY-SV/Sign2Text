#include "LPC214x.h"
#include "lcd.h"

/* Manual mapping: RS=P0.2, EN=P0.3, D4..D7=P0.4..P0.7; R/W grounded. */
#define LCD_RS (1u << 2)
#define LCD_EN (1u << 3)
#define LCD_DATA (0xFu << 4)
#define LCD_PINS (LCD_RS | LCD_EN | LCD_DATA)

static void delay(volatile unsigned int n) { while (n--) { ; } }
static void pulse(void) { IO0SET = LCD_EN; delay(2000); IO0CLR = LCD_EN; delay(2000); }
static void nibble(unsigned char v) {
    IO0CLR = LCD_DATA;
    IO0SET = ((unsigned int)(v & 0x0F) << 4);
    pulse();
}
static void send(unsigned char value, int data) {
    if (data) IO0SET = LCD_RS; else IO0CLR = LCD_RS;
    nibble((unsigned char)(value >> 4));
    nibble(value);
    delay(3000);
}
static void command(unsigned char value) {
    send(value, 0);
    if (value == 0x01 || value == 0x02) delay(100000);
}
static void put_char(char c) { send((unsigned char)c, 1); }

void lcd_init(void) {
    IO0DIR |= LCD_PINS;
    IO0CLR = LCD_PINS;
    delay(300000);
    nibble(0x03); delay(100000);
    nibble(0x03); delay(30000);
    nibble(0x03); delay(30000);
    nibble(0x02); delay(30000);
    command(0x28); /* 4-bit, two lines, 5x8 */
    command(0x0C); command(0x06); command(0x01);
}
void lcd_clear(void) { command(0x01); }
void lcd_write_line(const char *text) {
    unsigned int i;
    for (i = 0; i < 16; ++i) put_char(text[i] ? text[i] : ' ');
}
void lcd_show_message(const char *text) {
    unsigned int i;
    command(0x80);
    for (i = 0; i < 16; ++i) put_char(text[i] ? text[i] : ' ');
    command(0xC0);
    for (i = 16; i < 32; ++i) put_char(text[i] ? text[i] : ' ');
}
