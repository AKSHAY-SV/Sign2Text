#include "config.h"
#include "font.h"
#include "ili9341.h"
#include "spi.h"
#include <avr/io.h>
#include <stdbool.h>
#include <util/delay.h>

#define BODY_TOP 48u
#define TEXT_SCALE 2u
#define CHAR_W (6u * TEXT_SCALE)
#define CHAR_H (8u * TEXT_SCALE)
#define TEXT_MARGIN 8u

static void select_display(void) { TFT_CS_PORT &= (uint8_t)~(1 << TFT_CS_BIT); }
static void deselect_display(void) { TFT_CS_PORT |= (1 << TFT_CS_BIT); }
static void data_mode(bool data) {
    if (data) TFT_DC_PORT |= (1 << TFT_DC_BIT);
    else TFT_DC_PORT &= (uint8_t)~(1 << TFT_DC_BIT);
}
static void command(uint8_t value) {
    data_mode(false);
    select_display();
    spi_transfer(value);
    deselect_display();
}

static void address_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1) {
    /* Keep CS asserted from CASET through RAMWR and into the pixel stream. */
    select_display();
    data_mode(false); spi_transfer(0x2A); /* CASET */
    data_mode(true);
    spi_transfer((uint8_t)(x0 >> 8)); spi_transfer((uint8_t)x0);
    spi_transfer((uint8_t)(x1 >> 8)); spi_transfer((uint8_t)x1);
    data_mode(false); spi_transfer(0x2B); /* PASET */
    data_mode(true);
    spi_transfer((uint8_t)(y0 >> 8)); spi_transfer((uint8_t)y0);
    spi_transfer((uint8_t)(y1 >> 8)); spi_transfer((uint8_t)y1);
    data_mode(false); spi_transfer(0x2C); /* RAMWR */
    data_mode(true); /* caller sends RGB565 pixels before deselecting CS */
}

void ili9341_fill_rect(uint16_t x, uint16_t y, uint16_t width, uint16_t height, uint16_t color) {
    uint32_t count;
    if (!width || !height || x >= TFT_WIDTH || y >= TFT_HEIGHT) return;
    if (width > TFT_WIDTH - x) width = TFT_WIDTH - x;
    if (height > TFT_HEIGHT - y) height = TFT_HEIGHT - y;
    address_window(x, y, (uint16_t)(x + width - 1), (uint16_t)(y + height - 1));
    count = (uint32_t)width * height;
    while (count--) { spi_transfer((uint8_t)(color >> 8)); spi_transfer((uint8_t)color); }
    deselect_display();
}

void ili9341_clear(uint16_t color) { ili9341_fill_rect(0, 0, TFT_WIDTH, TFT_HEIGHT, color); }

static void draw_char(uint16_t x, uint16_t y, char c, uint16_t fg, uint16_t bg, uint8_t scale) {
    uint8_t col, row, dx, dy;
    const uint16_t width = (uint16_t)(6u * scale);
    const uint16_t height = (uint16_t)(8u * scale);
    if (x + width > TFT_WIDTH || y + height > TFT_HEIGHT) return;
    address_window(x, y, (uint16_t)(x + width - 1), (uint16_t)(y + height - 1));
    for (row = 0; row < 8; ++row) {
        for (dy = 0; dy < scale; ++dy) {
            for (col = 0; col < 6; ++col) {
                uint8_t bits = col < 5 ? font5x7_column(c, col) : 0;
                uint16_t color = (row < 7 && (bits & (1u << row))) ? fg : bg;
                for (dx = 0; dx < scale; ++dx) {
                    spi_transfer((uint8_t)(color >> 8));
                    spi_transfer((uint8_t)color);
                }
            }
        }
    }
    deselect_display();
}

void ili9341_draw_text(uint16_t x, uint16_t y, const char *text, uint16_t fg,
                       uint16_t bg, uint8_t scale) {
    while (*text && x + 6u * scale <= TFT_WIDTH) {
        draw_char(x, y, *text++, fg, bg, scale);
        x = (uint16_t)(x + 6u * scale);
    }
}

static char next_display_glyph(const char **cursor) {
    const uint8_t first = (uint8_t)**cursor;
    uint8_t continuation_count = 0;
    if (first < 0x80) {
        ++*cursor;
        return (first >= 0x20 && first <= 0x7E) ? (char)first : '?';
    }
    if ((first & 0xE0u) == 0xC0u) continuation_count = 1;
    else if ((first & 0xF0u) == 0xE0u) continuation_count = 2;
    else if ((first & 0xF8u) == 0xF0u) continuation_count = 3;
    ++*cursor;
    while (continuation_count && **cursor && (((uint8_t)**cursor & 0xC0u) == 0x80u)) {
        ++*cursor;
        --continuation_count;
    }
    return '?';
}

static void write_command_data(uint8_t cmd, const uint8_t *values, uint8_t count) {
    data_mode(false);
    select_display();
    spi_transfer(cmd);
    data_mode(true);
    while (count--) spi_transfer(*values++);
    deselect_display();
}

void ili9341_init(void) {
    static const uint8_t madctl[] = { (uint8_t)(TFT_ROTATION == 0 ? 0x48 :
                                      TFT_ROTATION == 1 ? 0x28 :
                                      TFT_ROTATION == 2 ? 0x88 : 0xE8) };
    static const uint8_t pixel_format[] = {0x55};
    static const uint8_t power1[] = {0x23};
    static const uint8_t power2[] = {0x10};
    static const uint8_t vcom1[] = {0x3E, 0x28};
    static const uint8_t vcom2[] = {0x86};
    static const uint8_t frame_rate[] = {0x00, 0x18};
    static const uint8_t display_function[] = {0x08, 0x82, 0x27};
    static const uint8_t gamma[] = {0x01};
    static const uint8_t positive_gamma[] = {
        0x0F,0x31,0x2B,0x0C,0x0E,0x08,0x4E,0xF1,0x37,0x07,0x10,0x03,0x0E,0x09,0x00
    };
    static const uint8_t negative_gamma[] = {
        0x00,0x0E,0x14,0x03,0x11,0x07,0x31,0xC1,0x48,0x08,0x0F,0x0C,0x31,0x36,0x0F
    };
    TFT_CS_DDR |= (1 << TFT_CS_BIT);
    TFT_DC_DDR |= (1 << TFT_DC_BIT);
    TFT_RESET_DDR |= (1 << TFT_RESET_BIT);
    deselect_display();
    TFT_RESET_PORT &= (uint8_t)~(1 << TFT_RESET_BIT); _delay_ms(10);
    TFT_RESET_PORT |= (1 << TFT_RESET_BIT); _delay_ms(120);
    command(0x01); _delay_ms(5);
    command(0x11); _delay_ms(120);
    write_command_data(0xC0, power1, sizeof(power1));
    write_command_data(0xC1, power2, sizeof(power2));
    write_command_data(0xC5, vcom1, sizeof(vcom1));
    write_command_data(0xC7, vcom2, sizeof(vcom2));
    write_command_data(0xB1, frame_rate, sizeof(frame_rate));
    write_command_data(0xB6, display_function, sizeof(display_function));
    write_command_data(0x26, gamma, sizeof(gamma));
    write_command_data(0xE0, positive_gamma, sizeof(positive_gamma));
    write_command_data(0xE1, negative_gamma, sizeof(negative_gamma));
    write_command_data(0x3A, pixel_format, 1);
    write_command_data(0x36, madctl, 1);
    command(0x13);
    command(0x29); _delay_ms(20);
    ili9341_clear(COLOR_BLACK);
    ili9341_draw_text(TEXT_MARGIN, 12, "Sign2Text", COLOR_WHITE, COLOR_BLACK, TEXT_SCALE);
}

void ili9341_show_message(const char *text) {
    uint16_t x = TEXT_MARGIN, y = BODY_TOP;
    bool truncated = false;
    const uint16_t right_edge = TFT_WIDTH - TEXT_MARGIN;
    ili9341_fill_rect(0, BODY_TOP, TFT_WIDTH, (uint16_t)(TFT_HEIGHT - BODY_TOP), COLOR_BLACK);
    if (!text[0]) {
        ili9341_draw_text(x, y, "Waiting for text", COLOR_WHITE, COLOR_BLACK, TEXT_SCALE);
        return;
    }
    while (*text && y + CHAR_H <= TFT_HEIGHT - TEXT_MARGIN) {
        char c = next_display_glyph(&text);
        if (x + CHAR_W > right_edge) { x = TEXT_MARGIN; y = (uint16_t)(y + CHAR_H + 4); }
        draw_char(x, y, c, COLOR_WHITE, COLOR_BLACK, TEXT_SCALE);
        x = (uint16_t)(x + CHAR_W);
    }
    if (*text) truncated = true;
    if (truncated)
        ili9341_draw_text((uint16_t)(TFT_WIDTH - 26), (uint16_t)(TFT_HEIGHT - 40),
                          "...", COLOR_WHITE, COLOR_BLACK, 1);
}
