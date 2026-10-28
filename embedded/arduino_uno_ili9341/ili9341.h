#ifndef ILI9341_H
#define ILI9341_H
#include <stdint.h>
void ili9341_init(void);
void ili9341_clear(uint16_t color);
void ili9341_fill_rect(uint16_t x, uint16_t y, uint16_t width, uint16_t height, uint16_t color);
void ili9341_draw_text(uint16_t x, uint16_t y, const char *text, uint16_t foreground,
                       uint16_t background, uint8_t scale);
void ili9341_show_message(const char *text);
#endif
