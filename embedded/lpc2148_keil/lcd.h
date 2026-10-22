#ifndef LCD_H
#define LCD_H
void lcd_init(void);
void lcd_clear(void);
void lcd_write_line(const char *text);
void lcd_show_message(const char *text);
#endif
