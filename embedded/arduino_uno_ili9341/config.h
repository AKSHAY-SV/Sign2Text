#ifndef SIGN2TEXT_CONFIG_H
#define SIGN2TEXT_CONFIG_H

#define F_CPU 16000000UL
#define UART_BAUD 115200UL
#define UART_RX_MESSAGE_CAPACITY 129u

/* Confirmed module configuration supplied for this project. */
#ifndef TFT_ROTATION
#define TFT_ROTATION 0u
#endif
#if (TFT_ROTATION == 1u) || (TFT_ROTATION == 3u)
#define TFT_WIDTH 320u
#define TFT_HEIGHT 240u
#else
#define TFT_WIDTH 240u
#define TFT_HEIGHT 320u
#endif

/* Uno hardware SPI: MOSI=D11/PB3, MISO=D12/PB4, SCK=D13/PB5. */
#define TFT_CS_DDR DDRB
#define TFT_CS_PORT PORTB
#define TFT_CS_BIT PB2       /* D10, hardware SS */
#define TFT_DC_DDR DDRB
#define TFT_DC_PORT PORTB
#define TFT_DC_BIT PB1       /* D9 */
#define TFT_RESET_DDR DDRB
#define TFT_RESET_PORT PORTB
#define TFT_RESET_BIT PB0    /* D8 */

#define COLOR_BLACK 0x0000u
#define COLOR_WHITE 0xFFFFu
#define COLOR_NAVY  0x0841u
#define COLOR_CYAN  0x07FFu
#define COLOR_LIME  0x07E0u
#define COLOR_AMBER 0xFD20u

#endif
