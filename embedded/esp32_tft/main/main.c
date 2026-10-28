/*
 * SPDX-FileCopyrightText: 2021-2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: CC0-1.0
 */

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/lock.h>
#include <sys/param.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_err.h"
#include "esp_log.h"
#include "driver/uart.h"
#include "lvgl.h"

// ILI9341 display configuration using esp_lcd
#if CONFIG_EXAMPLE_LCD_CONTROLLER_ILI9341
#include "esp_lcd_ili9341.h"
#endif

static const char *TAG = "sign2text_esp32";

// Using SPI2 in the example
#define LCD_HOST  SPI2_HOST

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// Please update the following configuration according to your LCD spec //////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define EXAMPLE_LCD_PIXEL_CLOCK_HZ     (20 * 1000 * 1000)
#define EXAMPLE_LCD_BK_LIGHT_ON_LEVEL  1
#define EXAMPLE_LCD_BK_LIGHT_OFF_LEVEL !EXAMPLE_LCD_BK_LIGHT_ON_LEVEL
#define EXAMPLE_PIN_NUM_SCLK           18
#define EXAMPLE_PIN_NUM_MOSI           19
#define EXAMPLE_PIN_NUM_MISO           21
#define EXAMPLE_PIN_NUM_LCD_DC         5
#define EXAMPLE_PIN_NUM_LCD_RST        3
#define EXAMPLE_PIN_NUM_LCD_CS         4
#define EXAMPLE_PIN_NUM_BK_LIGHT       2

// The pixel number in horizontal and vertical
#if CONFIG_EXAMPLE_LCD_CONTROLLER_ILI9341
#define EXAMPLE_LCD_H_RES              240
#define EXAMPLE_LCD_V_RES              320
#elif CONFIG_EXAMPLE_LCD_CONTROLLER_GC9A01
#define EXAMPLE_LCD_H_RES              240
#define EXAMPLE_LCD_V_RES              240
#endif
// Bit number used to represent command and parameter
#define EXAMPLE_LCD_CMD_BITS           8
#define EXAMPLE_LCD_PARAM_BITS         8

#define EXAMPLE_LVGL_DRAW_BUF_LINES    20 // number of display lines in each draw buffer
#define EXAMPLE_LVGL_TICK_PERIOD_MS    2
#define EXAMPLE_LVGL_TASK_MAX_DELAY_MS 500
#define EXAMPLE_LVGL_TASK_MIN_DELAY_MS 1000 / CONFIG_FREERTOS_HZ
#define EXAMPLE_LVGL_TASK_STACK_SIZE   (4 * 1024)
#define EXAMPLE_LVGL_TASK_PRIORITY     2

// UART configuration for serial reception
#define UART_PORT UART_NUM_0
#define UART_BAUD_RATE 115200
#define UART_TX_GPIO 16
#define UART_RX_GPIO 17

// Task handle for UART reception
TaskHandle_t uart_task_handle = NULL;

// LVGL draw buffers
static lv_color_t *buf1 = NULL;
static lv_color_t *buf2 = NULL;

// LVGL display handle
static lv_display_t *display = NULL;

// Forward declarations
static void example_lvgl_init(void);
static void example_lvgl_flush_cb(lv_display_t *disp, const lv_area_t *area, uint8_t *px_map);
static void example_increase_lvgl_tick(void *arg);
static void example_draw_text(const char *text);
static void uart_task(void *arg);

// app_main entry point
void app_main(void)
{
    ESP_LOGI(TAG, "Initializing display...");
    example_lvgl_init();

    ESP_LOGI(TAG, "Configuring UART...");
    ESP_ERROR_CHECK(uart_driver_install(UART_PORT, 256, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_set_pin(UART_PORT, UART_TX_GPIO, UART_RX_GPIO, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    // Create task to handle UART reception
    xTaskCreate(uart_task, "uart_task", 4096, NULL, 5, &uart_task_handle);

    ESP_LOGI(TAG, "System initialized - waiting for serial text input");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// LCD and LVGL initialization functions //////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Initialize the display and LVGL
static void example_lvgl_init(void)
{
    ESP_LOGI(TAG, "Initialize SPI bus");
    spi_bus_config_t buscfg = {
        .sclk_io_num = EXAMPLE_PIN_NUM_SCLK,
        .mosi_io_num = EXAMPLE_PIN_NUM_MOSI,
        .miso_io_num = EXAMPLE_PIN_NUM_MISO,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = EXAMPLE_LCD_H_RES * 80 * sizeof(lv_color16_t),
    };
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &buscfg, SPI_DMA_CH_AUTO));

    ESP_LOGI(TAG, "Install panel IO");
    esp_lcd_panel_io_handle_t io_handle = NULL;
    esp_lcd_panel_io_spi_config_t io_config = {
        .dc_gpio_num = EXAMPLE_PIN_NUM_LCD_DC,
        .cs_gpio_num = EXAMPLE_PIN_NUM_LCD_CS,
        .pclk_hz = EXAMPLE_LCD_PIXEL_CLOCK_HZ,
        .lcd_cmd_bits = EXAMPLE_LCD_CMD_BITS,
        .lcd_param_bits = EXAMPLE_LCD_PARAM_BITS,
        .spi_mode = 0,
        .trans_queue_depth = 10,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_config, &io_handle));

    esp_lcd_panel_handle_t panel_handle = NULL;
    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = EXAMPLE_PIN_NUM_LCD_RST,
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_BGR,
        .bits_per_pixel = 16,
    };
#if CONFIG_EXAMPLE_LCD_CONTROLLER_ILI9341
    ESP_LOGI(TAG, "Install ILI9341 panel driver");
    ESP_ERROR_CHECK(esp_lcd_new_panel_ili9341(io_handle, &panel_config, &panel_handle));
#endif

    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel_handle, true, false));

    // Turn on the screen
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel_handle, true));

    ESP_LOGI(TAG, "Turn on LCD backlight");
    gpio_config_t bk_gpio_config = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = 1ULL << EXAMPLE_PIN_NUM_BK_LIGHT
    };
    ESP_ERROR_CHECK(gpio_config(&bk_gpio_config));

    // Initialize LVGL
    ESP_LOGI(TAG, "Initialize LVGL library");
    lv_init();

    // Allocate draw buffers
    size_t draw_buffer_sz = EXAMPLE_LCD_H_RES * EXAMPLE_LVGL_DRAW_BUF_LINES * sizeof(lv_color16_t);
    buf1 = heap_caps_malloc(draw_buffer_sz, MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    buf2 = heap_caps_malloc(draw_buffer_sz, MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    assert(buf1 && buf2);

    // Initialize LVGL draw buffers
    lv_display_set_buffers(display, buf1, buf2, draw_buffer_sz, LV_DISPLAY_RENDER_MODE_PARTIAL);

    // Create the display
    display = lv_display_create(EXAMPLE_LCD_H_RES, EXAMPLE_LCD_V_RES);

    // Set color depth
    lv_display_set_color_format(display, LV_COLOR_FORMAT_RGB565);

    // Set the flush callback
    lv_display_set_flush_cb(display, example_lvgl_flush_cb);

    // Create LVGL tick timer
    const esp_timer_create_args_t lvgl_tick_timer_args = {
        .callback = &example_increase_lvgl_tick,
        .name = "lvgl_tick"
    };
    esp_timer_handle_t lvgl_tick_timer = NULL;
    ESP_ERROR_CHECK(esp_timer_create(&lvgl_tick_timer_args, &lvgl_tick_timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(lvgl_tick_timer, EXAMPLE_LVGL_TICK_PERIOD_MS * 1000));
}

// LVGL flush callback - called by LVGL to copy the rendered buffer to the display
static void example_lvgl_flush_cb(lv_display_t *disp, const lv_area_t *area, uint8_t *px_map)
{
    // Mark the area as flushed
    lv_display_flush_ready(disp);
}

// LVGL tick increment function
static void example_increase_lvgl_tick(void *arg)
{
    lv_tick_inc(EXAMPLE_LVGL_TICK_PERIOD_MS);
}

// Simple text rendering function using LVGL
static void example_draw_text(const char *text)
{
    if (text == NULL) return;

    // Create a label with the received text
    lv_obj_t *label = lv_label_create(lv_display_get_screen_active(display));
    lv_label_set_text(label, text);

    // Center the label
    lv_obj_center(label);
}

// UART task to receive and display text
static void uart_task(void *arg)
{
    ESP_LOGI(TAG, "UART task started");

    uint8_t data[128];
    int len;
    char text_buf[256] = {0};
    int text_idx = 0;

    while (1) {
        // Read data from UART
        len = uart_read_bytes(UART_PORT, data, sizeof(data) - 1, pdMS_TO_TICKS(10));

        if (len > 0) {
            data[len] = '\0';

            // Process received characters
            for (int i = 0; i < len; i++) {
                char c = data[i];

                if (c == '\n' || c == '\r') {
                    // Null-terminate the text buffer
                    text_buf[text_idx] = '\0';

                    // Draw the text on the display
                    example_draw_text(text_buf);
                    ESP_LOGI(TAG, "Received text: %s", text_buf);

                    // Reset buffer
                    text_idx = 0;
                    memset(text_buf, 0, sizeof(text_buf));
                } else {
                    // Add character to buffer (limit size)
                    if (text_idx < sizeof(text_buf) - 1) {
                        text_buf[text_idx++] = c;
                    }
                }
            }
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}