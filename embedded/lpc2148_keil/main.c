#include "LPC214x.h"
#include "lcd.h"
#include "uart.h"

#define MESSAGE_CAPACITY 32u
static char message[MESSAGE_CAPACITY + 1u];
static char incoming[MESSAGE_CAPACITY + 1u];
static unsigned int incoming_len = 0;

static void display(const char *text) {
    unsigned int i = 0;
    while (i < MESSAGE_CAPACITY && text[i] != '\0') {
        message[i] = text[i];
        ++i;
    }
    while (i < MESSAGE_CAPACITY) message[i++] = '\0';
    message[MESSAGE_CAPACITY] = '\0';
    lcd_show_message(message);
}
int main(void) {
    int ch;
    unsigned int i;
    for (i = 0; i <= MESSAGE_CAPACITY; ++i) message[i] = incoming[i] = 0;
    lcd_init();
    uart0_init();
    display("Sign2Text Ready");
    for (;;) {
        ch = uart0_getchar_nonblocking();
        if (ch < 0) continue;
        if (ch == '\r') continue;
        if (ch == '\n') {
            incoming[incoming_len] = '\0';
            display(incoming);
            /* Clear the buffer so a shorter follow-up line cannot reveal
               stale characters from a previous longer line. */
            for (i = 0; i <= MESSAGE_CAPACITY; ++i) incoming[i] = 0;
            incoming_len = 0;
        } else if (ch >= 32 && ch <= 126) {
            if (incoming_len < MESSAGE_CAPACITY) incoming[incoming_len++] = (char)ch;
            /* Excess printable bytes are discarded until the line terminator. */
        }
    }
}
