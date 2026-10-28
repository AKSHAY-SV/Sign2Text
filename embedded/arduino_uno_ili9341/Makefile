MCU := atmega328p
F_CPU := 16000000UL
PROGRAMMER ?= arduino
PORT ?= /dev/ttyACM0
UPLOAD_BAUD ?= 115200
ROTATION ?= 0
CC := avr-gcc
OBJCOPY := avr-objcopy
AVRDUDE := avrdude
CFLAGS := -mmcu=$(MCU) -DF_CPU=$(F_CPU) -DTFT_ROTATION=$(ROTATION) -Os -std=gnu11 -Wall -Wextra -Werror -Wshadow -Wstrict-prototypes -Wmissing-prototypes -ffunction-sections -fdata-sections
LDFLAGS := -mmcu=$(MCU) -Wl,--gc-sections
SOURCES := main.c uart.c spi.c ili9341.c font.c
OBJECTS := $(SOURCES:%.c=build/%.o)
ELF := build/sign2text_ili9341.elf
HEX := build/firmware.hex
DIAGNOSTIC_ELF := build/display_diagnostic.elf
DIAGNOSTIC_HEX := build/display_diagnostic.hex

.PHONY: all clean flash verify firmware diagnostic
all: firmware diagnostic

firmware: $(ELF) $(HEX) verify

build:
	mkdir -p $@

build/%.o: %.c config.h | build
	$(CC) $(CFLAGS) -c $< -o $@

$(ELF): $(OBJECTS) | build
	$(CC) $(LDFLAGS) $^ -o $@

$(HEX): $(ELF) | build
	$(OBJCOPY) -O ihex -R .eeprom $< $@

verify: $(ELF) $(HEX)
	test -s $(ELF)
	test -s $(HEX)

$(DIAGNOSTIC_ELF): display_diagnostic.c spi.c ili9341.c font.c config.h | build
	$(CC) $(CFLAGS) -o $@ display_diagnostic.c spi.c ili9341.c font.c

$(DIAGNOSTIC_HEX): $(DIAGNOSTIC_ELF) | build
	$(OBJCOPY) -O ihex -R .eeprom $< $@

diagnostic: $(DIAGNOSTIC_ELF) $(DIAGNOSTIC_HEX)
	test -s $(DIAGNOSTIC_ELF)
	test -s $(DIAGNOSTIC_HEX)

flash: $(HEX)
	test -s $(HEX)
	$(AVRDUDE) -p $(MCU) -c $(PROGRAMMER) -P $(PORT) -b $(UPLOAD_BAUD) -D -U flash:w:$(HEX):i

clean:
	rm -rf build
