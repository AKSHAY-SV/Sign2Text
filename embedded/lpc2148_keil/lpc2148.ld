/* Linker script for the NXP LPC2148 (ARM7TDMI-S), GNU ld.
 *
 * Memory map (verify against the LPC2148 datasheet / user manual and the
 * board documentation before programming hardware):
 *
 *   Flash: 512 KB at 0x00000000 - 0x0007FFFF   (LPC2148)
 *   SRAM:  32 KB at 0x40000000 - 0x40007FFF    (device documents 40 KB total
 *           SRAM; the remaining 8 KB requires confirmation against the
 *           lab manual / schematic)
 *
 * The exception vector table is placed first so it lands at
 * 0x00000000, where the ARM7TDMI looks for the reset vector on power-on
 * reset. Flash Magic programs the Intel HEX image into flash starting at
 * address 0x00000000.
 */

ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 512K
    RAM   (rwx) : ORIGIN = 0x40000000, LENGTH = 32K
}

SECTIONS
{
    /* Interrupt vectors must be at the start of flash (0x00000000). */
    .vectors : {
        . = ALIGN(4);
        KEEP(*(.vectors))
        . = ALIGN(4);
    } > FLASH

    .text : {
        . = ALIGN(4);
        *(.text*)
        . = ALIGN(4);
    } > FLASH

    .rodata : {
        . = ALIGN(4);
        *(.rodata*)
        . = ALIGN(4);
    } > FLASH

    /* Initialised data: VMA in RAM, load address (LMA) in flash.
       The startup code copies _sdata.._edata from _sidata at reset. */
    _sidata = LOADADDR(.data);
    .data : {
        . = ALIGN(4);
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    /* Zero-initialised data in RAM. */
    .bss : {
        . = ALIGN(4);
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    _end = .;

    /* Stack symbols (full-descending stacks grow down from the top of RAM).
       Each mode gets a 1 KiB stack; the actual SP is programmed by
       startup.s. No RAM section is emitted for the stacks. */
    _estack    = ORIGIN(RAM) + LENGTH(RAM);   /* 0x40008000 */
    _stack_top_svc = _estack;
    _stack_top_irq = _estack - 0x400;
    _stack_top_fiq = _estack - 0x800;
    _stack_top_und = _estack - 0xC00;
    _stack_top_abt = _estack - 0x1000;

    /DISCARD/ : {
        *(.ARM.attributes)
        *(.comment)
    }
}
