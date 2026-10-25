#ifndef LPC214X_H
#define LPC214X_H

/* Minimal LPC2148 device register definitions for GNU Arm GCC.
 *
 * This replaces the Keil LPC214x.h device header so the firmware builds
 * with arm-none-eabi-gcc. It covers only the registers used by this
 * project (PINSEL0, VPBDIV, UART0, GPIO0).
 *
 * Addresses follow the NXP LPC214x APB memory map. Verify every address
 * against the LPC2148 datasheet / user manual and the board schematic
 * before hardware use; this header does not come from a Keil device pack.
 *
 *   PINSEL0  0xE002C000   PINSEL register 0 (UART0 TXD0/RXD0 pin select)
 *   VPBDIV   0xE01FC100   VPB clock divider (PCLK vs CCLK)
 *   UART0    0xE000C000   UART0 register block
 *   GPIO0    0xE0028000   GPIO port 0 register block (LCD data/control)
 */

/* Pin connect block */
#define PINSEL0  (*(volatile unsigned long *)0xE002C000UL)

/* VPB divider */
#define VPBDIV   (*(volatile unsigned long *)0xE01FC100UL)

/* UART0 registers (base 0xE000C000; DLM/DLL share addresses with IER/RBR
   depending on the DLAB bit of U0LCR) */
#define U0RBR    (*(volatile unsigned long *)0xE000C000UL)
#define U0THR    (*(volatile unsigned long *)0xE000C000UL)
#define U0DLL    (*(volatile unsigned long *)0xE000C000UL)
#define U0DLM    (*(volatile unsigned long *)0xE000C004UL)
#define U0IER    (*(volatile unsigned long *)0xE000C004UL)
#define U0IIR    (*(volatile unsigned long *)0xE000C008UL)
#define U0FCR    (*(volatile unsigned long *)0xE000C008UL)
#define U0LCR    (*(volatile unsigned long *)0xE000C00CUL)
#define U0LSR    (*(volatile unsigned long *)0xE000C014UL)

/* GPIO0 registers (base 0xE0028000) */
#define IO0DIR   (*(volatile unsigned long *)0xE0028000UL)
#define IO0SET   (*(volatile unsigned long *)0xE0028004UL)
#define IO0CLR   (*(volatile unsigned long *)0xE0028008UL)
#define IO0PIN   (*(volatile unsigned long *)0xE002800CU)

#endif /* LPC214X_H */
