/* startup.s - LPC2148 (ARM7TDMI-S) startup code for GNU Arm Embedded GCC.
 *
 * Provides:
 *   - exception vector table at the start of flash (0x00000000)
 *   - reset handler: disables IRQ/FIQ, sets up a stack for each required
 *     processor mode (SVC, IRQ, FIQ, Undefined, Abort), copies .data from
 *     flash to RAM, clears .bss, then branches to main()
 *   - default (infinite-loop) handlers for the unused exceptions
 *
 * Built with: arm-none-eabi-gcc -mcpu=arm7tdmi (ARM state, not Thumb)
 */

.syntax unified
.arch armv4t

.text

.globl __vectors
.globl Reset_Handler
.globl Und_Handler
.globl SWI_Handler
.globl PAbort_Handler
.globl DAbort_Handler
.globl IRQ_Handler
.globl FIQ_Handler

/* Exception vector table. The ARM7TDMI fetches the reset PC from address 0,
 * so this table must be linked at the very beginning of flash (0x00000000).
 */
.section .vectors, "ax", %progbits
__vectors:
    b Reset_Handler       /* 0x00 Reset */
    b Und_Handler         /* 0x04 Undefined instruction */
    b SWI_Handler         /* 0x08 Software interrupt */
    b PAbort_Handler      /* 0x0C Prefetch abort */
    b DAbort_Handler      /* 0x10 Data abort */
    b .                   /* 0x14 Reserved */
    b IRQ_Handler         /* 0x18 IRQ */
    b FIQ_Handler         /* 0x1C FIQ */

.text
.align 2

/* Default exception handlers: stay in a tight loop. */
Und_Handler:   b .
SWI_Handler:   b .
PAbort_Handler:b .
DAbort_Handler:b .
IRQ_Handler:   b .
FIQ_Handler:   b .

.type Reset_Handler, %function
Reset_Handler:
    /* The ARM7TDMI enters Supervisor mode on reset. Disable IRQ/FIQ. */
    mrs r0, cpsr
    orr r0, r0, #0xC0          /* set I and F bits */
    msr cpsr_c, r0

    /* Initialise the stack pointer for each required processor mode.
       Mode bits with I(0x80) + F(0x40) set:
         SVC 0xD3, IRQ 0xD2, FIQ 0xD1, Und 0xDB, Abort 0xDF */
    ldr r0, =_stack_top_svc
    mov sp, r0

    mov r0, #0xD2              /* IRQ mode */
    msr cpsr_c, r0
    ldr r0, =_stack_top_irq
    mov sp, r0

    mov r0, #0xD1              /* FIQ mode */
    msr cpsr_c, r0
    ldr r0, =_stack_top_fiq
    mov sp, r0

    mov r0, #0xDB              /* Undefined mode */
    msr cpsr_c, r0
    ldr r0, =_stack_top_und
    mov sp, r0

    mov r0, #0xDF              /* Abort mode */
    msr cpsr_c, r0
    ldr r0, =_stack_top_abt
    mov sp, r0

    mov r0, #0xD3              /* return to Supervisor mode */
    msr cpsr_c, r0

    /* Copy initialised data (.data) from its flash load address to RAM. */
    ldr r0, =_sidata
    ldr r1, =_sdata
    ldr r2, =_edata
    add r2, r1, r2             /* r2 = end of .data in RAM */
    cmp r1, r2
    bge 2f                     /* nothing to copy */
1:
    ldr r3, [r0], #4
    str r3, [r1], #4
    cmp r1, r2
    blt 1b
2:

    /* Clear zero-initialised data (.bss) in RAM. */
    ldr r0, =_sbss
    ldr r1, =_ebss
    mov r2, #0
    cmp r0, r1
    bge bss_done
bss_loop:
    str r2, [r0], #4
    cmp r0, r1
    blt bss_loop
bss_done:

    /* Branch to the C entry point. If main() ever returns, hang. */
    bl main
hang:
    b hang

.size Reset_Handler, .-Reset_Handler
