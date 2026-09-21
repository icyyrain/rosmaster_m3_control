# Experimental firmware build record

- Build date: 2026-09-21
- Target: STM32H743VGTX
- Base: Yahboom `Microros_Samples/Subscriber_uart_servo`
- Compiler: xPack GNU Arm Embedded GCC 12.3.1-1.2.1
- Build tools: xPack Windows Build Tools 4.4.1-3.1
- Result: compile and link succeeded
- ELF size report: text 130100, data 1268, bss 131456 bytes
- Output: `artifacts/Subscriber_uart_servo_feedback_EXPERIMENTAL.hex`
- Output SHA-256:
  `D07E03003896FFBBDE6D1DA6F4BEA10D8C791922A22E6ECA734D630AAB7B3263`

The generated makefiles' machine-specific linker paths were changed to relative
paths for this local build, and the STM32CubeIDE-only
`-fcyclomatic-complexity` flag was removed. These are build-system adjustments,
not target-code changes.

This image has only been compiled. It has not been flashed or hardware-tested.
