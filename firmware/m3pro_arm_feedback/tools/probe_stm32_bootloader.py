#!/usr/bin/env python3
"""Probe an STM32 ROM UART bootloader without modifying flash.

The script tries a small set of DTR/RTS reset/BOOT0 sequences, sends the
STM32 autobaud byte (0x7f), reads the chip ID when a bootloader answers, and
then asks it to jump back to the application at 0x08000000.

It deliberately implements no erase or write-memory commands.
"""

from __future__ import annotations

import argparse
import time

import serial


ACK = 0x79
NACK = 0x1F


def read_byte(port: serial.Serial, timeout: float = 0.8) -> int | None:
    old_timeout = port.timeout
    port.timeout = timeout
    try:
        data = port.read(1)
    finally:
        port.timeout = old_timeout
    return data[0] if data else None


def command(port: serial.Serial, value: int) -> bool:
    port.reset_input_buffer()
    port.write(bytes((value, value ^ 0xFF)))
    port.flush()
    return read_byte(port) == ACK


def get_chip_id(port: serial.Serial) -> bytes | None:
    if not command(port, 0x02):
        return None
    count_minus_one = read_byte(port)
    if count_minus_one is None:
        return None
    payload = port.read(count_minus_one + 1)
    trailer = read_byte(port)
    if len(payload) != count_minus_one + 1 or trailer != ACK:
        return None
    return payload


def jump_to_application(port: serial.Serial) -> bool:
    if not command(port, 0x21):
        return False
    address = bytes((0x08, 0x00, 0x00, 0x00))
    checksum = address[0] ^ address[1] ^ address[2] ^ address[3]
    port.write(address + bytes((checksum,)))
    port.flush()
    return read_byte(port) == ACK


def apply_sequence(port: serial.Serial, steps: tuple[tuple[bool, bool], ...]) -> None:
    for dtr, rts in steps:
        port.dtr = dtr
        port.rts = rts
        time.sleep(0.15)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/myserial")
    args = parser.parse_args()

    # The two first sequences implement the two logical polarities of the
    # Yahboom/FlyMCU setting "DTR low reset, RTS high enters BootLoader".
    # The final two cover boards on which DTR and RTS are swapped.
    sequences = (
        ("dtr-reset/rts-boot", ((False, False), (False, True), (True, True))),
        ("dtr-reset/rts-boot-inverted", ((True, True), (True, False), (False, False))),
        ("rts-reset/dtr-boot", ((False, False), (True, False), (True, True))),
        ("rts-reset/dtr-boot-inverted", ((True, True), (False, True), (False, False))),
    )

    port = serial.Serial()
    port.port = args.port
    port.baudrate = 115200
    port.bytesize = serial.EIGHTBITS
    port.parity = serial.PARITY_EVEN
    port.stopbits = serial.STOPBITS_ONE
    port.timeout = 0.35
    port.write_timeout = 1.0
    port.dtr = False
    port.rts = False
    port.open()

    try:
        for name, steps in sequences:
            print(f"TRY {name}", flush=True)
            apply_sequence(port, steps)
            port.reset_input_buffer()
            port.write(b"\x7f")
            port.flush()
            response = read_byte(port)
            label = "timeout" if response is None else f"0x{response:02x}"
            print(f"SYNC {name}: {label}", flush=True)
            if response != ACK:
                continue

            chip_id = get_chip_id(port)
            if chip_id is None:
                print("BOOTLOADER_ACK but GET_ID failed", flush=True)
            else:
                print(f"BOOTLOADER_ACK chip_id=0x{chip_id.hex()}", flush=True)

            jumped = jump_to_application(port)
            print(f"JUMP_TO_0x08000000={'ok' if jumped else 'failed'}", flush=True)
            return 0

        print("NO_BOOTLOADER_ACK", flush=True)
        return 2
    finally:
        # Return modem-control outputs to an unasserted state. The normal
        # micro-ROS agent will reopen the port immediately after this probe.
        port.dtr = False
        port.rts = False
        port.close()


if __name__ == "__main__":
    raise SystemExit(main())
