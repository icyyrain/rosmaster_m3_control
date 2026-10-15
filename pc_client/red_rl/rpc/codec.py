# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Length-prefixed JSON codec with lossless NumPy array support."""

from __future__ import annotations

import base64
import gymnasium as gym
import json
import numpy as np
import socket
import struct
from typing import Any

_FRAME_HEADER = struct.Struct("!I")
_MAX_FRAME_BYTES = 128 * 1024 * 1024
_TYPE_KEY = "__hil_serl_type__"


def send_message(connection: socket.socket, message: Any) -> None:
    """Encode and send one framed message."""
    payload = json.dumps(_to_wire(message), allow_nan=False, separators=(",", ":")).encode("utf-8")
    assert len(payload) <= _MAX_FRAME_BYTES, f"RPC frame is too large: {len(payload)} bytes"
    connection.sendall(_FRAME_HEADER.pack(len(payload)) + payload)


def receive_message(connection: socket.socket) -> Any:
    """Receive and decode one framed message."""
    header = _receive_exact(connection, _FRAME_HEADER.size)
    (payload_size,) = _FRAME_HEADER.unpack(header)
    assert payload_size <= _MAX_FRAME_BYTES, f"RPC frame is too large: {payload_size} bytes"
    payload = _receive_exact(connection, payload_size)
    return _from_wire(json.loads(payload.decode("utf-8")))


def space_to_spec(space: gym.Space) -> dict[str, Any]:
    """Convert supported Gymnasium spaces to a wire-safe specification."""
    if isinstance(space, gym.spaces.Box):
        return {
            "type": "box",
            "low": space.low,
            "high": space.high,
            "shape": list(space.shape),
            "dtype": np.dtype(space.dtype).str,
        }
    if isinstance(space, gym.spaces.Dict):
        return {
            "type": "dict",
            "spaces": {key: space_to_spec(value) for key, value in space.spaces.items()},
        }
    if isinstance(space, gym.spaces.Discrete):
        return {"type": "discrete", "n": int(space.n), "start": int(space.start)}
    raise TypeError(f"Unsupported Gymnasium space: {type(space).__name__}")


def space_from_spec(spec: dict[str, Any]) -> gym.Space:
    """Reconstruct a Gymnasium space from :func:`space_to_spec`."""
    space_type = spec["type"]
    if space_type == "box":
        dtype = np.dtype(spec["dtype"])
        return gym.spaces.Box(
            low=np.asarray(spec["low"], dtype=dtype),
            high=np.asarray(spec["high"], dtype=dtype),
            shape=tuple(spec["shape"]),
            dtype=dtype,
        )
    if space_type == "dict":
        return gym.spaces.Dict({key: space_from_spec(value) for key, value in spec["spaces"].items()})
    if space_type == "discrete":
        return gym.spaces.Discrete(n=spec["n"], start=spec["start"])
    raise ValueError(f"Unsupported Gymnasium space specification: {space_type!r}")


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError("RPC peer closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def _to_wire(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            _TYPE_KEY: "ndarray",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {_TYPE_KEY: "tuple", "items": [_to_wire(item) for item in value]}
    if isinstance(value, list):
        return [_to_wire(item) for item in value]
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value), "RPC dictionaries require string keys"
        return {key: _to_wire(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported RPC value: {type(value).__name__}")


def _from_wire(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_wire(item) for item in value]
    if not isinstance(value, dict):
        return value
    value_type = value.get(_TYPE_KEY)
    if value_type == "ndarray":
        dtype = np.dtype(value["dtype"])
        shape = tuple(value["shape"])
        raw = base64.b64decode(value["data"], validate=True)
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        assert (
            len(raw) == expected_bytes
        ), f"Invalid ndarray payload: expected {expected_bytes} bytes, received {len(raw)}"
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if value_type == "tuple":
        return tuple(_from_wire(item) for item in value["items"])
    return {key: _from_wire(item) for key, item in value.items()}
