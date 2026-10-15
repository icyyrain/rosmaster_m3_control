# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gymnasium client for an environment hosted by :mod:`rpc_server`."""

from __future__ import annotations

import contextlib
import gymnasium as gym
import socket
from typing import Any

from .codec import receive_message, send_message, space_from_spec
from .rpc_server import PROTOCOL_VERSION


class RemoteEnv(gym.Env):
    """Proxy a Windows-hosted environment into the WSL HIL-SERL actor."""

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        super().__init__()
        self._connection = socket.create_connection((host, port), timeout=timeout)
        self._connection.settimeout(timeout)
        self._closed = False

        description = self._rpc("describe")
        assert (
            description["protocol_version"] == PROTOCOL_VERSION
        ), f"Environment RPC protocol mismatch: server={description['protocol_version']}, client={PROTOCOL_VERSION}"
        self.observation_space = space_from_spec(description["observation_space"])
        self.action_space = space_from_spec(description["action_space"])
        self.metadata = description["metadata"]

    def ping(self) -> dict[str, Any]:
        return self._rpc("ping")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        observation, info = self._rpc("reset", seed=seed, options=options)
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._rpc("step", action=action)
        return observation, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if self._closed:
            return
        with contextlib.suppress(ConnectionError, EOFError, OSError):
            self._rpc("close")
        self._connection.close()
        self._closed = True

    def _rpc(self, method: str, **params: Any) -> Any:
        assert not self._closed, "Remote environment is closed"
        send_message(self._connection, {"method": method, "params": params})
        response = receive_message(self._connection)
        assert isinstance(response, dict), "RPC response must be a dictionary"
        if not response.get("ok"):
            error = response.get("error", {})
            raise RuntimeError(f"Remote {error.get('type', 'Error')}: {error.get('message', '')}")
        return response.get("result")
