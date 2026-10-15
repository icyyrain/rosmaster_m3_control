"""Environment RPC shared with the WSL HIL-SERL learner.

codec.py, rpc_server.py and remote_env.py are unmodified copies of
isaaclab_hil_serl/protocol in C:\\Projects\\isaac-hil-serl (last changed in
1eac8c566, copied at 16cfcbb42). The learner in WSL imports the originals, so
the robot environment served from Windows speaks exactly what it expects.
Both sides check PROTOCOL_VERSION on connect; change the copies only together
with the originals.
"""

from .remote_env import RemoteEnv
from .rpc_server import PROTOCOL_VERSION, EnvironmentRpcServer

__all__ = ["EnvironmentRpcServer", "PROTOCOL_VERSION", "RemoteEnv"]
