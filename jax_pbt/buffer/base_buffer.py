import jax
from flax import struct

from ..env import Observation, Action


class Experience(struct.PyTreeNode):
    obs: Observation
    action: Action
    reward: jax.Array
    done: jax.Array
    log_p: jax.Array
    val: jax.Array

