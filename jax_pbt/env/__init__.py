from typing import Sequence

import jax.numpy as jnp

# Importing for external use; within the 'env' module, continue using BaseXXX
from .base_env import BaseEnvConst as EnvConst, BaseEnvState as EnvState, BaseEnv as Env
from .spaces import Observation, Action, ObservationSpace, ActionSpace

def get_example_observation(batch_shape: Sequence[int], obs_space: ObservationSpace) -> ObservationSpace:
    return {
        k: jnp.zeros((*batch_shape, *obs_shape))
        for k, obs_shape in obs_space.items()
    }
