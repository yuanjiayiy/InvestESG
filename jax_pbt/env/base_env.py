from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, TypeVar

from flax import struct
import jax
import jax.numpy as jnp

from .spaces import Observation, Action, ObservationSpace, ActionSpace


class BaseEnvConst(struct.PyTreeNode):
    # Stores the constant variables of the environment
    max_steps: int

class BaseEnvState(struct.PyTreeNode):
    # Represents the internal state of the environment that changes throughout the episode based on actions and random events
    _step: int

EnvConstType = TypeVar('EnvConstType', bound='BaseEnvConst')
EnvStateType = TypeVar('EnvStateType', bound='BaseEnvState')


class BaseEnv(ABC, Generic[EnvConstType, EnvStateType]):
    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def num_agents(self) -> int:
        ...

    @abstractmethod
    def get_default_const(self) -> BaseEnvConst:
        ...

    @abstractmethod
    def get_observation_space(self) -> ObservationSpace:
        ...
    
    @abstractmethod
    def get_action_space(self) -> ActionSpace:
        ...
   
    @abstractmethod
    def reset(
        self,
        rng: jax.Array,
        const: BaseEnvConst
    ) -> tuple[BaseEnvState, Observation]:
        ...
    
    @abstractmethod
    def step(
        self,
        rng: jax.Array,
        const: BaseEnvConst,
        state: BaseEnvState,
        action: Action
    ) -> tuple[BaseEnvState, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        # Return: [state, obs, reward, done, info]
        # For all of obs, reward, done, the shape must follows [num_agents, *] or [*batch_shape, num_agent, *]
        ...
