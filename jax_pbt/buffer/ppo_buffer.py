from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import struct

from .base_buffer import Experience


class BaseAgentState(ABC, struct.PyTreeNode):
    @abstractmethod
    def get_batch_shape(self) -> tuple[int]:
        ...

class PPOAgentState(struct.PyTreeNode):
    actor_rnn_state: jax.Array
    critic_rnn_state: jax.Array
    def get_batch_shape(self) -> tuple[int]:
        return self.actor_rnn_state.shape[:-1]

class PPOExperience(Experience):
    agent_state: PPOAgentState
    def compute_advantage(self, last_val: jax.Array, gamma=0.99, gae_lam=0.95) -> jax.Array:
        # print('in compute advantage', last_val.shape)
        def backward_iteration(gae_and_next_val: tuple[jax.Array, jax.Array], transition: PPOExperience):
            gae, next_val = gae_and_next_val # (batch_size,) (batch_size,)
            done, value, reward = (
                transition.done, # (batch_size,)
                transition.val, # (batch_size,)
                transition.reward # (batch_size,)
            ) 
            delta = reward + gamma * next_val * (1 - done) - value # (batch_size,)
            gae = delta + gamma * gae_lam * (1 - done) * gae # (batch_size,)
            return (gae, value), gae
        _, advantage = jax.lax.scan(
            backward_iteration,
            (jnp.zeros_like(last_val), last_val),
            self,
            unroll=16,
            reverse=True
        )
        return advantage # (chunk_length, batch_size)
