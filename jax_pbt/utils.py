from typing import Sequence, TypeVar

import jax
import jax.numpy as jnp
import flax.struct as struct
import numpy as np
from jaxmarl.environments import spaces


class RoleIndex(struct.PyTreeNode):
    env_idx: jax.Array
    agent_idx: jax.Array
    def __len__(self) -> int:
        return len(self.env_idx)

def rng_batch_split(rng: jax.Array, batch_size: int | Sequence[int] = 1) -> jax.Array:
    rng, rng_batch = jax.random.split(rng)
    rng_batch = jax.random.split(rng_batch, batch_size)
    return rng, rng_batch

# TODO: check @partial(jax.jit, static_argnames=['chunk_length', 'num_minibatches', 'minibatch_num_chunks'])
def split_into_minibatch(rng: jax.Array, data: struct.PyTreeNode, chunk_length: int, num_minibatches: int | None = 1, minibatch_num_chunks: int | None = None) -> struct.PyTreeNode:
    """Input
        data: [episode_length, batch_size, *data_shape]
    Output
        minibatches: [num_minibatches, chunk_length, minibatch_num_chunks, *data_shape]
    """
    def reshape_data(data: jax.Array):
        # data: [L, B, *]
        episode_length = data.shape[0]
        chunks_per_episode = episode_length // chunk_length
        data = data[:chunks_per_episode*chunk_length].reshape(chunks_per_episode, chunk_length, *data.shape[1:]) # [B', L', B, *]
        data = jnp.swapaxes(data, 1, 2).reshape(-1, chunk_length, *data.shape[3:]) # [BB', L', *]
        p = jax.random.permutation(rng, data.shape[0])
        data = jnp.take(data, p, axis=0)
        n = data.shape[0] // minibatch_num_chunks if minibatch_num_chunks is not None else num_minibatches
        data = data[:data.shape[0]//n*n].reshape(n, -1, *data.shape[1:]) # [N, BB'/N, L, *]
        return jnp.swapaxes(data, 1, 2) # [N, L, BB'/N, *]
    return jax.tree_util.tree_map(reshape_data, data)

def global_norm(params: struct.PyTreeNode) -> jax.Array:
    return jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(params)))

_PyTree_T = TypeVar('_PyTree_T', bound=struct.PyTreeNode)

def select_env_agent(x: _PyTree_T, idx: RoleIndex) -> _PyTree_T:
    return jax.tree_util.tree_map(lambda x: x[idx.env_idx, idx.agent_idx], x)

def set_env_agent_elements(x: _PyTree_T, y: _PyTree_T, idx: RoleIndex) -> _PyTree_T:
    # return jax.tree_util.tree_map(lambda xx, yy: xx.at[idx.env_idx, idx.agent_idx].set(yy), x, y)
    return x.at[idx.env_idx, idx.agent_idx].set(y)

def set_env_agent_elements_tree_map(x: _PyTree_T, y: _PyTree_T, idx: RoleIndex) -> _PyTree_T:
    return jax.tree_util.tree_map(lambda xx, yy: xx.at[idx.env_idx, idx.agent_idx].set(yy), x, y)
    


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(action_space.shape[0])
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
