from typing import Sequence

from distrax import Distribution
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax_pbt.utils import get_action_dim
from jaxmarl.environments.spaces import Box, MultiDiscrete

from ..env import Observation, ObservationSpace, ActionSpace
from .models import NNConfig, RNN, FeatureExtractor # TODO: redo cnn config and others


class FeatureModel(nn.Module):
    nn_configs: dict[str, NNConfig]
    obs_space: ObservationSpace
    mlp_hidden_layers: int = 2
    mlp_hidden_size: int = 64
    use_rnn: bool = False
    rnn_hidden_layers: int = 1
    rnn_hidden_size: int = 64
    use_rnn_layer_norm: bool = True
    use_embedding_layer_norm: bool = True
    
    def setup(self):
        self.embedding_model = FeatureExtractor(self.nn_configs)
        if self.use_embedding_layer_norm:
            self.layer_norms = {
                k: nn.LayerNorm() for k in sorted(self.nn_configs)
            }
    
    def __call__(self, x: Observation) -> dict[str, jax.Array]:
        embeddings = self.embedding_model(x)
        if self.use_embedding_layer_norm:
            embeddings = {
                k: layer_norm(embeddings[k]) for k, layer_norm in sorted(self.layer_norms.items())
            }
        return embeddings

class Actor(FeatureModel):
    action_space: ActionSpace = 1
    use_final_layer_norm: bool = False

    def setup(self):
        super().setup()
        if self.use_rnn:
            self.rnn_model = RNN(
                rnn_layers=self.rnn_hidden_layers,
                hidden_size=self.rnn_hidden_size,
                use_layer_norm=self.use_rnn_layer_norm
            )
        if self.use_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm()
        if isinstance(self.action_space, int):
            from .distribution_layer import NaiveDisrete
            self.policy_layer = NaiveDisrete(output_cardinality=self.action_space)
        elif isinstance(self.action_space, Box):
            from .distribution_layer import DiagGaussianGlobalVariance
            # TODO: more different action space
            self.policy_layer = DiagGaussianGlobalVariance(output_shape=get_action_dim(self.action_space), high=self.action_space.high, low=self.action_space.low)
        elif isinstance(self.action_space, MultiDiscrete):
            from .distribution_layer import BernoulliDisrete
            self.policy_layer = BernoulliDisrete(output_cardinality=get_action_dim(self.action_space))

        # TODO: add not implemented
    
    def __call__(self, rnn_states: jax.Array, obs: Observation) -> tuple[jax.Array, Distribution]:
        embeddings = super().__call__(obs)
        feature = jnp.concatenate([v for k, v in sorted(embeddings.items())])
        if self.use_rnn:
            rnn_states, feature = self.rnn_model(rnn_states, feature)
        if self.use_final_layer_norm:
            feature = self.final_layer_norm(feature)
        pi = self.policy_layer(feature)
        return rnn_states, pi

    @nn.nowrap
    def default_rnn_state(self, batch_shape: Sequence[int]) -> jax.Array:
        if self.use_rnn:
            return RNN.default_rnn_states(batch_shape, rnn_layers=self.rnn_hidden_layers, hidden_size=self.rnn_hidden_size)
        else:
            return jnp.zeros((*batch_shape, 1))

class Critic(FeatureModel):
    use_final_layer_norm: bool = False

    def setup(self):
        super().setup()
        if self.use_rnn:
            self.rnn_model = RNN(
                rnn_layers=self.rnn_hidden_layers,
                hidden_size=self.rnn_hidden_size,
                use_layer_norm=self.use_rnn_layer_norm
            )
        if self.use_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm()
        self.value_layer = nn.Dense(1)

    def __call__(self, rnn_states: jax.Array, obs: Observation) -> tuple[jax.Array, Distribution]:
        embeddings = super().__call__(obs)
        feature = jnp.concatenate([v for k, v in sorted(embeddings.items())])
        if self.use_rnn:
            rnn_states, feature = self.rnn_model(rnn_states, feature)
        if self.use_final_layer_norm:
            feature = self.final_layer_norm(feature)
        val = self.value_layer(feature)
        return rnn_states, val.squeeze(-1)

    @nn.nowrap
    def default_rnn_state(self, batch_shape: Sequence[int]) -> jax.Array:
        if self.use_rnn:
            return RNN.default_rnn_states(batch_shape, rnn_layers=self.rnn_hidden_layers, hidden_size=self.rnn_hidden_size)
        else:
            return jnp.zeros((*batch_shape, 1))
