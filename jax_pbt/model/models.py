from typing import Sequence, Literal

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct

from ..env import Observation
from .nn_blocks.cnn import CNN
from .nn_blocks.mlp import MLP
from .nn_blocks.gru import MultiLayerGRU


class RNN(nn.Module):
    rnn_layers: int = 1
    hidden_size: int = 64
    use_layer_norm: bool = False

    def setup(self):
        self.scan_gru = nn.scan(
            MultiLayerGRU, variable_broadcast="params",
            split_rngs={"params": False}, in_axes=0, out_axes=0
        )(
            rnn_layers=self.rnn_layers,
            hidden_size=self.hidden_size,
            use_layer_norm=self.use_layer_norm
        )
    
    def __call__(self, carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Input:
            carry: [*batch_shape, rnn_layers * hidden_size]
            x: [L, *batch_shape, input_dim]
        Output:
            carry: [*batch_shape, rnn_layers * hidden_size]
            x: [L, *batch_shape, hidden_size]
        """
        carry, x = self.scan_gru(carry, x)
        return carry, x
    
    @staticmethod
    def default_rnn_states(batch_shape, rnn_layers, hidden_size):
        return jnp.zeros((*batch_shape, rnn_layers * hidden_size))

from typing import Sequence, Literal
from flax import struct

class NNConfig(struct.PyTreeNode):
    model_arch: Literal['mlp', 'cnn']

class CNNConfig(NNConfig):
    model_arch: Literal['mlp', 'cnn'] = 'cnn'
    feature_lst: Sequence[int] = struct.field(default_factory=lambda: [64, 64, 64])
    kernel_lst: Sequence[Sequence[int]] = struct.field(default_factory=lambda: [(3, 3), (3, 3), (3, 3)])
    activation_type: Literal['relu'] = 'relu'
    mlp_hidden_size: int = 64

class MLPConfig(NNConfig):
    model_arch: Literal['mlp', 'cnn'] = 'mlp'
    hidden_layers: int = 2
    hidden_size: int = 64
    activation_type: Literal['relu'] = 'relu'

class FeatureExtractor(nn.Module):
    nn_configs: dict[str, NNConfig]
    def setup(self):
        feature_extractors = {}
        for k in sorted(self.nn_configs):
            config = self.nn_configs[k]
            if config.model_arch == 'cnn':
                feature_extractors[k] = nn.Sequential([
                    CNN(
                        feature_lst=config.feature_lst,
                        kernel_lst=config.kernel_lst,
                        activation_type=config.activation_type
                    ),
                    lambda x: x.reshape(*x.shape[:-3], -1),
                    MLP(
                        hidden_size_lst=[config.mlp_hidden_size],
                        activation_type=config.activation_type
                    )
                ])
            elif config.model_arch == 'mlp':
                feature_extractors[k] = MLP(
                    hidden_size_lst=[config.hidden_size for _ in range(config.hidden_layers)],
                    activation_type=config.activation_type
                )
            else:
                raise NotImplementedError(f"NN model {config.model_arch} is not supported")
        self.feature_extractors = feature_extractors
    
    def __call__(self, obs: Observation) -> dict[str, jax.Array]:
        res = {}
        for k, f in sorted(self.feature_extractors.items()):
            res[k] = f(obs[k])
        return res
