from typing import Sequence, Literal

import jax
import jax.numpy as jnp
import flax.linen as nn


class MultiLayerGRU(nn.Module):
    rnn_layers: int
    hidden_size: int
    use_layer_norm: bool

    def setup(self):
        self.layer_norms = [
            nn.LayerNorm() if self.use_layer_norm else lambda x: x
            for i in range(self.rnn_layers)
        ]
        self.gru_layers = [nn.GRUCell(features=self.hidden_size) for i in range(self.rnn_layers)]
    
    def __call__(self, carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Input:
            carry: [*batch_shape, rnn_layers * hidden_size]
            x: [*batch_shape, input_dim]
        Output: 
            carry: [*batch_shape, rnn_layers * hidden_size]
            x: [*batch_shape, hidden_size]
        """
        new_carry = []
        for i, (gru_layer, layer_norm) in enumerate(zip(self.gru_layers, self.layer_norms)):
            x = layer_norm(x)
            new_carry_layer, x = gru_layer(carry[..., i*self.hidden_size:(i+1)*self.hidden_size], x)
            new_carry.append(new_carry_layer)
        new_carry = jnp.concatenate(new_carry, -1)
        return new_carry, x