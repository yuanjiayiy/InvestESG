from typing import Sequence, Literal

import jax
import flax.linen as nn


class MLP(nn.Module):
    hidden_size_lst: Sequence[int]
    activation_type: Literal['relu'] = 'relu'

    def setup(self):
        self.fc_layers = [nn.Dense(hidden_size) for hidden_size in self.hidden_size_lst]
        self.activation_fn = [
            nn.relu
        ][
            ['relu'].index(self.activation_type)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        """Input: [*batch_shape, input_dim]
        Output: [*batch_shape, hidden_size_lst[-1]]
        """
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = self.activation_fn(x)
        return x