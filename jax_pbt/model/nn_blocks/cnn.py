from typing import Sequence, Literal

import jax
import flax.linen as nn


class CNN(nn.Module):
    feature_lst: Sequence[int]
    kernel_lst: Sequence[Sequence[int]]
    activation_type: Literal['relu'] = 'relu'
    def setup(self):
        self.cnn_layers = [
            nn.Conv(
                features=num_features,
                kernel_size=kernel_shape,
            )
            for (num_features, kernel_shape) in zip(self.feature_lst, self.kernel_lst)
        ]
        self.activation_fn = [
            nn.relu
        ][
            ['relu'].index(self.activation_type)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        """Input: [*batch_shape, W, H, C]
        Output: [*batch_shape, W', H', feature_lst[-1]]
        """
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
            x = self.activation_fn(x)
        return x
