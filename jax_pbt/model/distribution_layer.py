from typing import Sequence

import distrax
from distrax import Distribution
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class DiagGaussianGlobalVariance(nn.Module):
    output_shape: Sequence[int]
    high: float
    low: float
    def setup(self):
        self.mu = nn.Dense(np.prod(self.output_shape))
        self.log_std = self.param('global_log_std', lambda rng, shape: jnp.zeros(shape), self.output_shape)

    def __call__(self, feature: jax.Array) -> Distribution:
        mu = self.mu(feature)
        # mu = mu.reshape(*mu.shape[:-1], *self.output_shape)
        std = jnp.exp(self.log_std)
        return distrax.MultivariateNormalDiag(mu, std)

class NaiveDisrete(nn.Module):
    output_cardinality: int
    def setup(self):
        self.logits = nn.Dense(self.output_cardinality)
    
    def __call__(self, feature: jax.Array) -> Distribution:
        logits = self.logits(feature)
        return distrax.Categorical(logits=logits)
    
class BernoulliDisrete(nn.Module):
    output_cardinality: int
    def setup(self):
        self.logits = nn.Dense(self.output_cardinality)
    
    def __call__(self, feature: jax.Array) -> Distribution:
        logits = self.logits(feature)
        return distrax.Bernoulli(logits=logits)
