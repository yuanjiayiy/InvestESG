from typing import Sequence, TypeAlias

import jax


Observation: TypeAlias = dict[str, jax.Array]
Action: TypeAlias = dict[str, jax.Array]

ObservationSpace: TypeAlias = dict[str, Sequence[int | float]]
ActionSpace: TypeAlias = int | Sequence[int] | float | Sequence[float]
