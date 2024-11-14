from typing import Any, Sequence, TypeAlias

import jax
import jax.numpy as jnp
from .investesg import InvestESG as _InvestESG, State as _InvestESGState
from flax.core.frozen_dict import FrozenDict
import wandb

from ..base_env import BaseEnvConst, BaseEnvState, BaseEnv

Observation: TypeAlias = dict[str, jax.Array]
Action: TypeAlias = dict[str, jax.Array]

ObservationSpace: TypeAlias = dict[str, Sequence[float]]
ActionSpace: TypeAlias = float | Sequence[float]

class InvestESGConst(BaseEnvConst):
    shaped_reward_factor: float = 0.0

class InvestESGState(BaseEnvState):
    _state: _InvestESGState

class InvestESGEnv(BaseEnv[InvestESGConst, InvestESGState]):
    def __init__(self, max_steps=100, reward_shaping_steps=None, config={}):
        self._env = _InvestESG(**config)
        self.agent_lst = self._env.agents
        self.max_steps = max_steps
        self.reward_shaping_steps = reward_shaping_steps
        self.state = None
        
    
    @property
    def num_agents(self) -> int:
        return self._env.num_companies + self._env.num_investors
        
    def get_default_const(self) -> InvestESGConst:
        return InvestESGConst(max_steps=self._env.max_steps)

    def get_observation_space(self) -> ObservationSpace:
        obs_space = self._env.observation_space().shape
        obs_space = {'obs': obs_space}
        return obs_space
    
    def get_action_space(self, agent) -> ActionSpace:
        return self._env.action_space(agent)
    
    def reset(
        self,
        rng: jax.Array,
        const: InvestESGConst
    ) -> tuple[InvestESGState, Observation]:
        obs, _state = self._env.reset(rng)
        stacked_obs = jnp.stack([obs[agent] for agent in obs])
        state = InvestESGState(_state=_state, _step=0)
        self.state = state
        return jax.lax.stop_gradient(state), {'obs': jax.lax.stop_gradient(stacked_obs)}
    
    def step(
        self,
        rng: jax.Array,
        const: InvestESGConst,
        state: InvestESGState,
        action: Action
    ) -> tuple[InvestESGState, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        # Return: [state, obs, reward, done, info]
        # For all of obs, reward, done, the shape must follows [num_agents, *]
        obs, states, rewards, dones, infos = self._env.step(key=rng, state=state._state, actions=action)
        stacked_obs = jnp.stack([obs[agent] for agent in obs])
        obs = {'obs': stacked_obs}
        state = InvestESGState(_state=states, _step=state._step+1)
        self.state = state
        reward = jnp.stack([rewards[agent] for agent in rewards])
        done = jnp.stack([dones[agent] for agent in dones])
        info = {'info': infos}
        return jax.lax.stop_gradient(state), jax.lax.stop_gradient(obs), jax.lax.stop_gradient(reward), jax.lax.stop_gradient(done), info
    
    def render(self, state, mode="rgb_array"):
        return self._env.render(state=state, mode="rgb_array")
    
    def log(self, state, episode):
        # import pdb; pdb.set_trace()
        if len(state.history_esg_investment.shape) == 2:
            history_esg_investment = state.history_esg_investment.mean(0)
            history_climate_risk = state.history_climate_risk.mean(0)
            history_climate_event_occurs = state.history_climate_event_occurs.mean(0)
            history_company_mitigation_amount = state.history_company_mitigation_amount.mean(0)
            history_company_greenwash_amount = state.history_company_greenwash_amount.mean(0)
            history_company_resilience_amount = state.history_company_resilience_amount.mean(0)
            history_company_climate_risk = state.history_company_climate_risk.mean(0)
            history_company_capitals = state.history_company_capitals.mean(0)
            history_company_esg_score = state.history_company_esg_score.mean(0)
            history_investment_matrix = state.history_investment_matrix.mean(0)
            history_investor_capitals = state.history_investor_capitals.mean(0)
            history_investor_utility = state.history_investor_utility.mean(0)
            history_market_total_wealth = state.history_market_total_wealth.mean(0)
            history_greenwash_investment = state.history_greenwash_investment.mean(0)
            history_resilience_investment = state.history_resilience_investment.mean(0)
            history_company_rewards = state.history_company_rewards.mean(0)
            history_investor_rewards = state.history_investor_rewards.mean(0)

        else:
            history_esg_investment = state.history_esg_investment
            history_climate_risk = state.history_climate_risk
            history_climate_event_occurs = state.history_climate_event_occurs
            history_company_mitigation_amount = state.history_company_mitigation_amount
            history_company_greenwash_amount = state.history_company_greenwash_amount
            history_company_resilience_amount = state.history_company_resilience_amount
            history_company_climate_risk = state.history_company_climate_risk
            history_company_capitals = state.history_company_capitals
            history_company_esg_score = state.history_company_esg_score
            history_investment_matrix = state.history_investment_matrix
            history_investor_capitals = state.history_investor_capitals
            history_investor_utility = state.history_investor_utility
            history_market_total_wealth = state.history_market_total_wealth
            history_greenwash_investment = state.history_greenwash_investment
            history_resilience_investment = state.history_resilience_investment
            history_company_rewards = state.history_company_rewards
            history_investor_rewards = state.history_investor_rewards

        d = {
            "episode": episode,
            "total climate_event_occurs": sum(history_climate_event_occurs),
            "final climate risk": history_climate_risk[-1],
            "cumulative climate risk": sum(history_climate_risk),
            "final mitigation investment": history_esg_investment[-1],
            "final greenwash investment": history_greenwash_investment[-1],
            "final resilience investment": history_resilience_investment[-1],
            "market total wealth": history_market_total_wealth[-1]
        }

        for i in range(self._env.num_companies):
            d[f'company_{i} total investment'] = sum(history_investment_matrix[:, i])
            d[f'company_{i} episodal reward'] = sum(history_company_rewards[i])
            d[f'company_{i} final capital'] = history_company_capitals[i][-1]
            d[f'company_{i} mitigation amount'] = sum(history_company_mitigation_amount[i])
            
            if self._env.allow_greenwash_investment:
                d[f'company_{i} greenwash amount'] = sum(history_company_greenwash_amount[i])
                
            if self._env.allow_resilience_investment:
                d[f'company_{i} resilience amount'] = sum(history_company_resilience_amount[i])
        
        for i in range(self._env.num_investors):
            d[f'cumulative investor_{i} capital'] = sum(history_investor_capitals[i])
            d[f'investor_{i} episodal reward'] = sum(history_investor_rewards[i])
            # total_investment = sum(state.history_investment_matrix[i, :])
            # d[f'investor_{i} total investment'] = total_investment
            
            # for company, investment in enumerate(state.history_investment_matrix[i, :]):
            #     d[f'investor_{i} investment to company_{company}'] = investment / total_investment if total_investment > 0 else 0

        if episode % 1 == 0:
            img = self.render(state=state, mode='rgb_array')
            images = wandb.Image(img)
            d["figure"] = images
        
        d['episode'] = episode
        wandb.log(d)
