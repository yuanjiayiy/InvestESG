from argparse import Namespace
from collections import deque
from typing import Any, Callable, Sequence, TypeAlias
from jaxmarl.environments.spaces import Box, MultiDiscrete

import jax
import jax.numpy as jnp
from tqdm import tqdm
import wandb

from ..agent.ppo_agent import PPOAgent, PPOTrainState
from ..buffer.ppo_buffer import PPOAgentState, PPOExperience
from ..env import Env, EnvConst, EnvState, Observation
from ..model.models import NNConfig
from ..utils import RoleIndex, get_action_dim, rng_batch_split, select_env_agent, set_env_agent_elements, set_env_agent_elements_tree_map
from .base_controller import BaseController


RunnerState: TypeAlias = tuple[jax.Array, Sequence[PPOTrainState], Sequence[PPOAgentState], EnvState, Observation]

class IPPOController(BaseController):
    def __init__(self, args_lst: Sequence[Namespace], env_fn: Env, model_config_lst: Sequence[NNConfig], default_args: Namespace = None):
        # load configuration
        args = default_args or args_lst[0]

        self.total_env_steps = args.total_env_steps
        self.episode_length: int = args.episode_length
        self.num_envs: int = args.num_envs
        self.num_agents: int = env_fn.num_agents
        self.ppo_epochs: int = args.ppo_epochs
        self.gamma: float = args.gamma
        self.gae_lam: float = args.gae_lam

        # init env_fn
        self.env_fn: Env = env_fn

        # init agents
        # TODO: different obs_space/act_space for agents with different roles?
        if hasattr(env_fn, "agent_lst"):
            self.agent_fn_lst = []
            for agent in env_fn.agent_lst:
                obs_space, action_space = env_fn.get_observation_space(), env_fn.get_action_space(agent)
                self.agent_fn_lst.extend([PPOAgent.init(agent, ind_args, obs_space, action_space, model_config) for ind_args, model_config in zip(args_lst, model_config_lst)])
        else:
            obs_space, action_space = env_fn.get_observation_space(), env_fn.get_action_space()
            self.agent_fn_lst = [PPOAgent.init(None, ind_args, obs_space, action_space, model_config) for ind_args, model_config in zip(args_lst, model_config_lst)]
        
    def init_runner_state(self, rng: jax.Array, env_const: EnvConst, agent_roles_lst: Sequence[RoleIndex]) -> RunnerState:
        agent_state_lst = [
            agent_fn.agent_state_init(batch_shape=(len(agent_roles),))
            for agent_fn, agent_roles in zip(self.agent_fn_lst, agent_roles_lst)
        ]
        rng, rng_init = jax.random.split(rng)
        train_state_lst = [agent_fn.train_state_init(rng_init) for agent_fn in self.agent_fn_lst] # train_state carry all variables which will only to changed by gradient udpate and remain static in the episode
        rng, rng_reset = rng_batch_split(rng, self.num_envs)
        env_state, obs = jax.vmap(self.env_fn.reset, in_axes=(0, None))(rng_reset, env_const) # TODO: change None
        # import pdb; pdb.set_trace()
        return rng, train_state_lst, agent_state_lst, env_state, obs

    def rollout(self, runner_state: RunnerState, env_const: EnvConst, agent_roles_lst: Sequence[RoleIndex]) -> tuple[RunnerState, PPOExperience, dict]:
        def step(runner_state: RunnerState, unused=None) -> tuple[RunnerState, tuple[PPOExperience, dict]]:
            rng, train_state_lst, agent_state_lst, env_state, all_obs = runner_state

            # TODO allow different actions for different agents in the environment?
            all_action = dict()
            all_clipped_action = dict()
            for agent in self.env_fn.agent_lst:
                action_space = self.env_fn.get_action_space(agent)
                all_action[agent] = jnp.zeros([self.num_envs, get_action_dim(action_space)])
                all_clipped_action[agent] = jnp.zeros([self.num_envs, get_action_dim(action_space)])
            all_log_p, all_val = jnp.zeros([self.num_envs, self.num_agents]), jnp.zeros([self.num_envs, self.num_agents])
            next_agent_state_lst = []
            # TODO: can we do this in parallel?
            for i, (agent_name, agent_fn, train_state, agent_state, agent_roles) in enumerate(zip(self.env_fn.agent_lst, self.agent_fn_lst, train_state_lst, agent_state_lst, agent_roles_lst)):
                obs = select_env_agent(all_obs, agent_roles)
                rng, rng_action = rng_batch_split(rng, len(agent_roles))
                next_agent_state, action, log_p, val = jax.vmap(agent_fn.rollout_step, in_axes=(0, None, 0, 0))(
                    rng_action, train_state, agent_state, obs
                ) # TODO: remove None
                # Clipping
                if isinstance(self.env_fn.get_action_space(agent_name), Box):
                    action_space = self.env_fn.get_action_space(agent_name)
                    # first scale it from [-1, 1] to [0, action_capping]
                    clipped = (action + 1) / 2 * action_space.high
                    clipped = jnp.clip(clipped, action_space.low, action_space.high)
                else:
                    clipped = action
                # print("actionaction", i, agent_name)
                # print('all_action.shape', all_action[agent_name].shape)
                # print('action.shape', action.shape)
                all_action[agent_name] = action
                all_clipped_action[agent_name] = clipped
                
                # print("new")
                # print('all_action.shape', all_action[agent].shape)

                # print('all_log_p.shape', all_log_p.shape)
                # print('log_p.shape', log_p.shape)
                all_log_p = set_env_agent_elements_tree_map(all_log_p, log_p, agent_roles)
                all_val = set_env_agent_elements_tree_map(all_val, val, agent_roles)
                next_agent_state_lst.append(next_agent_state)

            rng, rng_env_step = rng_batch_split(rng, self.num_envs)
            next_env_state, all_next_obs, all_reward, all_done, info = jax.vmap(self.env_fn.step, in_axes=(0, None, 0, 0))(rng_env_step, env_const, env_state, all_clipped_action) # TODO: remove None
            # TODO: add reset
            
            for i, agent_roles in enumerate(agent_roles_lst):
                done = select_env_agent(all_done, agent_roles)
                next_agent_state_lst[i] = self.agent_fn_lst[i].agent_state_reset(next_agent_state_lst[i], done)

            next_runner_state = (rng, train_state_lst, next_agent_state_lst, next_env_state, all_next_obs)
            # print('action.shape', all_action)
            # print('reward.shape', all_reward)
            # print('done.shape', all_done)
            # print('log_p.shape', all_log_p)
            # print('val.shape', all_val) # TODO: remove debug infor
            # jax.debug.print("{}", all_action)

            all_agent_state = PPOAgentState(
                actor_rnn_state=jnp.zeros((self.num_envs, self.num_agents, *agent_state_lst[0].actor_rnn_state.shape[1:])),
                critic_rnn_state=jnp.zeros((self.num_envs, self.num_agents, *agent_state_lst[0].critic_rnn_state.shape[1:]))
            )
            for agent_state, agent_roles in zip(agent_state_lst, agent_roles_lst):
                all_agent_state = set_env_agent_elements_tree_map(all_agent_state, agent_state, agent_roles)

            # TODO: test each buffer for each agent
            transition = PPOExperience(
                obs=all_obs,
                action=all_action,
                reward=all_reward,
                done=all_done,
                log_p=all_log_p,
                val=all_val,
                agent_state=all_agent_state
            )
            return next_runner_state, (transition, info)
        runner_state, (buffer, env_info) = jax.lax.scan(
            step, runner_state, None, self.episode_length
        )
        return runner_state, buffer, env_info
    
    def run_epoch(self, runner_state: RunnerState, env_const: EnvConst, agent_roles_lst: Sequence[RoleIndex]) -> tuple[RunnerState, dict]:
        runner_state, all_buffer, env_info = self.rollout(runner_state, env_const, agent_roles_lst)
        env_stats = {
            'return': all_buffer.reward.sum(0).mean(0),
            'done': all_buffer.done
        }
        
        # rng, train_state, agent_state, env_state, all_last_obs = runner_state
        rng, train_state_lst, agent_state_lst, env_state, all_last_obs = runner_state

        new_train_state_lst = []
        # TODO: parallel?
        for i, (agent_name, agent_fn, train_state, agent_state, agent_roles) in enumerate(zip(self.env_fn.agent_lst,self.agent_fn_lst, train_state_lst, agent_state_lst, agent_roles_lst)):
            # print('stop_training', i, self.[i])
            if self.stop_training[i]:
                continue
            buffer = jax.vmap(select_env_agent, in_axes=(0, None))(all_buffer, agent_roles)

            buffer = buffer.replace(action = all_buffer.action)

            last_obs = select_env_agent(all_last_obs, agent_roles)
            last_val = agent_fn.get_val(train_state, agent_state, last_obs)
            advantage = buffer.compute_advantage(last_val, gamma=self.gamma, gae_lam=self.gae_lam)
            target_val = advantage + buffer.val
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            for ppo_epoch in range(self.ppo_epochs):
                rng, rng_model_update = jax.random.split(rng)
                train_state, optim_log = agent_fn.learn(rng_model_update, train_state, buffer, advantage, target_val)
            new_train_state_lst.append(train_state)
        # import pdb; pdb.set_trace()
        
        runner_state = rng, new_train_state_lst, agent_state_lst, env_state, all_last_obs
        return runner_state, {'env_stats': env_stats, 'optim_log': optim_log}
    
    def run(
        self,
        rng: jax.Array,
        agent_roles_lst: Sequence[RoleIndex],
        get_init_env_const: Callable[[], EnvConst] | None = None,
        get_epoch_env_const: Callable[[Env, 'IPPOController', int, EnvConst], EnvConst] | None = None,
        eval_at_steps: list[int] = None,
        resume_trainer_state_lst: dict[int, PPOTrainState] = None,
    ) -> None:
        print("\n" + "=" * 20 + " start compilation " + "=" * 20 +"\n")
        print("jax.jit compilation of the [ippo_controller.run_epoch] function...")
        run_epoch_jitted = jax.jit(self.run_epoch) # jax.jit(self.run_epoch)
        print("\n" + "=" * 20 + " finish compilation " + "=" * 20 +"\n")

        if get_init_env_const is None:
            get_init_env_const = self.env_fn.get_default_const

        env_const = get_init_env_const()
        # TODO: test static agent_roles_lst by making it self.agent_roles_lst?
        runner_state = self.init_runner_state(rng, env_const, agent_roles_lst)
        rng, train_state_lst, agent_state_lst, env_state, all_obs = runner_state
        self.stop_training = [False for _ in self.agent_fn_lst]
        if resume_trainer_state_lst is not None:
            for i, resume_trainer_state in resume_trainer_state_lst:
                train_state_lst[i] = resume_trainer_state
                self.stop_training[i] = True
        runner_state = rng, train_state_lst, agent_state_lst, env_state, all_obs

        env_step = 0
        if eval_at_steps is None:
            eval_at_steps = []
        eval_at_steps: deque = deque(eval_at_steps)
        episode = 0
        # TODO: eval
        with tqdm(total=self.total_env_steps, desc="Training progress", unit="frames", position=2) as pbar:
            while env_step < self.total_env_steps:
                env_const = get_epoch_env_const(env_fn=self.env_fn, controller=self, current_env_step=env_step, env_const=env_const)
                runner_state, info = run_epoch_jitted(runner_state, env_const, agent_roles_lst)

                env_stats = info['env_stats']
                env_step += self.episode_length * self.num_envs
                pbar.update(self.episode_length * self.num_envs)
                if len(eval_at_steps) > 0 and env_step > eval_at_steps[0]:
                    eval_at_steps.popleft()
                    tqdm.write("=" * 20 + f"episode {episode} env_step: {env_step} / {self.total_env_steps} " + "=" * 20)
                    tqdm.write(f"\nEnv stats: {env_stats}\n")
                
                d = dict()
                for i, ret in enumerate(env_stats['return']):
                    d[f"{i} return"] = ret
                wandb.log(d)

                
                self.env_fn.log(runner_state[-2]._state, episode)
                
                # reset env manullay
                rng, train_state_lst, agent_state_lst, env_state, all_obs = runner_state
                rng, rng_reset = rng_batch_split(rng, self.num_envs)
                env_state, all_obs = jax.vmap(self.env_fn.reset, in_axes=(0, None))(rng_reset, env_const) 
                runner_state = rng, train_state_lst, agent_state_lst, env_state, all_obs
                
                episode += 1

                
        return runner_state # TODO: just temperory usage
