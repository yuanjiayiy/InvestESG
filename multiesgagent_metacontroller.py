from copy import deepcopy
from hardcode_agents import HardCodePolicy
from pettingzoo import ParallelEnv
from gym.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import (configure_logger, obs_as_tensor,
                                            safe_mean)
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Any, Dict, List, Optional, Type, Union
from itertools import count
import math
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import wandb
import wandb.wandb_run
import gym
import time
from utils import DictList, plot_single_frame, make_video, extract_mode_from_path
import concurrent.futures

LEARNING_MODES = ["PPO"]

# Multithreading version of the loop
def multithreaded_processing(function, policies, multithreading=True):
    if multithreading:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit each policy processing as a separate thread
            futures = [
                executor.submit(function, polid, policy)
                for polid, policy in enumerate(policies)
            ]
            
            # Optional: Wait for all threads to complete
            for future in concurrent.futures.as_completed(futures):
                # Check for any exceptions in the thread
                try:
                    future.result()  # This will raise any exceptions that occurred in the thread
                except Exception as exc:
                    print(f"Policy {future} generated an exception: {exc}")
                    raise exc
    else:
        for polid, policy in enumerate(policies):
            function(polid, policy)

class DummyGymEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class MultiESGAgent():
    """This is a meta agent that creates and controls several sub agents. If model_others is True,
    Enables sharing of buffer experience data between agents to allow them to learn models of the 
    other agents. """

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        self.num_agents = env.n_agents
        self.num_envs = 1
        self.env = env
        self.company_agents = [agent for agent in env.possible_agents if agent.startswith("company")]
        self.investor_agents = [agent for agent in env.possible_agents if agent.startswith("investor")]
        self.possible_agents = env.possible_agents

        # Observation and action sizes

        self.company_observation_space = env.observation_space(self.company_agents[0])
        self.investor_observation_space = env.observation_space(self.investor_agents[0])

        self.company_action_space = env.action_space(self.company_agents[0]) # Assuming Discrete
        self.investor_action_space = env.action_space(self.investor_agents[0])  # Assuming MultiDiscrete

        self.n_steps = config.update_every
        self.episode = 0
        self.tensorboard_log = None
        self.verbose = config.verbose
        self._logger = None
        self.multithreading = config.multithreading if hasattr(config, "multithreading") else True

        """ LEARNER SETUP """

        policy_kwargs = dict()
        policy_kwargs['net_arch'] = [256, 128]
        policy_kwargs['activation_fn'] = nn.Tanh
        # initially, set up one company use learning, the other two doing random
        env_fn_company = lambda: DummyGymEnv(self.company_observation_space, self.company_action_space)
        dummy_env = DummyVecEnv([env_fn_company] * self.num_envs)
        self.company_policies = [
            PPO(
                policy="MlpPolicy",
                env=dummy_env,
                n_steps=self.n_steps,
                learning_rate=config.lr,
                ent_coef=config.ent_coef,
                policy_kwargs=policy_kwargs,
                verbose=config.verbose,
                device=device,
                seed=config.seed
            )
            for _ in range(len(self.company_agents))
        ]

        env_fn_investor = lambda: DummyGymEnv(self.investor_observation_space, self.investor_action_space)
        dummy_env = DummyVecEnv([env_fn_investor] * self.num_envs)
        self.investor_policies = [
            PPO(
                policy="MlpPolicy",
                env=dummy_env,
                n_steps=self.n_steps,
                learning_rate=config.lr,
                ent_coef=config.ent_coef,
                policy_kwargs=policy_kwargs,
                verbose=config.verbose,
                device=device,
                seed=config.seed
            )
            for _ in range(len(self.investor_agents))
        ]
        self.policies = self.company_policies + self.investor_policies
        for i, policy in enumerate(self.policies):
            policy.mode = config.agent_policies[i]["mode"]
            if policy.mode == "Hardcoded":
                policy.policy = HardCodePolicy(policy.policy, **config.agent_policies[i])

    
        self.max_steps = env.max_steps
        self.total_steps = 0
        self.config = config
        self.device = device
        self.training = training
        self.with_expert = with_expert
        self.loss = None
        self.buffer = []
        self.debug = debug
        self.model_others = False
        self.company_initial_memory = config.company_initial_memory if hasattr(config, "company_initial_memory") else 0
        self.investor_initial_memory = config.investor_initial_memory if hasattr(config, "investor_initial_memory") else 0

    def learn(
        self,
        total_timesteps: int,
        callbacks: Optional[List[MaybeCallback]] = None,
        log_interval: int = 1,
        tb_log_name: str = "IndependentPPO",
        reset_num_timesteps: bool = True,
    ):

        num_timesteps = 0
        all_total_timesteps = []
        if not callbacks:
            callbacks = [None] * self.num_agents
        self._logger = configure_logger(
            self.verbose,
            self.tensorboard_log,
            tb_log_name,
            reset_num_timesteps,
        )
        logdir = None

        # Setup for each policy
        for polid, policy in enumerate(self.policies):
            if policy.mode in LEARNING_MODES:
                policy.start_time = time.time()
                if policy.ep_info_buffer is None or reset_num_timesteps:
                    policy.ep_info_buffer = deque(maxlen=100)
                    policy.ep_success_buffer = deque(maxlen=100)

                if policy.action_noise is not None:
                    policy.action_noise.reset()

                if reset_num_timesteps:
                    policy.num_timesteps = 0
                    policy._episode_num = 0
                    all_total_timesteps.append(total_timesteps)
                    policy._total_timesteps = total_timesteps
                else:
                    # make sure training timestamps are ahead of internal counter
                    all_total_timesteps.append(total_timesteps + policy.num_timesteps)
                    policy._total_timesteps = total_timesteps + policy.num_timesteps

                policy._logger = configure_logger(
                    policy.verbose,
                    logdir,
                    "policy",
                    reset_num_timesteps,
                )

                callbacks[polid] = policy._init_callback(callbacks[polid])


        for polid, policy in enumerate(self.policies):
            if policy.mode in LEARNING_MODES:
                callback = callbacks[polid]
                callback.on_training_start(locals(), globals())
                policy._last_episode_starts = np.ones((self.num_envs,), dtype=bool)

        def train_loop(polid, policy):
            if policy.mode in LEARNING_MODES:
                policy._update_current_progress_remaining(
                    policy.num_timesteps, total_timesteps
                )
                if log_interval is not None and policy.num_timesteps % log_interval == 0:
                    fps = int(policy.num_timesteps / (time.time() - policy.start_time))
                    policy.logger.record("policy_id", polid, exclude="tensorboard")
                    policy.logger.record(
                        "time/iterations", num_timesteps, exclude="tensorboard"
                    )
                    if (
                        len(policy.ep_info_buffer) > 0
                        and len(policy.ep_info_buffer[0]) > 0
                    ):
                        policy.logger.record(
                            "rollout/ep_rew_mean",
                            safe_mean(
                                [ep_info["r"] for ep_info in policy.ep_info_buffer]
                            ),
                        )
                        policy.logger.record(
                            "rollout/ep_len_mean",
                            safe_mean(
                                [ep_info["l"] for ep_info in policy.ep_info_buffer]
                            ),
                        )
                    policy.logger.record("time/fps", fps)
                    policy.logger.record(
                        "time/time_elapsed",
                        int(time.time() - policy.start_time),
                        exclude="tensorboard",
                    )
                    policy.logger.record(
                        "time/total_timesteps",
                        policy.num_timesteps,
                        exclude="tensorboard",
                    )
                    policy.logger.dump(step=policy.num_timesteps)
                if (polid < len(self.company_policies) and policy.num_timesteps > self.company_initial_memory) \
                    or (polid >= len(self.company_policies) and policy.num_timesteps > self.investor_initial_memory):
                    # import pdb; pdb.set_trace()
                    policy.train()

        while num_timesteps < total_timesteps:
            self.collect_rollouts(callbacks)
            if num_timesteps % 10000 == 0:
                img = self.env.render(mode='rgb_array')
                images = wandb.Image(img)
                wandb.log({"figure": images})
                # self.save(self.episode)
            num_timesteps += self.num_envs * self.n_steps
            multithreaded_processing(train_loop, self.policies, multithreading=self.multithreading)
                

        for polid, policy in enumerate(self.policies):
            if policy.mode in LEARNING_MODES:
                callback = callbacks[polid]
                callback.on_training_end()
    

    def collect_rollouts(self, callbacks):
        last_obs, info = self.env.reset()
        last_obs = list(last_obs.values())
        

        all_last_episode_starts = [None] * self.num_agents
        all_obs = [None] * self.num_agents
        all_last_obs = [None] * self.num_agents
        all_rewards = [None] * self.num_agents
        all_dones = [None] * self.num_agents
        all_infos = [None] * self.num_agents
        steps = 0

        for polid, policy in enumerate(self.policies):
            if policy.mode in LEARNING_MODES:
                policy.policy.set_training_mode(False)
                policy.rollout_buffer.reset()
                callbacks[polid].on_rollout_start()
                all_last_episode_starts[polid] = policy._last_episode_starts

        while steps < self.n_steps:
            if steps == 0 or True in all_dones[0]:
                last_obs, info = self.env.reset()
                last_obs = list(last_obs.values())
                for polid, policy in enumerate(self.policies):
                    for envid in range(self.num_envs):
                        assert (
                            last_obs[envid * self.num_agents + polid] is not None
                        ), f"No previous observation was provided for env_{envid}_policy_{polid}"
                        all_last_obs[polid] = np.array(
                            [
                                last_obs[envid * self.num_agents + polid]
                                for envid in range(self.num_envs)
                            ]
                        )
            all_actions = [None] * self.num_agents
            all_values = [None] * self.num_agents
            all_log_probs = [None] * self.num_agents
            all_clipped_actions = [None] * self.num_agents

            
            def evaluate_action(polid, policy):
                with torch.no_grad():
                    obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                    (
                        all_actions[polid],
                        all_values[polid],
                        all_log_probs[polid],
                    ) = policy.policy.forward(obs_tensor)
                    clipped_actions = all_actions[polid].cpu().numpy()
                    action_space = self.company_action_space if policy in self.company_policies else self.investor_action_space
                    if isinstance(action_space, Box) and policy.mode != "Hardcoded":
                        # first scale it from [-1, 1] to [0, action_capping]
                        clipped_actions = (clipped_actions + 1) / 2 * action_space.high
                        # then clip
                        clipped_actions = np.clip(
                            clipped_actions,
                            action_space.low,
                            action_space.high,
                        )
                    elif isinstance(action_space, Discrete):
                        # get integer from numpy array
                        clipped_actions = np.array(
                            [action.item() for action in clipped_actions]
                        )
                    all_clipped_actions[polid] = clipped_actions
            multithreaded_processing(evaluate_action, self.policies, multithreading=self.multithreading)

            # all_clipped_actions = (
            #     np.vstack(all_clipped_actions).transpose().reshape(-1)
            # )  # reshape as (env, action)
            obs, rewards, dones, truncation, infos = self.env.step(all_clipped_actions)
            
            if True in dones:
                self.env.log(self.episode)
                self.episode += 1

            for polid in range(self.num_agents):
                try:
                    all_obs[polid] = np.array(
                        [
                            obs[envid * self.num_agents + polid]
                            for envid in range(self.num_envs)
                        ]
                    )
                except Exception:
                    import pdb; pdb.set_trace()
                all_rewards[polid] = np.array(
                    [
                        rewards[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_dones[polid] = np.array(
                    [
                        dones[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )
                all_infos[polid] = np.array(
                    [
                        infos[envid * self.num_agents + polid]
                        for envid in range(self.num_envs)
                    ]
                )

            for polid, policy in enumerate(self.policies):
                if policy.mode in LEARNING_MODES:
                    policy.num_timesteps += self.num_envs
                    callback = callbacks[polid]
                    callback.update_locals(locals())
                    callback.on_step()

            for polid, policy in enumerate(self.policies):
                if policy.mode in LEARNING_MODES:
                    policy._update_info_buffer(all_infos[polid])

            steps += 1


            # add data to the rollout buffers
            def add_data_to_rollout_buffer(polid, policy):
                if policy.mode in LEARNING_MODES:
                    action_space = self.company_action_space if policy in self.company_policies else self.investor_action_space
                    if isinstance(action_space, Discrete):
                        all_actions[polid] = all_actions[polid].reshape(-1, 1)
                    all_actions[polid] = all_actions[polid].cpu().numpy()
                    policy.rollout_buffer.add(
                        all_last_obs[polid],
                        all_actions[polid],
                        all_rewards[polid],
                        all_last_episode_starts[polid],
                        all_values[polid],
                        all_log_probs[polid],
                    )
            multithreaded_processing(add_data_to_rollout_buffer, self.policies, multithreading=False)
                
                    
            all_last_obs = all_obs
            all_last_episode_starts = all_dones

            def compute_returns_and_advantage(polid, policy):
                with torch.no_grad():
                    if policy.mode in LEARNING_MODES:
                        obs_tensor = obs_as_tensor(all_last_obs[polid], policy.device)
                        _, value, _ = policy.policy.forward(obs_tensor)
                        policy.rollout_buffer.compute_returns_and_advantage(
                            last_values=value, dones=all_dones[polid]
                        )
            multithreaded_processing(compute_returns_and_advantage, self.policies, multithreading=self.multithreading)
                

        for polid, policy in enumerate(self.policies):
            if policy.mode in LEARNING_MODES:
                callback = callbacks[polid]
                callback.on_rollout_end()
                policy._last_episode_starts = all_last_episode_starts[polid]
        return obs

    @classmethod
    def load(
        cls,
        path: str,
        policy: Union[str, Type[ActorCriticPolicy]],
        num_agents: int,
        env: GymEnv,
        n_steps: int,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        **kwargs,
    ) -> "MultiESGAgent":
        model = cls(
            policy=policy,
            num_agents=num_agents,
            env=env,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **kwargs,
        )
        env_fn = lambda: DummyGymEnv(env.observation_space, env.action_space)
        dummy_env = DummyVecEnv([env_fn] * (env.num_envs // num_agents))
        for polid in range(num_agents):
            model.policies[polid] = PPO.load(
                path=path + f"/policy_{polid + 1}/model", env=dummy_env, **kwargs
            )
        return model

    def save(self, episode) -> None:
        for polid, policy in enumerate(self.policies):
            if policy.mode in LEARNING_MODES:
                self.policies[polid].save(os.path.join(wandb.run.dir, f"policy_{polid}/model_{episode}"))
                wandb.save(f"policy_{polid}/model_{episode}.zip")
        