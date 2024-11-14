import gym
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from moviepy.editor import *
import numpy as np
import os
import random
import seaborn as sns
import torch
import wandb
import yaml
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from wrapper_env import WrapperInvestESGEnv


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Collect rewards
        if len(self.locals["infos"]) > 0:
            for info in self.locals["infos"]:
                if "episode" in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    print(f"Episode reward: {info['episode']['r']}")
        return True

    def _on_training_end(self) -> None:
        # Print average reward
        avg_reward = np.mean(self.episode_rewards)
        print(f"Average reward over {len(self.episode_rewards)} episodes: {avg_reward}")

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def merge_configs(update, default):
    if isinstance(update,dict) and isinstance(default,dict):
        for k,v in default.items():
            if k not in update:
                update[k] = v
            else:
                update[k] = merge_configs(update[k],v)
    return update

def make_env(config):
    if 'MultiGrid' in config.domain:
        from envs import gym_multigrid
        from envs.gym_multigrid import multigrid_envs
        env = gym.make(config.domain)
    elif 'InvestESG' == config.domain:
        env = WrapperInvestESGEnv(**config)
    else:
        raise NotImplementedError
    return env

def argmax_2d_index(arr):
    assert len(arr.shape) == 2
    best_2d_index = (arr==torch.max(arr)).nonzero()
    if best_2d_index.shape[0] > 1:  # Handle case with multiple equal maxs
        # Randomly select from multiple equal maxes
        best_2d_index = best_2d_index[random.randrange(best_2d_index.shape[0]),:]
    return best_2d_index.squeeze()

def process_state(state, observation_shape):
    if len(observation_shape) == 3:
        state = torch.tensor(state)
        state = state.transpose(0, 2).transpose(1, 2)
        state = state.float().unsqueeze(0)  # swapped RGB dimension to come first
    return state

def generate_parameters(mode, domain, debug=False, seed=None, with_expert=None, wandb_project=None):
    os.environ["WANDB_MODE"] = "online"

    # config parameters
    config_default = yaml.safe_load(open("config/default.yaml", "r"))
    config_domain = yaml.safe_load(open("config/domain/" + domain + ".yaml", "r"))
    config_mode = yaml.safe_load(open("config/mode/" + mode + ".yaml", "r"))

    # override default random seed 
    if seed:
        config_default['seed'] = seed

    config_default['experiment_name'] = 'MultiGrid'  # TODO: change me

    # Merge configs
    config_with_domain = merge_configs(config_domain, config_default)
    config = dotdict(merge_configs(config_mode, config_with_domain))

    if debug:
        # Disable weights and biases logging during debugging
        print('Debug selected, disabling wandb')
        wandb.init(project = wandb_project + '-' + domain, config=config, 
            mode='disabled')
    else:
        wandb.init(project = wandb_project + '-' + domain, config=config)
    
    path_configs = {'model_name': config.mode + "_seed_" + str(config.seed) + "_domain_" + config.domain + "_version_" + config.version,
                    'load_model_path': config.load_model_start_path + "_seed_" + str(config.seed) + "_domain_" + config.domain + "_version_" + config.version,
                    'wandb_project': wandb_project + '-' + config.domain}
    wandb.config.update(path_configs)

    print("CONFIG")
    print(wandb.config)

    wandb.define_metric("episode/x_axis")
    wandb.define_metric("step/x_axis")

    # set all other train/ metrics to use this step
    wandb.define_metric("episode/*", step_metric="episode/x_axis")
    wandb.define_metric("step/*", step_metric="step/x_axis")

    if not os.path.exists("models/"):
        os.makedirs("models/")

    if not os.path.exists("traj/"):
        os.makedirs("traj/")

    wandb.run.name = config.model_name

    return wandb.config


def plot_single_frame(frame_id, full_env_image, agents_partial_images, actions, rewards, action_dict,
                      fig_dir, expt_name, figsize=(10,10), shared_ylim=False, min_ylim=.0001, predicted_actions=None, all_actions=None):
    # Seaborn palette.
    sns.set()
    color_palette = sns.palettes.color_palette()

    # Hardcoded plot settings
    linewidth = 1.25
    ms_current = 9
    xlabelpad = 9
    ylabelpad = 10

    # Determine variables
    n_agents = len(actions[0])
    max_val = np.max(full_env_image)

    # Create figure
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    total_subplots_horizontal = 2 + n_agents
    total_subplots_vertical = 3
    gs = GridSpec(total_subplots_vertical, total_subplots_horizontal, figure=fig)
     
    # Create sub plots as grid
    full_obs_ax = fig.add_subplot(gs[:2, :2])  # Overall view fig is 2x2 (larger)
    collective_reward_ax = fig.add_subplot(gs[2,:2]) 
    agents_obs_axes = []
    agents_rewards_axes = []
    for i in range(n_agents):
        agents_obs_axes.append(fig.add_subplot(gs[0, i+2]))
        agents_rewards_axes.append(fig.add_subplot(gs[2, i+2]))

    # Determine grid proportions
    full_obs_proportion = 2.0 / total_subplots_horizontal
    agent_proportion = 1.0 / total_subplots_horizontal

    # Plot shared obervation in top left
    full_obs_ax.imshow(full_env_image, interpolation='none')
    full_obs_ax.set_title('Full environment state')
    full_obs_ax.grid(False)

    # Plot individual agents' observations across top right
    for i in range(n_agents):
        agents_obs_axes[i].imshow(agents_partial_images[i], interpolation='none')
        agents_obs_axes[i].set_title('Agent' + str(i) + ' partial obs')
        agents_obs_axes[i].grid(False)

    # Plot collective return bottom left
    collective_return = np.sum(rewards,axis=1)
    cum_collective_return = np.cumsum(collective_return)
    steps = np.arange(len(cum_collective_return))
    collective_reward_ax.plot(steps, cum_collective_return, color=color_palette[0], lw=linewidth)
    if frame_id > 0:
        collective_reward_ax.plot(frame_id, cum_collective_return[frame_id - 1], 'o', ms=ms_current, 
              mfc=color_palette[0], mew=0)
        
        # Write the reward for previous timestep
        s = 'R_t={}: {}'.format(frame_id-1, collective_return[frame_id-1])
        collective_reward_ax.text(0.1, .85, s, fontsize=10,
                                  horizontalalignment='left', verticalalignment='bottom', transform=collective_reward_ax.transAxes)
    collective_reward_ax.set_xlabel('Step', fontsize=10, labelpad=xlabelpad)
    collective_reward_ax.set_ylabel('Collective return', fontsize=10, labelpad=ylabelpad)

    # Write the reward for current timestep
    s = 'R_t={}: {}'.format(frame_id, collective_return[frame_id])
    collective_reward_ax.text(0.1, 0.7, s, fontsize=10, 
                              horizontalalignment='left', verticalalignment='bottom', transform=collective_reward_ax.transAxes)

    # Plot individual agent returns and actions
    for i in range(n_agents):
        # Cumulative return graphs across bottom right
        cum_return = np.cumsum(rewards[:,i])
        agents_rewards_axes[i].plot(steps, cum_return, color=color_palette[0], lw=linewidth)
        if frame_id > 0:
            agents_rewards_axes[i].plot(frame_id, cum_return[frame_id - 1], 'o', ms=ms_current, mfc=color_palette[0], mew=0)
        agents_rewards_axes[i].set_xlabel('Step', fontsize=10, labelpad=xlabelpad)
        agents_rewards_axes[i].set_ylabel('Agent' + str(i) + ' return', fontsize=10, labelpad=ylabelpad)

        # Write the current action and rewards in the space between subplots
        text_horizontal_loc = full_obs_proportion + agent_proportion * i + agent_proportion * 0.2
        if predicted_actions is not None:
            text_vertical_loc = 0.75
        else:
            text_vertical_loc = 0.65
        act_text = 'a^{}_t={}: {}'.format(i, frame_id, action_dict[int(actions[frame_id, i])])  # action
        fig.text(text_horizontal_loc, text_vertical_loc, act_text, fontsize=10)
        r_text = 'R_t={}: {}'.format(frame_id, rewards[frame_id, i])
        fig.text(text_horizontal_loc, text_vertical_loc-0.1, r_text, fontsize=10)
        if frame_id > 0:
            r_prev_text = 'R_t={}: {}'.format(frame_id-1, rewards[frame_id-1, i])
            fig.text(text_horizontal_loc, text_vertical_loc-0.05, r_prev_text, fontsize=10)

    filename = '{}_{:05d}.png'.format(expt_name, frame_id)
    fig_path = os.path.join(fig_dir, filename)
    plt.savefig(fig_path)
    plt.close()

def make_video(video_path, video_name='trajectory_video', frame_rate=10, img_extension='.png'):
    image_files = [os.path.join(video_path, img) for img in os.listdir(video_path) if img.endswith(img_extension)]
    image_files.sort()

    clips = [ImageClip(img).set_duration(1) for img in image_files]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(os.path.join(video_path, video_name + '.mp4'), fps=frame_rate)

    # Another option: os.system("ffmpeg -r 1 -i img%01d.png -vcodec mpeg4 -y movie.mp4")

def print_network_params(net):
    for name, p in net.named_parameters(): 
        print(name, p.data.shape)

def extract_mode_from_path(str):
    for mode in ['dqn', 'bcaux', 'basis', 'psiphi', 'copy']:
        if mode in str:
            return mode 
    assert False, 'No known mode in path ' + str

class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.

    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value

