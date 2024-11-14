import argparse
import random
import torch
import numpy as np
from wandb.integration.sb3 import WandbCallback
# from gym_minigrid.wrappers import ImgObsWrapper

from multiesgagent_metacontroller import MultiESGAgent
import utils
from utils import RewardCallback

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--env_name', type=str, default='MultiGrid-Cluttered-Fixed-15x15',
      help='Name of environment.')
  parser.add_argument(
      '--mode', type=str, default='ppo',
      help="Name of experiment. Can be 'ppo'")
  parser.add_argument(
      '--debug', action=argparse.BooleanOptionalAction,
      help="If used will disable wandb logging.")
  parser.add_argument(
      '--seed', type=int, default=None,
      help="Random seed.")
  parser.add_argument(
      '--keep_training', action=argparse.BooleanOptionalAction,
      help="If used will continue training from previous checkpoint.")
  parser.add_argument(
      '--visualize', action=argparse.BooleanOptionalAction,
      help="If used will disable wandb logging.")
  parser.add_argument(
      '--with_expert', type=bool, default=False,
      help="With expert. Default False.")
  parser.add_argument(
      '--video_dir', type=str, default='videos',
      help="Name of location to store videos.")
  parser.add_argument(
      '--load_checkpoint_from',  type=str, default=None,
      help="Path to find model checkpoints to load")
  parser.add_argument(
        '--wandb_project', type=str, default='',
        help="Name of wandb project. Choose from 'multiagent_copying_ii' for 2 experts or 'multiagent_copying_1_expert_1_novice'. ")

  return parser.parse_args()

def get_metacontroller_class(config, env):
    if hasattr(env, "n_agents") and env.n_agents == 1:
       return SingleAgent
    elif 'InvestESG' in config.domain:
       return MultiESGAgent
    else:
       return MultiAgent

def initialize(mode, env_name, debug, visualize, seed, with_expert, wandb_project):

    config = utils.generate_parameters(
      mode=mode, domain=env_name, debug=(debug or visualize), 
      seed=seed, with_expert=with_expert, wandb_project=wandb_project)
    
    if config.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    print("Use device:", device)

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = utils.make_env(config)

    metacontroller_class = get_metacontroller_class(config, env)

    return device, config, env, metacontroller_class

def main(args):
    device, config, env, metacontroller_class = initialize(
      args.mode, args.env_name, args.debug, args.visualize, args.seed, args.with_expert, args.wandb_project)

    # Ensure if you're logging to wandb, it's to the right wandb
    if not args.debug and not args.visualize:  # Real run that logs to wandb
      if not args.wandb_project:
        print('ERROR: when logging to wandb, must specify a valid wandb project.')
        exit(1)

      current_wandb_projects = ['multigrid']  # Add your wandb project here
      if str(args.wandb_project) not in current_wandb_projects:
          print('ERROR: wandb project not in current projects. '
                'Change the project name or add your new project to the current projects in current_wandb_projects. '
                'Current projects are:', current_wandb_projects)
          exit(1)

    if args.visualize:
      agent = metacontroller_class(config, env, device, with_expert=args.with_expert, training=False)
      agent.load_models(model_path=args.load_checkpoint_from)
      agent.visualize(env, args.mode, args.video_dir)

      print('A video of the trained policies being tested in the environment'
            'has been generated and is located in', config.load_model_path)
      exit(0)
    
    # Train Model
    model = metacontroller_class(config, env, device, with_expert=args.with_expert, debug=args.debug)
    reward_call_backs = [RewardCallback() for _ in range(env.n_agents)]
    model.learn(total_timesteps=config.total_timesteps, callbacks=reward_call_backs)

if __name__ == '__main__':
    main(parse_args())