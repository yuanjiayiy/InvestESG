import argparse
import os
import yaml
import wandb

def parse_tuple(s):
    """Parse a string to a tuple of integers. Example input: '3,3'."""
    try:
        return tuple(map(int, s.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be in the format 'int,int'")

def get_base_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', default=0, type=int, help="Random seed for the entire process.")
    parser.add_argument('--run_id', default='default', type=str, help="Identifier for the experiment run.")
    parser.add_argument('--algo', default='ippo', choices=['ippo'], help="Reinforcement Learning algorithm to use.")
    parser.add_argument('--total_env_steps', default=1_000_000, type=int, help="Total number of environment interaction steps.")
    parser.add_argument('--episode_length', default=200, type=int, help="Number of steps per episode for training; this is not the length of an environment episode.")
    parser.add_argument('--num_envs', default=64, type=int, help="Number of environments to run in parallel for rollouts.")
    parser.add_argument('--ppo_epochs', default=10, type=int, help="Number of PPO iterations after collecting a batch of trajectories.")
    parser.add_argument('--num_minibatches', default=1, type=int, help="Number of minibatches for training.")
    parser.add_argument('--batch_size', default=None, type=int, help="Size of each minibatch; this parameter typically overrides --num_minibatches.")
    parser.add_argument('--chunk_length', default=10, type=int, help="Chunk length for RNN processing.")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor for rewards.")
    parser.add_argument('--gae_lam', default=0.95, type=float, help="Lambda parameter for Generalized Advantage Estimation (GAE).")
    parser.add_argument('--pi_lr', default=1e-4, type=float, help="Learning rate for the policy (actor) network.")
    parser.add_argument('--val_lr', default=1e-4, type=float, help="Learning rate for the value (critic) network.")
    parser.add_argument('--entropy_coef', default=0.01, type=float, help="Entropy bonus coefficient for the policy loss to encourage exploration.")
    parser.add_argument('--ratio_clip', default=0.2, type=float, help="PPO clipping parameter for the probability ratio (pi'/pi).")
    parser.add_argument('--value_clip', default=10.0, type=float, help="Clipping parameter for value function updates.")
    parser.add_argument('--grad_clip_norm', default=10.0, type=float, help="Maximum norm for gradient clipping to prevent exploding gradients.")
    parser.add_argument('--debug', default=False, type=bool, help="Debug setting. Default to false.")

    # Neural network blocks
    parser.add_argument('--mlp_hidden_layer', default=2, type=int, help="Number of hidden layers in the MLP network.")
    parser.add_argument('--mlp_hidden_size', default=64, type=int, help="Number of hidden units in each MLP layer.")

    parser.add_argument('--cnn_feature_lst', type=int, nargs='+', default=[64, 64, 64], help="List of features as integers (e.g., 64 64 64) for CNN.")
    parser.add_argument('--cnn_kernel_lst', type=parse_tuple, nargs='+', default=[(3, 3), (3, 3), (3, 3)], help="List of kernel sizes as tuples (e.g., 3,3 3,3 3,3) for CNN")
    
    parser.add_argument('--use_rnn', default=False, action='store_true', help="Flag to enable RNN usage in the model.")
    parser.add_argument('--rnn_hidden_layers', default=1, type=int, help="Number of layers in the RNN; by default, uses GRU cells.")
    parser.add_argument('--rnn_hidden_size', default=64, type=int, help="Number of hidden units in each RNN layer.")
    parser.add_argument('--no_rnn_layer_norm', default=False, action='store_true', help="Disable LayerNorm before each RNN layer (enabled by default).")   

    parser.add_argument('--no_embedding_layer_norm', default=False, action='store_true', help="Disable LayerNorm after all input sources are mapped to embeddings, before passing them to the aggregation model.")   
    parser.add_argument('--use_final_layer_norm', default=False, action='store_true', help="Add LayerNorm after the last hidden layer")   

    # Log coonfigurations
    parser.add_argument('--eval_points', default=10, type=int, help="Number of evaluation points logged during training.")
    parser.add_argument('--eval_step_interval', default=None, type=int, help="Step interval for logging evaluations; overridden by --eval_points if eval_points > 0.")

    # Environment configurations
    parser.add_argument('--env_config_name', default=None, type=str, help="Environment configuration yaml file name")
    
    return parser

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

def generate_parameters(domain, debug=False, wandb_project=None, config_from_arg={}):
    os.environ["WANDB_MODE"] = "online"

    # config parameters
    config_domain = yaml.safe_load(open("config/domain/" + domain + ".yaml", "r"))

    # Merge configs
    config_with_domain = merge_configs(config_domain, config_from_arg)
    config = dotdict(config_with_domain)

    if debug:
        # Disable weights and biases logging during debugging
        print('Debug selected, disabling wandb')
        wandb.init(project = wandb_project + '-' + domain, config=config, 
            mode='disabled')
    else:
        wandb.init(project = wandb_project + '-' + domain, config=config)

        path_configs = {'model_name': config.domain + "_seed_" + str(config.seed) + "_domain_" + config.domain,
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