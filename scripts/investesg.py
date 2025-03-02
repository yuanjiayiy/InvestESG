if __name__ == '__main__':
    from jax_pbt.controller.ippo_controller import IPPOController
    from jax_pbt.config import get_base_parser, generate_parameters
    import numpy as np
    import wandb
    parser = get_base_parser()
    args = parser.parse_args()
    print(args)
    config = generate_parameters(domain=args.env_config_name, debug=args.debug, wandb_project="InvestESG", config_from_arg=vars(args))

    print(config)

    from jax_pbt.env.investesg.investesg_env import InvestESGConst, InvestESGEnv
    env_fn = InvestESGEnv(config=config)
    from jax_pbt.model.models import MLPConfig
    model_config = {'obs': MLPConfig(hidden_layers=2, hidden_size=256)}
    
    def get_epoch_env_const(env_fn: InvestESGEnv, current_env_step: int, **kwargs) -> InvestESGConst:
        return InvestESGConst(
            max_steps=env_fn._env.max_steps,
            shaped_reward_factor=0.0 if env_fn.reward_shaping_steps is None else 1.0 - current_env_step / env_fn.reward_shaping_steps
        )

    controller = IPPOController([args], env_fn=env_fn, model_config_lst=[model_config])
    import jax
    import jax.numpy as jnp
    rng = jax.random.key(config.seed)
    
    if args.eval_points > 0:
        eval_at_steps = list((np.arange(args.eval_points + 1) * args.total_env_steps / args.eval_points).astype(int))
    else:
        eval_at_steps = list(np.arange(args.total_env_steps)[::args.eval_step_interval]) + [args.total_env_steps]
    agent_roles_lst = []
    for i in range(env_fn.num_agents):
        agent_roles = []
        for j in range(args.num_envs):
            agent_roles.append((j, i))
        agent_roles_lst.append(agent_roles)
    from jax_pbt.utils import RoleIndex
    agent_roles_lst = [
        RoleIndex(jnp.array(x, dtype=int)[:, 0], jnp.array(x, dtype=int)[:, 1]) for x in agent_roles_lst
    ]
    
    runner_state = controller.run(rng, agent_roles_lst, get_epoch_env_const=get_epoch_env_const, eval_at_steps=eval_at_steps)
    import flax.training.checkpoints as checkpoints
    rng, train_state_lst, agent_state_lst, env_state, all_obs = runner_state
    ckpt = {
        'rng': jax.random.key_data(rng),
        'train_state': train_state_lst,
        'agent_state': agent_state_lst,
        'env_state': env_state,
        'all_obs': all_obs,
    }
    import shutil
    import os
    if os.path.exists(f"./results/fcp_sp/{args.run_id}"):
        shutil.rmtree(f"./results/fcp_sp/{args.run_id}")

    checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(f"./results/fcp_sp/{args.run_id}"), target=ckpt, step=1, keep=1)
    import pickle
    with open(f"./results/fcp_sp/{args.run_id}/args.pkl", "wb") as f:
        pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)
