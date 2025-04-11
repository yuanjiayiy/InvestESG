import os
import glob
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import matplotlib.pyplot as plt

from jax_pbt.controller.ippo_controller import IPPOController
from jax_pbt.env.investesg.investesg_env import InvestESGEnv
from jax_pbt.model.models import MLPConfig
from jax_pbt.config import get_base_parser, generate_parameters
from jax_pbt.buffer.ppo_buffer import PPOAgentState
from jax_pbt.utils import RoleIndex, rng_batch_split, select_env_agent


######################################
# 1) Forced actions
######################################
def always_defect_action():
    """3D action to invest fully in resilience: (0,0,0.1)."""
    return jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

def always_cooperate_action():
    """3D action to invest fully in mitigation: (0.1, 0, 0)."""
    return jnp.array([0.005, 0.0, 0.0], dtype=jnp.float32)

######################################
# 2) Extract final capitals
######################################
def extract_company_capitals_from_obs(all_obs, num_companies=5):
    """
    'all_obs' is a dict: {'obs' -> jnp.array([num_envs, num_agent, obs_dim])}.
    We pick the first agent's array (identical for all), then parse out
    each company's capital. 
    Each company c has a 7-feature block -> this is wrong
    obs_dim = num_company * 4 + num_investor * (num_company + 1) + 3
    [capital=offset c*4].
    """
    first_agent_key = next(iter(all_obs.keys()))
    obs_mat = all_obs[first_agent_key]  # shape [num_envs, num_agent, obs_dim]
    
    # all agent observation should be the same, so take the first one
    caps_list = []
    for c in range(num_companies):
        idx = c * 4
        caps_list.append(obs_mat[:, 0, idx])
    return jnp.stack(caps_list, axis=1)  # shape [num_envs, num_agent, num_companies]

######################################
# 3) Overwrite checkpoint parameters only
######################################
def update_resume_runner_state_from_checkpoint(ckpt_dict, resume_runner_state):
    """
    Overwrite just the actor/critic params in 'resume_runner_state'
    from the loaded 'ckpt_dict'. Ignores optimizer states.
    """
    rng, resume_train_state, agent_state_lst, env_state, all_obs = resume_runner_state
    new_train_state_lst = []
    for i, (actor_resume, critic_resume) in enumerate(resume_train_state):
        ckpt_agent = ckpt_dict['train_state'][str(i)]
        ckpt_actor_params = ckpt_agent['0']['params']
        ckpt_critic_params = ckpt_agent['1']['params']

        actor_updated = actor_resume.replace(params=ckpt_actor_params)
        critic_updated = critic_resume.replace(params=ckpt_critic_params)
        new_train_state_lst.append((actor_updated, critic_updated))

    return (rng, new_train_state_lst, agent_state_lst, env_state, all_obs)

######################################
# 4) Roll out scenario
######################################
def run_scenario(
    controller,
    runner_state,
    num_steps: int,
    forced_indices: list[int],
    forced_action: str = "defect"
):
    """
    forced_indices: which company indices are forced
    forced_action: "defect" or "cooperate"
    All other companies & all investors => loaded policy
    Returns final capitals: [num_envs, 5].
    """
    def pick_forced_action():
        if forced_action.lower() == "defect":
            return always_defect_action()
        else:
            return always_cooperate_action()

    env_const = controller.env_fn.get_default_const()
    rng, train_state_lst, agent_state_lst, env_state, obs = runner_state

    num_companies = 5

    for t in range(num_steps):
        all_actions = {}
        for i, (agent_name, agent_fn, train_state, agent_state, agent_roles) in enumerate(
            zip(
                controller.env_fn.agent_lst,
                controller.agent_fn_lst,
                train_state_lst,
                agent_state_lst,
                controller.role_index_lst
            )
        ):
            if not agent_name.startswith("company_"):
                # Investor => normal loaded policy
                obs_i = select_env_agent(obs, agent_roles)
                rng, rng_action = rng_batch_split(rng, len(agent_roles))
                next_agent_state, action, log_p, val = jax.vmap(
                    agent_fn.rollout_step, in_axes=(0, None, 0, 0)
                )(rng_action, train_state, agent_state, obs_i)
                agent_state_lst[i] = next_agent_state
                all_actions[agent_name] = action
                continue

            company_idx = int(agent_name.split("_")[1])
            if company_idx in forced_indices:
                # forced
                forced_a = pick_forced_action()
                a = jnp.vstack([forced_a for _ in range(controller.num_envs)])
                all_actions[agent_name] = a
            else:
                # loaded
                obs_i = select_env_agent(obs, agent_roles)
                rng, rng_action = rng_batch_split(rng, len(agent_roles))
                next_agent_state, action, log_p, val = jax.vmap(
                    agent_fn.rollout_step, in_axes=(0, None, 0, 0)
                )(rng_action, train_state, agent_state, obs_i)
                agent_state_lst[i] = next_agent_state

                action_space = controller.env_fn.get_action_space(agent_name)
                clipped = (action + 1.)/2. * action_space.high
                clipped = jnp.clip(clipped, action_space.low, action_space.high)
                all_actions[agent_name] = clipped

        rng, rng_env_step = rng_batch_split(rng, controller.num_envs)
        env_state, obs, rew, done, info = jax.vmap(
            controller.env_fn.step, in_axes=(0, None, 0, 0)
        )(rng_env_step, env_const, env_state, all_actions)

    final_company_capitals = extract_company_capitals_from_obs(obs, num_companies)
    new_runner_state = (rng, train_state_lst, agent_state_lst, env_state, obs)
    return new_runner_state, final_company_capitals

######################################
# 5) Main: python scripts/evaluation.py --run_id "investesg_42" --episode_length 100 --num_env 64 --env_config_name exp_default_1 --debug True
# python scripts/evaluation.py --run_id "investesg_42_esg_pref_10__score_not_observ_192m" --episode_length 100 --num_env 64 --env_config_name exp_default_1 --debug True --save_directory /network/scratch/a/ayoub.echchahed/InvestESG/checkpoints
######################################
def main():
    parser = get_base_parser()
    args = parser.parse_args()

    # always 5 companies
    config = generate_parameters(
        domain=args.env_config_name,
        debug=args.debug,
        wandb_project="InvestESG",
        config_from_arg=vars(args)
    )
    config.update({"num_companies": 5}, allow_val_change=True)

    env_fn = InvestESGEnv(config=config)
    model_config = {'obs': MLPConfig(hidden_layers=2, hidden_size=256)}
    controller = IPPOController([args], env_fn=env_fn, model_config_lst=[model_config])

    # Build role_index
    rng = jax.random.PRNGKey(config.seed)
    agent_roles_lst = []
    for ag_idx in range(env_fn.num_agents):
        arr = [(env_idx, ag_idx) for env_idx in range(args.num_envs)]
        agent_roles_lst.append(arr)

    role_index_lst = [
        RoleIndex(
            jnp.array(r, dtype=int)[:,0],
            jnp.array(r, dtype=int)[:,1],
        )
        for r in agent_roles_lst
    ]
    controller.role_index_lst = role_index_lst

    # init runner_state
    init_runner_state = controller.init_runner_state(rng, env_fn.get_default_const(), role_index_lst)

    # load checkpoint
    save_dir = os.path.join(args.save_directory, args.run_id)
    ckpt_files = glob.glob(os.path.join(save_dir, "checkpoint_*"))
    if ckpt_files:
        init_runner_state = checkpoints.restore_checkpoint(ckpt_dir=save_dir, target=init_runner_state)
        ckpt_dict = checkpoints.restore_checkpoint(ckpt_dir=save_dir, target=None)
        init_runner_state = update_resume_runner_state_from_checkpoint(ckpt_dict, init_runner_state)
        print("Loaded checkpoint for 5 companies.")
    else:
        print("No checkpoint found; using random init for 5 companies.")

    ###########################################
    # forced_counts = [5,4,3,2,1]
    # For each n, we run scenario with forced defectors => final capital
    #               we run scenario with forced cooperators => final capital
    # Then we plot 2 lines: defectors (red), cooperators (blue),
    # with x-axis from 5 down to 1
    ###########################################
    forced_counts = [5,4,3,2,1]
    trained_companies = [0,1,2,3,4]
    defector_payoffs = []
    cooperator_payoffs = []

    num_steps = 100

    for n in forced_counts:
        # reset
        runner_state = init_runner_state
        # pick which companies to force
        all_companies = list(range(5))
        forced_indices = all_companies[-n:]  # if n=5 => [0,1,2,3,4], n=1 => [4], etc.

        # forced defect
        _, final_caps_def = run_scenario(
            controller=controller,
            runner_state=runner_state,
            num_steps=num_steps,
            forced_indices=forced_indices,
            forced_action="defect"
        )
        # final_caps_def: num_env * num_company
        def_caps = final_caps_def[:, forced_indices]
        # def_caps shape (num_env, num_forced_indices, observation_space)
        # doesn't make sense to take average
        defector_payoffs.append(float(def_caps.mean()))

        # forced coop
        _, final_caps_coop = run_scenario(
            controller=controller,
            runner_state=runner_state,
            num_steps=num_steps,
            forced_indices=forced_indices,
            forced_action="cooperate"
        )
        coop_caps = final_caps_coop[:, forced_indices]
        cooperator_payoffs.append(float(coop_caps.mean()))

        print(f"[n={n}] Defectors => {defector_payoffs[-1]:.3f}, Cooperators => {cooperator_payoffs[-1]:.3f}")

    #  Plot
    plt.figure(figsize=(8,5))
    # Use explicit color='red' or 'blue'
    plt.plot(trained_companies, defector_payoffs, marker='o', color='red', label='Defector payoff')
    plt.plot(trained_companies, cooperator_payoffs, marker='s', color='blue', label='Cooperator payoff')

    plt.gca()
    plt.xlabel("Number of Trained Companies")
    plt.ylabel("Individual Ending Capital (avg)")
    plt.title("ESG Score 0")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_plot_.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
