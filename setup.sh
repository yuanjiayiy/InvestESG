#/bin/bash
# ensure conda is installed properly before running setup.sh
pip install -r requirements.txt
pip install -e .

# To run a sample training with PyTorch
# python main.py --env_name 'exp_5*10+3*16.7' --wandb_project multigrid --seed 3

# To run a sample training with JAX
# Please refer to https://github.com/jax-ml/jax for detailed instructions.
# CUDA_VISIBLE_DEVICES=0 python scripts/investesg.py --total_env_steps 300_000 --seed ${seed} --run_id "investesg_${seed}" --ppo_epochs 4 --episode_length 100 --num_minibatches 20 --num_env 10 --env_config_name exp_default