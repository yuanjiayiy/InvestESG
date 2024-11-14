for seed in 1
do
    time CUDA_VISIBLE_DEVICES=0 python scripts/investesg.py --total_env_steps 300_000 --seed ${seed} --run_id "investesg_${seed}" --ppo_epochs 4 --episode_length 100 --num_minibatches 20 --num_env 10 --env_config_name exp_default
done