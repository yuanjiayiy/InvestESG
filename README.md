
# InvestESG: A Multi-Agent Reinforcement Learning Benchmark for ESG Disclosure and Corporate Climate Investment

![](https://github.com/yuanjiayiy/InvestESG/blob/main/intro.png?raw=true)
Welcome to the official repository for **InvestESG**, a multi-agent reinforcement learning (MARL) benchmark designed to analyze the effects of Environmental, Social, and Governance (ESG) disclosure mandates on corporate climate investments. This repository contains code implementations in both PyTorch and JAX.

## Abstract

InvestESG models an intertemporal social dilemma in which companies face trade-offs between short-term profit losses from climate mitigation efforts and long-term benefits from reducing climate risk. ESG-conscious investors attempt to influence corporate behavior through their investment decisions. Within this framework:

- **Companies** allocate resources across **mitigation, greenwashing, and resilience** efforts, with varying strategies influencing climate outcomes and investor preferences.
- **Investors** prioritize ESG and aim to impact corporate behavior through their investment decisions, attempting to balance financial and ethical incentives.

Our experiments reveal that:
- Without ESG-conscious investors wielding significant capital, corporate mitigation efforts are generally limited under the ESG disclosure mandate.
- When a critical mass of investors prioritizes ESG, corporate cooperation increases, reducing climate risks and enhancing financial stability over the long term.
- Providing more information on global climate risks encourages increased corporate mitigation investment, even without direct investor influence.

These findings align with empirical research, illustrating the potential of MARL to inform policy and address socio-economic challenges through efficient testing of alternative policy and market designs.

## Getting Started

### Prerequisites
- Python 3.8+
- JAX, PyTorch, and other dependencies (listed in `requirements.txt`)

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/InvestESG.git
    cd InvestESG
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Structure

- `env/`: Environment code for InvestESG benchmark.
- `config/`: Configurations for different settings.
- `scripts/`: Scripts for running experiments described in the paper.
- `jax-pbt/`: Jax-based code.
- `README.md`: This README file.

## Usage

### Running the Benchmark

To train models within the InvestESG environment, use the scripts in `setup.sh` to set up and run your experiments. Example configurations for training agents are available in the `config/` directory.

### Example

To start training with default parameters:
```bash
# To run training in PyTorch-based agent
python main.py --env_name 'exp_5*10+3*16.7' --wandb_project multigrid --seed 3

# To run training in JAX-based agent
CUDA_VISIBLE_DEVICES=0 python scripts/investesg.py --total_env_steps 300_000 --seed ${seed} --run_id "investesg_${seed}" --ppo_epochs 4 --episode_length 100 --num_minibatches 20 --num_env 10 --env_config_name exp_default
```

## BibTeX
```
@article{hou2024investesg,
  author={Hou, Xiaoxuan and Yuan, Jiayi and Leibo, Joel Z and Jaques, Natasha},
  title={InvestESG: A Multi-Agent Reinforcement Learning Benchmark for ESG Disclosure and Corporate Climate Investment},
  journal={ArXiv Preprint},
  year={2024}
}
```



