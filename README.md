# TTCT

TTCT: Unified Trajectory-level Text Constraints Translator for Safe RL

## Installation

To install the required dependencies, run the following command:

`pip install -r requirements.txt`

## Train TTCT

train TTCT, run: `python train.py`

## Train Policy

After you train your own TTCT model, you can use TTCT to train a safe agent that constrained by trajectory-level text constrains.

Here are some commands of different training modes using ppo_lag algorithm:
1. train with the full TTCT: `python ppo_lag.py --use-predict-cost --use-credit-assignment --lagrangian-multiplier-init=0.1 --TL-loadpath=/your/TTCT/model/path`
2. train with the TTCT that removes cost assignment component: `python ppo_lag.py --use-predict-cost  --lagrangian-multiplier-init=0.1 --TL-loadpath=/your/TTCT/model/path`
3. train with stand ground-truth cost funtion: `python ppo_lag.py  --lagrangian-multiplier-init=0.1 --TL-loadpath=/your/TTCT/model/path`
