# U3T

U3T: Unified Trajectory-level Text Constraints Translator for Safe RL

## Installation

To install the required dependencies, run the following command:

`pip install -r requirements.txt`

## Train U3T

Get our Hazard-World-Grid dataset from [U3T dataset](https://drive.google.com/drive/folders/13AkiJSKpkxDYBR8tI0F0S72K_GvL2LPu?usp=drive_link), or generate your own dataset and put it in `./dataset/data.pkl`.

Then train U3T, run: `python train.py`

## Train Policy

After you train your own U3T model, you can use U3T to train a safe agent that constrained by trajectory-level text constrains.

Here are some commands of different training modes using ppo_lag algorithm:
1. train with the full U3T: `python ppo_lag.py --use-predict-cost --use-credit-assignment --lagrangian-multiplier-init=0.1 --TL-loadpath=/your/U3T/model/path`
2. train with the U3T that removes cost assignment component: `python ppo_lag.py --use-predict-cost  --lagrangian-multiplier-init=0.1 --TL-loadpath=/your/U3T/model/path`
3. train with stand ground-truth cost funtion: `python ppo_lag.py  --lagrangian-multiplier-init=0.1 --TL-loadpath=/your/U3T/model/path`
