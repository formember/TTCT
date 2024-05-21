# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



from __future__ import annotations
from copy import deepcopy
from tqdm import tqdm
import datetime
from torch.distributions import Categorical
import os
from torch.distributions import Normal
import random
import sys
import time
from collections import deque
from U3T import U3T
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import gym
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
import loralib as lora
import sys
from common.buffer import VectorizedOnPolicyBuffer
from common.lagrange import Lagrange
from common.logger import EpochLogger
from common.model import ActorVCriticTrajectory
from utils.config import single_agent_args, isaac_gym_map
from utils.util import BufferDataset
from utils.async_vector_env import AsyncVectorEnv
import gym_minigrid
import safety_gymnasium
from U3T import U3T


FOCOPS_LAM=1.50
FOCOPS_NU=5.0

default_cfg = {
    'hidden_sizes': [64, 64],
    'gamma': 0.99,
    'target_kl': 0.02,
    'batch_size': 64,
    'learning_iters': 40,
    'max_grad_norm': 40.0,
}

isaac_gym_specific_cfg = {
    'total_steps': 100000000,
    'steps_per_epoch': 32768,
    'hidden_sizes': [256, 128, 128, 64],
    'r_gamma': 0.95,
    'c_gamma': 0.95,
    'threshold_Mini':7.55,
    'threshold_Goal':5.5,
    'cost_value':1.0,
    'batch_size':512,
    'target_kl': 0.016,
    'num_mini_batch': 4,
    'learning_rate':3e-4,
    'use_value_coefficient': True,
    'learning_iters': 8,
    'max_grad_norm': 1.0,
    'use_critic_norm': False,
}

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def lora_model(model,rank):
    rank=rank
    alpha = 16
    layer_names_dict = model.state_dict().keys()
    module_list = []
    for key in layer_names_dict:
        module_list.append('.'.join(key.split('.')[:-1]))
    for submodule_key in module_list:
        if submodule_key.split('.')[-1] in ["query", "value"]:
            module_state_dict = model.get_submodule(submodule_key).state_dict()
            submodule = model.get_submodule(submodule_key)
            lora_layer = lora.Linear(
                submodule.in_features,
                submodule.out_features,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.1
            )
            lora_layer.load_state_dict(module_state_dict,strict=False)
            _set_module(model, submodule_key, lora_layer)
    

def load_from_save(tlmodel,name):
    model_path = name
    with open(model_path, 'rb') as opened_file:
        state_dict = torch.load(opened_file, map_location="cpu")
    tlmodel.load_state_dict(state_dict,strict=True)      

def main(args, cfg_env=None):
    if args.task == "MiniGrid":
        act_dim=1
        obs_dim=147
        obs_emb_dim=64
    else:
        act_dim=2
        obs_dim=60
        obs_emb_dim=256
        
    embed_dim=512
    trajectory_length=200
    context_length=77
    vocab_size=49408
    config = isaac_gym_specific_cfg
    transformer_width=512
    transformer_heads=8
    transformer_layers=12
    BERT_PATH='./bert-base-uncased'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_predict_cost=args.use_predict_cost
    language_model= args.language_model
    if language_model=="TLmodel":
        TL_loadpath=args.TL_loadpath
        if args.is_finetune:
            EncodeModel=U3T(
                    embed_dim=embed_dim,
                    trajectory_length=trajectory_length,
                    context_length=context_length,
                    vocab_size=vocab_size,
                    transformer_width=transformer_width,
                    transformer_heads=transformer_heads,
                    transformer_layers=transformer_layers,
                    act_dim=act_dim,
                    BERT_PATH='bert-base-uncased',
                    device=device,
                    obs_emb_dim=obs_emb_dim,
                    obs_dim=obs_dim,
                    threshold=config['threshold_Mini'] if args.task == "MiniGrid" else config['threshold_Goal'],
                    cost_value=config['cost_value']
                )
            load_from_save(EncodeModel,TL_loadpath)
            EncodeModel=EncodeModel.to(device)
            if args.use_lora:
                lora_model(EncodeModel,args.rank)
                lora.mark_only_lora_as_trainable(EncodeModel)
                EncodeModel=EncodeModel.to(device)
        TLmodel=U3T(
                    embed_dim=embed_dim,
                    trajectory_length=trajectory_length,
                    context_length=context_length,
                    vocab_size=vocab_size,
                    transformer_width=transformer_width,
                    transformer_heads=transformer_heads,
                    transformer_layers=transformer_layers,
                    act_dim=act_dim,
                    BERT_PATH='bert-base-uncased',
                    device=device,
                    obs_emb_dim=obs_emb_dim,
                    obs_dim=obs_dim,
                    threshold=config['threshold_Mini'] if args.task == "MiniGrid" else config['threshold_Goal'],
                )
        load_from_save(TLmodel,TL_loadpath)
        TLmodel=TLmodel.to(device)
    elif language_model=='Bert':
        TLmodel=U3T(
                embed_dim=embed_dim,
                trajectory_length=trajectory_length,
                context_length=context_length,
                vocab_size=vocab_size,
                transformer_width=transformer_width,
                transformer_heads=transformer_heads,
                transformer_layers=transformer_layers,
                act_dim=act_dim,
                BERT_PATH='bert-base-uncased',
                device=device,
                obs_emb_dim=obs_emb_dim,
                obs_dim=obs_dim,
            )
    else:
        raise NotImplementedError
    
    EncodeModel.train()
    
    if args.task == "MiniGrid":
        envB=[lambda: gym.make('MiniGrid-HazardWorld-B-v0') for _ in range(args.num_envs//3)]
        envS=[lambda: gym.make('MiniGrid-HazardWorld-S-v0') for _ in range(args.num_envs//3)]
        envL=[lambda: gym.make('MiniGrid-HazardWorld-L-v0') for _ in range(args.num_envs-(args.num_envs//3)*2)]
        allenv=envB+envS+envL
        # observation_space=gym.spaces.Box(low=-5, high=20, shape=(7,7,3), dtype=np.uint8)
        # action_space=gym.spaces.Discrete(4)
        env = AsyncVectorEnv(allenv)   
    elif args.task == "SafetyRacecarGoal2-v0":
        configB={"env_type":'budgetary','agent_name':'Racecar'}
        configR={"env_type":'relational','agent_name':'Racecar'}
        envB=[lambda: safety_gymnasium.make('SafetyRacecarGoal2-v0',max_episode_steps=199,render_mode='rgb_array',camera_name="human",width=256,height=256,config=configB) for _ in range(args.num_envs//2)]
        envR=[lambda: safety_gymnasium.make('SafetyRacecarGoal2-v0',max_episode_steps=199,render_mode='rgb_array',camera_name="human",width=256,height=256,config=configR) for _ in range(args.num_envs//2)]
        allenv=envB+envR
        env = safety_gymnasium.vector.SafetyAsyncVectorEnv(allenv)
    else:
        raise NotImplementedError
    obs_space = env.single_observation_space
    act_space = env.single_action_space
    obs_space = torch.zeros((embed_dim+embed_dim+obs_dim,))
    if args.task != "MiniGrid":
        config["steps_per_epoch"]=config["steps_per_epoch"]//4
        config["target_kl"]=0.08
    # set training steps
    steps_per_epoch = config.get("steps_per_epoch", args.steps_per_epoch)
    total_steps = config.get("total_steps", args.total_steps)
    local_steps_per_epoch = steps_per_epoch // args.num_envs
    epochs = total_steps // steps_per_epoch
    # create the actor-critic module
    policy = ActorVCriticTrajectory(
        obs_dim=obs_dim,
        trajectory_dim=embed_dim,
        text_dim=embed_dim,
        act_dim=act_space.n if args.task == "MiniGrid" else act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
        is_discrete=(args.task == "MiniGrid")
    ).to(device)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=config['learning_rate'])
    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs,
        verbose=False,
    )
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=config['learning_rate']
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=config['learning_rate']
    )

    if args.is_finetune:
        Encode_optimizer = torch.optim.Adam(EncodeModel.parameters(), lr=1e-5)
        Encode_scheduler = LinearLR(
            Encode_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=epochs,
            verbose=False,
        )
    # create the vectorized on-policy buffer
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=args.num_envs,
        r_gamma=config["r_gamma"],
        c_gamma=config["c_gamma"],
    )
    # setup lagrangian multiplier
    lagrange = Lagrange(
        cost_limit=args.cost_limit,
        lagrangian_multiplier_init=args.lagrangian_multiplier_init,
        lagrangian_multiplier_lr=args.lagrangian_multiplier_lr,
    )

    # set up the logger
    dict_args = vars(args)
    dict_args.update(config)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    rew_deque = deque(maxlen=50)
    train_cost_deque = deque(maxlen=50)
    true_cost_deque = deque(maxlen=50)
    len_deque = deque(maxlen=50)
    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    logger.save_config(dict_args)
    logger.setup_torch_saver(policy.actor)
    logger.setup_torch_saver1(EncodeModel)
    logger.log("Start with training.")
    actlist=[[] for i in range(args.num_envs)]
    obslist=[[] for i in range(args.num_envs)]
    truecostlist=[[] for i in range(args.num_envs)]
    predictcostlist=[[] for i in range(args.num_envs)]
    obs, info = env.reset()
    if act_dim==1:
        act=0
    elif act_dim==2:
        act=(0,0)
    else:
        raise ValueError("act_dim should be 1 or 2")
    mission=[]
    if args.task == 'MiniGrid':
        for idx in range(args.num_envs):
            mission.append(info[idx]['mission'])
    else:
        mission = info['mission']
    with torch.no_grad():
        emb_mission=TLmodel.test_encode_text(mission)
    if args.is_finetune:
        with torch.no_grad():
            finetune_mission=EncodeModel.test_encode_text(mission)
    for index,item in enumerate(obs):
        obslist[index].append(deepcopy(item))
        actlist[index].append(deepcopy(act))
    lengths=[1 for i in range(args.num_envs)]
    # if use_predict_cost:
    if args.is_finetune:
        with torch.no_grad():
            obswithconstraint=EncodeModel.test_encode(obslist,actlist,lengths,finetune_mission)
    else:
        with torch.no_grad():
            obswithconstraint = TLmodel.test_encode(obslist,actlist,lengths,emb_mission)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret,ep_cost_train,ep_cost_true, ep_len = (
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
    )
    TLmodel.eval()
    # training loop
    for epoch in range(epochs):
        rollout_start_time = time.time()
        # collect samples until we have enough to update
        for steps in tqdm(range(local_steps_per_epoch)):
            with torch.no_grad(): 
                act, log_prob, value_r, value_c = policy.step(obswithconstraint, deterministic=False)
            action = act.detach().squeeze() if args.task in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            next_obs, reward, true_cost, terminated, truncated, info = env.step(action)
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if done or time_out:
                    if args.task== 'MiniGrid':
                        final_obs=info[idx]["final_observation"]
                    else:
                        final_obs=info["final_observation"][idx]
                    obslist[idx].append(deepcopy(final_obs))
                else:
                    obslist[idx].append(deepcopy(next_obs[idx]))
                actlist[idx].append(action[idx])
                lengths[idx] += 1
            if args.is_finetune:
                with torch.no_grad():
                    next_obswithconstraint=EncodeModel.test_encode(obslist,actlist,lengths,finetune_mission)
            cost_train = true_cost
            ep_ret += reward.cpu().numpy() if args.task in isaac_gym_map.keys() else reward
            ep_cost_train += cost_train
            ep_cost_true += true_cost.cpu().numpy() if args.task in isaac_gym_map.keys() else true_cost
            ep_len += 1
            next_obswithconstraint, reward, cost_train, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obswithconstraint, reward, cost_train, terminated, truncated)
            )
            buffer.store(
                obs=obswithconstraint,
                act=torch.tensor(action),
                obslist=deepcopy([item[0:-1] for item in obslist]),
                actlist=deepcopy([item[0:-1] for item in actlist]),
                lengths=[item-1 for item in lengths],
                mission=deepcopy(mission),
                reward=reward,
                cost=cost_train,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
            )
            obs = next_obs
            obswithconstraint=next_obswithconstraint
            epoch_end = steps >= local_steps_per_epoch - 1
            is_change=False
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obswithconstraint[idx].unsqueeze(0).to(device), deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obswithconstraint[idx].unsqueeze(0).to(device), deterministic=False
                                )
                        last_value_r = last_value_r
                        last_value_c = last_value_c
                    predict_cost=buffer.finish_path(
                                last_value_r=last_value_r, last_value_c=last_value_c, idx=idx,TL_condition=use_predict_cost,TLmodel=TLmodel,use_cost_prediction=args.use_credit_assignment,
                                obslist=obslist[idx],actlist=actlist[idx],emb_mission=emb_mission[idx]
                            )
                    if done or time_out:
                        is_change=True
                        if args.task== 'MiniGrid':
                            mission[idx] = info[idx]['mission']
                        else:
                            mission[idx] = info['mission'][idx]
                        lengths[idx] = 1
                        truecostlist[idx]=[0]
                        predictcostlist[idx]=[0]
                        obslist[idx] = [deepcopy(obs[idx])]
                        if act_dim==1:
                            act=0
                        elif act_dim==2:
                            act=(0,0)
                        actlist[idx] = [act]
                        if use_predict_cost:
                            train_cost_deque.append(predict_cost.cpu().numpy().sum())
                        else:
                            train_cost_deque.append(ep_cost_train[idx])
                        rew_deque.append(ep_ret[idx])
                        true_cost_deque.append(ep_cost_true[idx])
                        len_deque.append(ep_len[idx])
                        logger.store(
                            **{
                                "Metrics/EpRet": np.mean(rew_deque),
                                "Metrics/EpCostTrain": np.mean(train_cost_deque),
                                "Metrics/EpCostTrue": np.mean(true_cost_deque),
                                "Metrics/EpLen": np.mean(len_deque),
                            }
                        )
                        ep_ret[idx] = 0.0
                        ep_cost_train[idx] = 0.0
                        ep_cost_true[idx] = 0.0
                        ep_len[idx] = 0.0
                        logger.logged = False
            if is_change:
                with torch.no_grad():
                    emb_mission=TLmodel.test_encode_text(mission)
                if args.is_finetune:
                    with torch.no_grad():
                        finetune_mission=EncodeModel.test_encode_text(mission)
                if args.is_finetune:
                    with torch.no_grad():
                        obswithconstraint=EncodeModel.test_encode(obslist,actlist,lengths,finetune_mission)
                else:
                    with torch.no_grad():
                        obswithconstraint=TLmodel.test_encode(obslist,actlist,lengths,emb_mission)        
        rollout_end_time = time.time()
        eval_start_time = time.time()
        eval_episodes = 1 if epoch < epochs - 1 else 10
        if args.use_eval:
            for _ in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device)
                eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
                while not eval_done:
                    with torch.no_grad():
                        act, log_prob, value_r, value_c = policy.step(eval_obs, deterministic=True)
                    next_obs, reward, cost, terminated, truncated, info = env.step(
                        act.detach().squeeze().cpu().numpy()
                    )
                    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                    eval_rew += reward
                    eval_cost += cost
                    eval_len += 1
                    eval_done = terminated[0] or truncated[0]
                    eval_obs = next_obs
                eval_rew_deque.append(eval_rew)
                eval_cost_deque.append(eval_cost)
                eval_len_deque.append(eval_len)
            logger.store(
                **{
                    "Metrics/EvalEpRet": np.mean(eval_rew),
                    "Metrics/EvalEpCost": np.mean(eval_cost),
                    "Metrics/EvalEpLen": np.mean(eval_len),
                }
            )

        eval_end_time = time.time()

        # update lagrange multiplier
        ep_costs = logger.get_stats("Metrics/EpCostTrain")
        lagrange.update_lagrange_multiplier(ep_costs)

        # update policy
        data = buffer.get()
        if args.task == "MiniGrid":
            with torch.no_grad():
                old_action_probs = policy.actor(data["obs"])
                old_distribution=Categorical(old_action_probs)
        else:
            with torch.no_grad():
                old_distribution = policy.actor(data["obs"])
                old_mean = old_distribution.mean
                old_std = old_distribution.stddev

        # comnpute advantage
        advantage = data["adv_r"] - lagrange.lagrangian_multiplier * data["adv_c"]
        advantage /= (lagrange.lagrangian_multiplier + 1)
        
        if args.task == "MiniGrid":
            dataloader = DataLoader(
                dataset=BufferDataset(
                    data["obslist"],
                    data["actlist"],
                    data["obs"],
                    data["mission"],
                    data["lengths"],
                    data["act"],
                    data["log_prob"],
                    data["target_value_r"],
                    data["target_value_c"],
                    advantage,
                    old_action_probs,
                ),
                batch_size=config.get("batch_size", args.steps_per_epoch//config.get("num_mini_batch", 1)),
                shuffle=True,collate_fn=lambda x:x
            )
        else:
            dataloader = DataLoader(
                dataset=BufferDataset(
                    data["obslist"],
                    data["actlist"],
                    data["obs"],
                    data["mission"],
                    data["lengths"],
                    data["act"],
                    data["log_prob"],
                    data["target_value_r"],
                    data["target_value_c"],
                    advantage,
                    old_mean,
                    old_std
                ),
                batch_size=config.get("batch_size", args.steps_per_epoch//config.get("num_mini_batch", 1)),
                shuffle=True,collate_fn=lambda x:x
            )
        update_counts = 0
        final_kl=None
        for _ in range(config["learning_iters"]):
            for traindata in tqdm(dataloader):
                if args.task == "MiniGrid":
                    (
                        obslist_b,
                        actlist_b,
                        obs_b,
                        mission_b,
                        lengths_b,
                        act_b,
                        log_prob_b,
                        target_value_r_b,
                        target_value_c_b,
                        adv_b,
                        old_action_probs_b
                    ) = list(zip(*traindata))
                else:
                    (
                        obslist_b,
                        actlist_b,
                        obs_b,
                        mission_b,
                        lengths_b,
                        act_b,
                        log_prob_b,
                        target_value_r_b,
                        target_value_c_b,
                        adv_b,
                        old_mean_b,
                        old_std_b
                    ) = list(zip(*traindata))
                if args.is_finetune and update_counts==0:
                    Encode_optimizer.zero_grad()
                    text_featrues=EncodeModel.test_encode_text(mission_b)
                    obs_b=EncodeModel.test_encode(obslist_b,actlist_b,lengths_b,text_featrues)
                else:
                    obs_b=torch.cat([ele.unsqueeze(0) for ele in obs_b],dim=0).to(device)
                reward_critic_optimizer.zero_grad()
                target_value_r_b=torch.tensor(target_value_r_b).to(device)
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                cost_critic_optimizer.zero_grad()
                target_value_c_b=torch.tensor(target_value_c_b).to(device)
                loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b), target_value_c_b)
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost_critic.parameters():
                        loss_c += param.pow(2).sum() * 0.001
                if args.task == "MiniGrid":
                    act_b=torch.tensor(act_b).to(device)
                    old_action_probs_b=torch.cat([ele.unsqueeze(0) for ele in old_action_probs_b],dim=0).to(device)
                    old_distribution_b = Categorical(old_action_probs_b)
                    action_probs = policy.actor(obs_b)
                    distribution=Categorical(action_probs)
                    log_prob = distribution.log_prob(act_b)
                else:
                    act_b=torch.cat([item.unsqueeze(0) for item in act_b],dim=0).to(device)
                    old_mean_b=torch.cat([item.unsqueeze(0) for item in old_mean_b],dim=0).to(device)
                    old_std_b=torch.cat([item.unsqueeze(0) for item in old_std_b],dim=0).to(device)
                    old_distribution_b = Normal(loc=old_mean_b, scale=old_std_b)
                    distribution = policy.actor(obs_b)
                    log_prob = distribution.log_prob(act_b).sum(dim=-1)
                log_prob_b=torch.cat([ele.unsqueeze(0) for ele in log_prob_b],dim=0).to(device)
                ratio = torch.exp(log_prob - log_prob_b)
                temp_kl = torch.distributions.kl_divergence(
                    distribution, old_distribution_b
                ).mean()
                adv_b=torch.tensor(adv_b).to(device)
                loss_pi = (temp_kl - (1 / FOCOPS_LAM) * ratio * adv_b) * (
                    temp_kl.detach() <= dict_args['target_kl']
                ).type(torch.float32)
                loss_pi = loss_pi.mean()
                actor_optimizer.zero_grad()
                total_loss = loss_pi + 2*loss_r + loss_c \
                    if config.get("use_value_coefficient", False) \
                    else loss_pi + loss_r + loss_c
                total_loss.backward()
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost_critic_optimizer.step()
                actor_optimizer.step()
                if args.is_finetune and update_counts==0:
                    Encode_optimizer.step()
                logger.store(
                    **{
                        "Loss/Loss_reward_critic": loss_r.mean().item(),
                        "Loss/Loss_cost_critic": loss_c.mean().item(),
                        "Loss/Loss_actor": loss_pi.mean().item(),
                    }
                )

            new_distribution = policy.actor(data["obs"])
            if args.task == "MiniGrid":
                new_distribution=Categorical(new_distribution)
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .mean()
                .item()
            )
            print(kl)
            final_kl = kl
            update_counts += 1
            if kl > config["target_kl"]:
                break
        update_end_time = time.time()
        actor_scheduler.step()
        if not logger.logged:
            # log data
            logger.log_tabular("Metrics/EpRet")
            logger.log_tabular("Metrics/EpCostTrain")
            logger.log_tabular("Metrics/EpCostTrue")
            logger.log_tabular("Metrics/EpLen")
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost")
                logger.log_tabular("Metrics/EvalEpLen")
            logger.log_tabular("Train/Epoch", epoch + 1)
            logger.log_tabular("Train/TotalSteps", (epoch + 1) * args.steps_per_epoch)
            logger.log_tabular("Train/StopIter", update_counts)
            # logger.log_tabular("Train/KL", final_kl)
            logger.log_tabular("Train/LagragianMultiplier", lagrange._lagrangian_multiplier)
            logger.log_tabular("Train/LR", actor_scheduler.get_last_lr()[0])
            logger.log_tabular("Loss/Loss_reward_critic")
            logger.log_tabular("Loss/Loss_cost_critic")
            logger.log_tabular("Loss/Loss_actor")
            logger.log_tabular("Time/Rollout", rollout_end_time - rollout_start_time)
            if args.use_eval:
                logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
            logger.log_tabular("Time/Update", update_end_time - eval_end_time)
            logger.log_tabular("Time/Total", update_end_time - rollout_start_time)
            logger.log_tabular("Value/RewardAdv", data["adv_r"].mean().item())
            logger.log_tabular("Value/CostAdv", data["adv_c"].mean().item())

            logger.dump_tabular()
            if (epoch+1) % 50 == 0 or epoch == 0:
                logger.torch_save(itr=epoch)
                logger.torch_save1(itr=epoch)

    logger.close()


if __name__ == "__main__":
    args, cfg_env = single_agent_args()
    relpath = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    if args.use_predict_cost:
        if args.use_credit_assignment:
            exp_fold="our"
        else:
            exp_fold="without_credit_assignment"
    else:
        exp_fold="standard"
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, exp_fold , algo, relpath)
    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args, cfg_env)
    else:
        main(args, cfg_env)
