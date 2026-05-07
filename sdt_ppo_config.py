# sdt_ppo.py
# Author(s): Evan Soper
# Parameters for SDT-related benchmarks using PPO
# Adapted from: https://github.com/s-marton/SYMPOL/blob/master/configs.py
# Optimal hyperparameters are taken from the paper

import argparse

SDT_PPO_PARAMS = {
    "CartPole-v1": {
        "adamW": False,
        "critic": "mlp",
        "depth": 7,
        "ent_coef": 0.0,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "learning_rate_actor": 0.0009255,
        "learning_rate_critic": 0.0001238,
        "max_grad_norm": 0.1,
        "minibatch_size": 128,
        "n_envs": 15,
        "n_steps": 512,
        "n_update_epochs": 4,
        "norm_adv": True,
        "reduce_lr": False,
        "temperature": 1,
        "vf_coef": 0.50,
    },
    "Pendulum-v1": {
        "adamW": False,
        "critic": "mlp",
        "depth": 7,
        "ent_coef": 0.2,
        "gae_lambda": 0.9,
        "gamma": 0.9,
        "learning_rate_actor": 0.000364,
        "learning_rate_critic": 0.0001127,
        "max_grad_norm": 0.1,
        "minibatch_size": 128,
        "n_envs": 7,
        "n_steps": 256,
        "n_update_epochs": 7,
        "norm_adv": False,
        "reduce_lr": False,
        "temperature": 0.1,
        "vf_coef": 0.50,
    },
    "MountainCar-v0": {
        "adamW": False,
        "critic": "sdt",
        "depth": 7,
        "ent_coef": 0.0,
        "gae_lambda": 0.9,
        "gamma": 0.99,
        "learning_rate_actor": 0.003937312675632405,
        "learning_rate_critic": 0.002474020951866176,
        "max_grad_norm": 0.5,
        "minibatch_size": 64,
        "n_envs": 4,
        "n_steps": 128,
        "n_update_epochs": 3,
        "norm_adv": True,
        "reduce_lr": False,
        "temperature": 1,
        "vf_coef": 0.75,
    },
    "MountainCarContinuous-v0": {
        "adamW": False,
        "critic": "mlp",
        "depth": 7,
        "ent_coef": 0.0,
        "gae_lambda": 0.9,
        "gamma": 0.99,
        "learning_rate_actor": 0.0008297,
        "learning_rate_critic": 0.007393,
        "max_grad_norm": 0.5,
        "minibatch_size": 64,
        "n_envs": 14,
        "n_steps": 512,
        "n_update_epochs": 1,
        "norm_adv": False,
        "reduce_lr": False,
        "temperature": 1,
        "vf_coef": 0.25,
    },
    "Acrobot-v1": {
        "adamW": False,
        "critic": "mlp",
        "depth": 6,
        "ent_coef": 0.1,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "learning_rate_actor": 0.0016799276982439016,
        "learning_rate_critic": 0.0003204461305956749,
        "max_grad_norm": 0.1,
        "minibatch_size": 128,
        "n_envs": 6,
        "n_steps": 128,
        "n_update_epochs": 10,
        "norm_adv": False,
        "reduce_lr": False,
        "temperature": 0.5,
        "vf_coef": 0.50,
    },
    "LunarLander-v2": {
        "adamW": False,
        "critic": "mlp",
        "depth": 8,
        "ent_coef": 0.2,
        "gae_lambda": 0.99,
        "gamma": 0.999,
        "learning_rate_actor": 0.0006108017244234425,
        "learning_rate_critic": 0.0011201875111956177,
        "max_grad_norm": 1.0,
        "minibatch_size": 128,
        "n_envs": 7,
        "n_steps": 512,
        "n_update_epochs": 2,
        "norm_adv": True,
        "reduce_lr": False,
        "temperature": 1,
        "vf_coef": 0.75,
    },
    "MiniGrid-LavaGapS5-v0": {
        "adamW": False,
        "critic": "sdt",
        "depth": 7,
        "ent_coef": 0.2,
        "gae_lambda": 0.99,
        "gamma": 0.999,
        "learning_rate_actor": 0.0004584,
        "learning_rate_critic": 0.0002554,
        "max_grad_norm": 0.5,
        "minibatch_size": 512,
        "n_envs": 10,
        "n_steps": 256,
        "n_update_epochs": 8,
        "norm_adv": True,
        "reduce_lr": False,
        "temperature": 1,
        "vf_coef": 0.75,
    },
    "DoorKey-5x5-v0": {
        "adamW": False,
        "critic": "mlp",
        "depth": 6,
        "ent_coef": 0.1,
        "gae_lambda": 0.95,
        "gamma": 0.9,
        "learning_rate_actor": 0.0008919,
        "learning_rate_critic": 0.001508,
        "max_grad_norm": 0.1,
        "minibatch_size": 256,
        "n_envs": 10,
        "n_steps": 256,
        "n_update_epochs": 10,
        "norm_adv": True,
        "reduce_lr": False,
        "temperature": 1,
        "vf_coef": 0.75,
    },
    "Empty-Random-6x6-v0": {
        "adamW": False,
        "critic": "sdt",
        "depth": 7,
        "ent_coef": 0.1,
        "gae_lambda": 0.9,
        "gamma": 0.99,
        "learning_rate_actor": 0.004439821550585952,
        "learning_rate_critic": 0.0004488603528109438,
        "max_grad_norm": 0.1,
        "minibatch_size": 512,
        "n_envs": 10,
        "n_steps": 512,
        "n_update_epochs": 5,
        "norm_adv": True,
        "reduce_lr": False,
        "temperature": 1,
        "vf_coef": 0.75,
    },
    "MiniGrid-LavaGapS7-v0": {
        "adamW": False,
        "critic": "sdt",
        "depth": 7,
        "ent_coef": 0.1,
        "gae_lambda": 0.95,
        "gamma": 0.95,
        "learning_rate_actor": 0.0019084174917437576,
        "learning_rate_critic": 0.00510744773741252,
        "max_grad_norm": 0.1,
        "minibatch_size": 256,
        "n_envs": 13,
        "n_steps": 128,
        "n_update_epochs": 4,
        "norm_adv": True,
        "reduce_lr": False,
        "temperature": 1,
        "vf_coef": 0.25,
    },
    "MiniGrid-DistShift1-v0": {
        "adamW": False,
        "critic": "sdt",
        "depth": 7,
        "ent_coef": 0.1,
        "gae_lambda": 0.9,
        "gamma": 0.95,
        "learning_rate_actor": 0.0008335968492146473,
        "learning_rate_critic": 0.0020439708107133216,
        "max_grad_norm": 1000.0,
        "minibatch_size": 512,
        "n_envs": 5,
        "n_steps": 512,
        "n_update_epochs": 7,
        "norm_adv": True,
        "reduce_lr": False,
        "temperature": 1,
        "vf_coef": 0.75,
    },
}

MLP_PPO_PARAMS = {
    "CartPole-v1": {
        "num_layers": 2,
        "neurons_per_layer": 139,
    },
    "Pendulum-v1": {
        "num_layers": 2,
        "neurons_per_layer": 75,
    },
    "MountainCar-v0": {
        "num_layers": 3,
        "neurons_per_layer": 144,
    },
    "MountainCarContinuous-v0": {
        "num_layers": 2,
        "neurons_per_layer": 240,
    },
    "Acrobot-v1": {
        "num_layers": 2,
        "neurons_per_layer": 185,
    },
    "LunarLander-v2": {
        "num_layers": 3,
        "neurons_per_layer": 46,
    },
    "MiniGrid-LavaGapS5-v0": {
        "num_layers": 1,
        "neurons_per_layer": 76,
    },
    "MiniGrid-DoorKey-8x8": {
        "num_layers": 1,
        "neurons_per_layer": 169,
    },
    "MiniGrid-Empty-8x8": {
        "num_layers": 3,
        "neurons_per_layer": 112,
    },
    "MiniGrid-LavaGapS7-v0": {
        "num_layers": 1,
        "neurons_per_layer": 28,
    },
    "MiniGrid-DistShift1-v0": {
        "num_layers": 2,
        "neurons_per_layer": 158,
    },
}

def get_sdt_params(env_id):
    return SDT_PPO_PARAMS[env_id].copy()

def get_mlp_params(env_id):
    return MLP_PPO_PARAMS[env_id].copy()

def get_args():
    parser = argparse.ArgumentParser(description="PPO runner for SDT actor.")
    
    parser.add_argument("--env_id", type=str, required=True, help="ex. CartPole-v1")
    parser.add_argument("--actor", type=str, required=True, help="sdt or d-sdt")
    parser.add_argument("--critic", type=str, required=True, help="sdt or mlp")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--exp_name", type=str, default="experiment")
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument("--total_steps", type=int, default=1000000)
    parser.add_argument("--clip_coef", type=int, default=0.1)
    parser.add_argument("--eval_freq", type=int, default=50000)

    return parser.parse_args()
