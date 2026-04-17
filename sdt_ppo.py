# sdt_ppo.py
# Author(s): Evan Soper
# Implementation of PPO for SDT-related benchmarks
# Adapted from: https://github.com/s-marton/SYMPOL/blob/master/ppo.py

import os
from datetime import datetime
import gymnasium as gym
import wandb
from minigrid.wrappers import OneHotPartialObsWrapper, ViewSizeWrapper
from env_wrappers import FlatCurrentReducedWrapper, NormalizeWrapperLunarLander
from sdt_ppo_config import get_args, get_sdt_params, get_mlp_params

# Fix OOM issues
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

"""
Helper to make a single gym environment. Wrappers are mostly kept the same
from the reference code for reproducibility.

Args:
    env_id: ID of the environment (for example, CartPole-v1)
    view_size: size of the agent's partial observation window    
"""
def make_env(env_id, view_size=3):
    env = gym.make(env_id)

    if "MiniGrid" in env_id:
        env = ViewSizeWrapper(env, agent_view_size=view_size)
        env = OneHotPartialObsWrapper(env)
        env = FlatCurrentReducedWrapper(env)
    elif "LunarLander" in env_id:
        env = NormalizeWrapperLunarLander(env)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

"""
Creates metadata for the action space, including a mapping that limits
unnecessary exploration costs for certain environments. These are not
specified in the paper and mostly kept the same for reproducibility.

Args:
    env_id: ID of the environment
    action_space: action space of a single env
    
Returns:
    action_dim: number of actions the policy outputs
    action_indices: map model outputs => environment actions
    is_discrete: True if the action space is discrete
"""
def get_action_mapping(env_id, action_space):
    if isinstance(action_space, gym.spaces.Box):
        # if the action space is continuous
        action_dim = action_space.shape[-1]
        return action_dim, list(range(action_dim)), False

    if any(s in env_id for s in ['Crossing', 'DistShift', 'Empty', 'LavaGap', 'FourRooms', 'Dynamic-Obstacles']):
        return 3, [0, 1, 2], True

    if any(s in env_id for s in ['MultiRoom', 'Unlock', 'GoToDoor', 'RedBlueDoors']):
        if 'GoToDoor' in env_id:
            return 4, [0, 1, 2, 6], True
        return 4, [0, 1, 2, 5], True

    if any(s in env_id for s in ['UnlockPickup', 'DoorKey']):
        return 5, [0, 1, 2, 3, 5], True

    action_dim = action_space.n
    return action_dim, list(range(action_dim)), False\

def main():
    args = get_args()
    actor_params = get_sdt_params()
    critic_params = get_sdt_params() if args.critic == 'sdt' else get_mlp_params()

    envs = [make_env(args.env_id) for _ in range(args.n_envs)]
    action_dim, action_indices, is_discrete = get_action_mapping(args.env_id, envs.single_action_space)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = '-'.join([args.run_name, timestamp])
    wandb_run = wandb.init(
                project=f"{args.exp_name}_{args.env_id}",
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True, 
            )
    

if __name__ == '__main__':
    main()
