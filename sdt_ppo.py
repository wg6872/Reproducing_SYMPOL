# sdt_ppo.py
# Author(s): Evan Soper
# Implementation of PPO for SDT-related benchmarks
# Adapted from: https://github.com/s-marton/SYMPOL/blob/master/ppo.py

import os
from datetime import datetime

import jax
import flax
import numpy as np
import jax.numpy as jnp
import gymnasium as gym
import optax
import wandb

from flax.training.train_state import TrainState
from minigrid.wrappers import OneHotPartialObsWrapper, ViewSizeWrapper

from env_wrappers import FlatCurrentReducedWrapper, NormalizeWrapperLunarLander
from sdt_ppo_config import get_args, get_sdt_params, get_mlp_params
from sdt import Actor_SDT, Critic_SDT
from mlp import Critic_MLP

# Fix OOM issues
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

"""Halper class for neat episode statistic storage."""
@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array

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
unnecessary exploration costs for certain environments. These mappings
are not specified in the paper and mostly kept the same for
reproducibility.

Args:
    env_id: ID of the environment
    action_space: action space of a single env
    
Returns:
    action_dim: number of actions the policy outputs
    action_indices: map for model output => env actions, or None if action space is unmodified
    is_discrete: True if the action space is discrete
"""
def get_action_mapping(env_id, action_space):
    if isinstance(action_space, gym.spaces.Box):
        # if the action space is continuous
        action_dim = action_space.shape[-1]
        return action_dim, None, False

    if any(s in env_id for s in ['Crossing', 'DistShift', 'Empty', 'LavaGap', 'FourRooms', 'Dynamic-Obstacles']):
        return 3, [0, 1, 2], True

    if any(s in env_id for s in ['MultiRoom', 'Unlock', 'GoToDoor', 'RedBlueDoors']):
        if 'GoToDoor' in env_id:
            return 4, [0, 1, 2, 6], True
        return 4, [0, 1, 2, 5], True

    if any(s in env_id for s in ['UnlockPickup', 'DoorKey']):
        return 5, [0, 1, 2, 3, 5], True

    action_dim = action_space.n
    return action_dim, None, False

def rollout(args, params, actor_state, critic_state, action_indices, episode_stats, next_obs, next_done, storage, key, global_step, envs):
    for step in range(params['n_steps']):
        global_step += args.n_envs
        
        # sample action and value estimate
        storage, action, key = get_action_and_value(actor_state, critic_state, next_obs, next_done, storage, step, key)
        action = np.array(action)
        if action_indices is not None:
            # apply the mapping from output to action space
            action = np.array([action_indices[a] for a in action])
            
        next_obs, reward, next_done, trunc, _ = envs.step(action)
        
        new_episode_return = episode_stats.episode_returns + reward
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            # if the episode is finished, reset the accumulated return and length
            episode_returns=(new_episode_return) * (1 - next_done) * (1 - trunc),
            episode_lengths=(new_episode_length) * (1 - next_done) * (1 - trunc),
            # only update the final return value if the episode is done
            returned_episode_returns=jnp.where(
                next_done + trunc,
                new_episode_return,
                episode_stats.returned_episode_returns,
            ),
            returned_episode_lengths=jnp.where(
                next_done + trunc,
                new_episode_length,
                episode_stats.returned_episode_lengths,
            ),
        )
        
        storage = storage.replace(rewards=storage.rewards.at[step].set(reward))
        
    return actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step

def main():
    args = get_args()
    actor_params = get_sdt_params(args.env_id)
    critic_params = get_sdt_params(args.env_id) if args.critic == 'sdt' else get_mlp_params(args.env_id)

    raw_envs = [make_env(args.env_id) for _ in range(args.n_envs)]
    envs = gym.vector.AsyncVectorEnv(raw_envs)
    action_dim, action_indices, is_discrete = get_action_mapping(args.env_id, envs.single_action_space)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = '-'.join([args.run_name, timestamp])
    # logs and visualizes training metrics and outputs
    wandb_run = wandb.init(
                project=f"{args.exp_name}_{args.env_id}",
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True, 
            )
    
    actor = Actor_SDT(action_dim, actor_params['depth'], is_discrete)
    if args.critic == 'sdt':
        critic = Critic_SDT(critic_params['depth'])
    else:
        critic = Critic_MLP(critic_params['num_layers'], critic_params['neurons_per_layer'])

    # in JAX, parameters have to be passed explicitly when the model is initialized
    # with a key (ensures randomness) and a fake observation (to init. architecture shape) 
    model_key = jax.random.PRNGKey(args.seed)
    model_key, actor_key, critic_key = jax.random.split(model_key, 3)
    lr_actor, lr_critic = actor_params['learning_rate_actor'], critic_params['learning_rate_critic']
    
    actor_state = TrainState.create(
                apply_fn=actor.apply,
                params=actor.init(actor_key, jnp.array([envs.single_observation_space.sample()])),
                tx=optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.inject_hyperparams(optax.adam)(lr_actor),
                )
            )
    
    critic_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic.init(critic_key, jnp.array([envs.single_observation_space.sample()])),
            tx=optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm), 
                optax.adam(learning_rate=lr_critic)
            ),
        )
    
    lr_scheduler = optax.contrib.reduce_on_plateau(patience=3, factor=0.5)
    lr_scheduler_state = lr_scheduler.init(actor_state.params)
            
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
    )

if __name__ == '__main__':
    main()
