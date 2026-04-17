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
from sdt import Actor_SDT, Critic_SDT

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
unnecessary exploration costs for certain environments. These mappings
are not specified in the paper and mostly kept the same for
reproducibility.

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
    return action_dim, list(range(action_dim)), False

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
    
    actor = Actor_SDT(action_dim, actor_params['depth'], is_discrete)
    if args.critic == 'sdt':
        critic = Critic_SDT(critic_params)
    else:
        # TODO: pull Ryan's code for the MLP
        # critic = Critic_MLP()
        critic = None
    
    # IDK YET IF THIS IS IMPORTANT BUT ITS HERE
    #learning_rate_actor = args.learning_rate_actor
    #args.accumulate_gradients_every = 1
    
      critic_state = TrainState.create(
                apply_fn=None,
                params=critic.init(critic_key, jnp.array([envs.single_observation_space.sample()])),
                tx=optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm), optax.adam(learning_rate=args.learning_rate_critic)
                ),
            )    
      
      actor_state = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(actor_key, jnp.array([envs.single_observation_space.sample()])),
                    tx=optax.chain(
                        optax.clip_by_global_norm(args.max_grad_norm),
                        optax.inject_hyperparams(optax.adam)(learning_rate_actor),
                    ),
                    grad_accum=jax.tree.map(
                        jnp.zeros_like, actor.init(actor_key, jnp.array([envs.single_observation_space.sample()]))
                    ),
                    indices=None,
                )  
      
      critic.apply = jax.jit(critic.apply)
      actor.apply = jax.jit(actor.apply)
      
              lr_scheduler = optax.contrib.reduce_on_plateau(patience=3, factor=0.5)
        lr_scheduler_state = lr_scheduler.init(actor_state.params)
        #actor.apply = jax.jit(actor.apply)
        #critic.apply = jax.jit(critic.apply)
            
        episode_stats = EpisodeStatistics(
            episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
            episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
            returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
        )

if __name__ == '__main__':
    main()
