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
from distrax import Categorical, MultivariateNormalDiag

from flax.training.train_state import TrainState
from minigrid.wrappers import OneHotPartialObsWrapper, ViewSizeWrapper

from env_wrappers import FlatCurrentReducedWrapper, NormalizeWrapperLunarLander
from sdt_ppo_config import get_args, get_sdt_params, get_mlp_params
from sdt import Actor_SDT, Critic_SDT
from mlp import Critic_MLP

# Fix OOM issues
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

@flax.struct.dataclass
class batch:
    """Halper class for neat minibatch storage."""
    obs: jnp.array
    actions: jnp.array
    log_probs: jnp.array
    advantages: jnp.array
    returns: jnp.array

@flax.struct.dataclass
class EpisodeStatistics:
    """Halper class for neat episode statistic storage."""
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array
    
@flax.struct.dataclass
class Storage:
    """Halper class for neat sequential storage."""
    obs: jnp.array
    actions: jnp.array
    log_probs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array

def setup():
    """
    Setup an experiment run on Weights & Biases for visualization. 

    Returns:
        dict with all hyperparameters and configuration details for this run.
    """
    args = get_args()
    
    actor_params = get_sdt_params(args.env_id)
    
    if args.critic == 'mlp':
        critic_params = get_mlp_params(args.env_id)
    else:
        critic_params = {}
    
    config = {**actor_params, **critic_params, **vars(args)}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = '-'.join([args.run_name, timestamp])
    
    wandb.init(
        project=f"{args.exp_name}_{args.env_id}",
        config=config,
        name=run_name,
        monitor_gym=True,
        save_code=True, 
    )
    
    return config

def make_env(env_id, view_size=3):
    """
    Make a single gym environment. Wrappers are mostly kept the same from
    the reference code for reproducibility.

    Args:
        env_id: ID of the environment (for example, CartPole-v1)
        view_size: size of the agent's partial observation window

    Returns:
        gym environment
    """
    env = gym.make(env_id)

    if "MiniGrid" in env_id:
        env = ViewSizeWrapper(env, agent_view_size=view_size)
        env = OneHotPartialObsWrapper(env)
        env = FlatCurrentReducedWrapper(env)
    elif "LunarLander" in env_id:
        env = NormalizeWrapperLunarLander(env)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def get_action_mapping(env_id, action_space):
    """
    Compute metadata for the action space, including a mapping that limits
    unnecessary exploration costs for certain environments. These mappings
    are not specified in the paper and mostly kept the same for
    reproducibility.

    Args:
        env_id: ID of the environment
        action_space: action space of a single env
        
    Returns:
        action_dim: number of actions the policy outputs
        action_indices: map for model output => env actions; None action space unmodified
        is_discrete: True if the action space is discrete
    """
    if isinstance(action_space, gym.spaces.Box):
        # if the action space is continuous
        action_dim = action_space.shape[-1]
        return action_dim, None, False
    
    action_dim = action_space.n
    TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP, _, TOGGLE, DONE = range(7)

    if any(s in env_id for s in ['Crossing', 'DistShift', 'Empty', 'LavaGap', 'FourRooms', 'Dynamic-Obstacles']):
        # navigation only
        return 3, [TURN_LEFT, TURN_RIGHT, MOVE_FORWARD], True

    if any(s in env_id for s in ['MultiRoom', 'Unlock', 'GoToDoor', 'RedBlueDoors']):
        if 'GoToDoor' in env_id:
            return 4, [TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, DONE], True
        return 4, [TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, TOGGLE], True

    if any(s in env_id for s in ['UnlockPickup', 'DoorKey']):
        return 5, [TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP, TOGGLE], True

    return action_dim, None, True

def get_actor_and_critic_state(config, envs, action_dim, is_discrete):
    """
    Build training states for the actor and the critic.

    Args:
        config: dict with run configuration details
        envs: list of the gym environments used for this run
        action_dim: number of actions the policy outputs
        is_discrete: True if the action space is discrete

    Returns:
        TrainState objects for the actor and critic, respectively.
    """
    actor = Actor_SDT(action_dim, config['depth'], is_discrete)
    if config['critic'] == 'sdt':
        critic = Critic_SDT(config['depth'])
    else:
        critic = Critic_MLP(config['num_layers'], config['neurons_per_layer'])

    # in JAX, parameters have to be passed explicitly when the model is initialized
    # with a key (ensures randomness) and a fake observation (to init. architecture shape) 
    model_key = jax.random.PRNGKey(config['seed'])
    model_key, actor_key, critic_key = jax.random.split(model_key, 3)
    
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, 
                          jnp.array([envs.single_observation_space.sample()])
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(config['max_grad_norm']),
            optax.inject_hyperparams(optax.adam)(config['learning_rate_actor']),
        )
    )
    
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic.init(critic_key, 
                           jnp.array([envs.single_observation_space.sample()])
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(config['max_grad_norm']), 
            optax.adam(learning_rate=config['learning_rate_critic'])
        ),
    )

    return actor_state, critic_state

@jax.jit
def compute_gae(config, critic_state, next_obs, next_done, storage):
    """
    Compute Generalized Advantage Estimation (GAE) for a collected rollout.

    Args:
        config: dict with run configuration details
        critic_state: TrainState for the critic network
        next_obs: obs after final timestep of the rollout (n_envs, obs_dim)
        next_done: Done flags after final timestep (n_envs,)
        storage: Storage object containing rollout data

    Returns:
        storage: Updated Storage object with advantages and returns
    """
    def compute_gae_once(carry, inp):
        """Helper function for scan compatibility. Compute the advantage at one step."""
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + config['gamma'] * nextvalues * nextnonterminal - curvalues
        advantages = delta + config['gamma'] * config['gae_lambda'] * nextnonterminal * advantages
        return advantages
    
    next_value = critic_state.apply_fn(critic_state.params, next_obs).squeeze()

    advantages = jnp.zeros((config['n_envs'],))
    dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
    values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
    _, advantages = jax.lax.scan(compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True)
    
    storage = storage.replace(
        advantages=advantages,
        returns=advantages + storage.values,
    )
    
    return storage

@jax.jit
def get_ppo_loss(config, actor_state, critic_state, actor_params, critic_params, is_discrete, batch):
    """
    Compute the policy loss, which is used to update action probabilities; 
    value loss, which measures the difference between the critic's state
    value and the reward obtained; and KL divergence from a single minibatch obs.
    Actor and critic parameters are explicitly passed for gradient calculations.
    
    Returns:
        total loss, (policy loss, value loss, entropy, KL divergence)
    """
    new_val = critic_state.apply_fn(critic_params, batch.obs).squeeze()
    
    pol = actor_state.apply_fn(actor_params, batch.obs)
    if is_discrete:
        dist = Categorical(logits=pol)
    else:
        dist = MultivariateNormalDiag(pol[0], jnp.exp(pol[1]))
    
    new_log_probs = dist.log_prob(batch.actions)
    entropy = dist.entropy().mean()
        
    log_ratio = new_log_probs - batch.log_probs
    ratio = jnp.exp(log_ratio)
    # estimate divergence from old policy
    approx_kl = ((ratio - 1) - log_ratio).mean()

    if config['norm_adv']:
        mb_advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
    else:
         mb_advantages = batch.advantages

    pol_loss = -mb_advantages * ratio
    clipped_loss = -mb_advantages * jnp.clip(ratio, 1 - config['clip_coef'], 1 + config['clip_coef'])
    pol_loss = jnp.maximum(pol_loss, clipped_loss).mean()

    v_loss = 0.5 * ((new_val - batch.returns) ** 2).mean()

    # total loss includes policy loss, entropy bonus, and value loss
    loss = pol_loss - config['ent_coef'] * entropy + v_loss * config['vf_coef']
    
    return loss, (pol_loss, v_loss, entropy, jax.lax.stop_gradient(approx_kl))

@jax.jit
def update_ppo(config, actor_state, critic_state, storage, key, is_discrete):
    """Train and update model parameters using one collected rollout (in storage)."""
    def prepare_data(x, perm, minibatch_size):
        """Flatten and segment batch data to pass minibatches into SGD."""
        x = x.reshape((-1,) + x.shape[2:])

        num_minibatches = x.shape[0] // minibatch_size
        size = num_minibatches * minibatch_size

        x = x[perm][:size]
        x = x.reshape((num_minibatches, minibatch_size) + x.shape[1:])

        return x

    def update_epoch(carry, _):
        """Compute gradients and update parameters for one pass through the rollout data."""
        actor_state, critic_state, key = carry

        key, subkey = jax.random.split(key)
        batch_size = storage.obs.shape[0] * storage.obs.shape[1]
        perm = jax.random.permutation(subkey, batch_size)

        shuffled_storage = jax.tree_util.tree_map(
            lambda x: prepare_data(x, perm, config['minibatch_size']),
            storage
        )

        def update_minibatch(carry, minibatch):
            """Compute gradients and update parameters for one minibatch."""
            actor_state, critic_state = carry

            (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), (actor_grads, critic_grads) = \
                jax.value_and_grad(
                    get_ppo_loss,
                    argnums=(3, 4),
                    has_aux=True,
                )(
                    config,
                    actor_state,
                    critic_state,
                    actor_state.params,
                    critic_state.params,
                    is_discrete,
                    minibatch,
                )

            # update the model parameters after each minibatch (SDT accumulation is 1)
            actor_state = actor_state.apply_gradients(grads=actor_grads)
            critic_state = critic_state.apply_gradients(grads=critic_grads)

            return (actor_state, critic_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl)

        (actor_state, critic_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
            update_minibatch,
            (actor_state, critic_state),
            shuffled_storage,
        )

        return (actor_state, critic_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl)

    (actor_state, critic_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
        update_epoch,
        (actor_state, critic_state, key),
        xs=None,
        length=config['n_update_epochs'],
    )

    return actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

@jax.jit
def get_action(actor_state, critic_state, is_discrete, next_obs, next_done, storage, step, key):
    """
    Helper to sample an action and calculate the state value.

    Args:
        actor_state: TrainState object with actor configuration
        critic_state: TrainState object with critic configuration
        is_discrete: True if the action space is discrete
        next_obs: observation currently visible to the agent
        next_done: 1 if the previous action ended the episode, else 0
        storage: Storage object for full sequential run information
        step: index for this current step
        key: JAX key for this run

    Returns:
        Updated storage object, selected action, key used to sample the action
    """
    pol = actor_state.apply_fn(actor_state.params, next_obs)
    if is_discrete:
        dist = Categorical(logits=pol)
    else:
        dist = MultivariateNormalDiag(pol[0], jnp.exp(pol[1]))
    
    value = critic_state.apply_fn(critic_state.params, next_obs)

    # randomness is explicit in JAX for reproducibility
    key, sub_key = jax.random.split(key)
    action = dist.sample(seed=sub_key)
    log_prob = dist.log_prob(action)
    
    storage = storage.replace(
        obs=storage.obs.at[step].set(next_obs),
        dones=storage.dones.at[step].set(next_done),
        actions=storage.actions.at[step].set(action),
        log_probs=storage.log_probs.at[step].set(log_prob),
        values=storage.values.at[step].set(value.squeeze()),
    )

    return storage, action, key

def rollout(config, actor_state, critic_state, action_indices, is_discrete, episode_stats, next_obs, next_done, storage, key, global_step, envs):
    """Collect n_steps timesteps of experience from all envs."""
    for step in range(config['n_steps']):
        global_step += config['n_envs']
        
        # sample action and value estimate
        storage, action, key = get_action(actor_state, critic_state, is_discrete, next_obs, next_done, storage, step, key)
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
        
    return episode_stats, next_obs, next_done, storage, key, global_step

def evaluate_agent(actor_state, config, is_discrete, action_indices):
    env = make_env(config["env_id"])
    scores = []

    for _ in range(config["n_eval_episodes"]):
        obs, _ = env.reset()
        done, trunc = False, False
        total_reward = 0.0

        while not (done or trunc):
            obs_batch = jnp.array([obs])
            pol = actor_state.apply_fn(actor_state.params, obs_batch)

            if is_discrete:
                action = int(jnp.argmax(pol, axis=-1)[0])
                if action_indices is not None:
                    action = action_indices[action]

            else:
                mean, _ = pol
                action = np.array(mean[0])

            obs, reward, done, trunc, _ = env.step(action)
            total_reward += reward

        scores.append(total_reward)

    env.close()
    return float(np.mean(scores))

def main():
    config = setup()

    raw_envs = [make_env(config['env_id']) for _ in range(config['n_envs'])]
    envs = gym.vector.AsyncVectorEnv(raw_envs)
    
    action_dim, action_indices, is_discrete = get_action_mapping(config['env_id'], envs.single_action_space)
    
    actor_state, critic_state = get_actor_and_critic_state(config, envs, action_dim, is_discrete)
    
    # prepare parameters for training loop
    next_obs, _ = envs.reset(seed=config['env_seed'])
    next_done = np.zeros(config['n_envs']).astype(bool)
    batch_size = int(config['n_envs'] * config['n_steps'])
    key = jax.random.PRNGKey(config["seed"])
    
    iteration = 1
    last_eval = 0
    global_step = 0
    total_time = 0
    
    avg_episodic_return_list = []
    
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(config['n_envs'], dtype=jnp.float32),
        episode_lengths=jnp.zeros(config['n_envs'], dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(config['n_envs'], dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(config['n_envs'], dtype=jnp.int32),
    )
    
    while global_step < config['total_steps']:
        start_time = datetime.now()
        
        # storage for data in a single rollout
        storage = Storage(
                obs=jnp.zeros((config['n_steps'], config['n_envs']) + envs.single_observation_space.shape),
                actions=jnp.zeros((config['n_steps'], config['n_envs']) + envs.single_action_space.shape, dtype=jnp.int32),
                log_probs=jnp.zeros((config['n_steps'], config['n_envs'])),
                dones=jnp.zeros((config['n_steps'], config['n_envs'])),
                values=jnp.zeros((config['n_steps'], config['n_envs'])),
                advantages=jnp.zeros((config['n_steps'], config['n_envs'])),
                returns=jnp.zeros((config['n_steps'], config['n_envs'])),
                rewards=jnp.zeros((config['n_steps'], config['n_envs'])),
        )
        
        actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step = rollout(
            config, actor_state, critic_state, action_indices, is_discrete, episode_stats, next_obs, next_done, storage, key, global_step
        )
        
        storage = compute_gae(config, critic_state, next_obs, next_done, storage)
        
        actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            config,
            actor_state,
            critic_state,
            storage,
            key,
            is_discrete
        )
        
        elapsed_time = datetime.now() - start_time
        total_time += elapsed_time

        avg_return = np.mean(np.array(episode_stats.returned_episode_returns))
        avg_episodic_return_list.append(avg_return)

        # evaluation "bucket"
        current_eval = global_step // config["eval_freq"]
        
        # determine evaluation criteria
        is_first = (iteration == 1)
        is_new = (current_eval > last_eval)
        is_final = (global_step + batch_size >= config["total_steps"])

        if is_first or is_new or is_final:
            last_eval = current_eval
            render_now = config["render_each_eval"] or is_final

            eval_score = evaluate_agent(actor_state, envs, config, render_now)

            print(f"[eval] step={global_step} score={eval_score}")

            wandb.log({
                "eval/score": eval_score,
                "train/avg_return": avg_return,
                "global_step": global_step,
            })

        iteration += 1
        
if __name__ == '__main__':
    main()
