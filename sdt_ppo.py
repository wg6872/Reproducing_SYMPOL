# sdt_ppo.py
# Author(s): Evan Soper
# Implementation of PPO for SDT-related benchmarks
# Adapted from: https://github.com/s-marton/SYMPOL/blob/master/ppo.py

import os
import copy
from functools import partial
from datetime import datetime, timedelta

import jax
import flax
import numpy as np
import jax.numpy as jnp
import gymnasium as gym
import optax
import wandb
from distrax import Categorical, MultivariateNormalDiag
from flax.core import freeze, unfreeze

from flax.training.train_state import TrainState
from minigrid.wrappers import OneHotPartialObsWrapper, ViewSizeWrapper

from sdt_env_wrappers import FlatCurrentReducedWrapper, NormalizeWrapperLunarLander
from sdt_ppo_config import get_args, get_sdt_params, get_mlp_params
from sdt import Actor_SDT, Critic_SDT
from mlp import CriticMLP
from sdt_plot_util import plot_dsdt_from_params

# Fix OOM issues
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

os.environ["WANDB_MODE"] = "online"

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

@flax.struct.dataclass
class TrainConfig:
    """
    Apparently, we can't pass a dict into functions with a jax.jit decorator. So, this helper
    class passes the data to the several functions called during training.
    """
    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    minibatch_size: int
    n_update_epochs: int
    n_envs: int
    norm_adv: bool

def setup():
    """
    Setup an experiment run on Weights & Biases for visualization. 

    Returns:
        dict with all hyperparameters and configuration details for this run,
        dataclass containing hyperparameters for JAX compatibility.
    """
    args = get_args()
    
    actor_params = get_sdt_params(args.env_id)
    
    if args.critic == 'mlp':
        critic_params = get_mlp_params(args.env_id)
    elif args.critic == 'sdt':
        critic_params = get_sdt_params(args.env_id)
    else:
        critic_params = {}
    
    config = {**actor_params, **critic_params, **vars(args)}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = '-'.join([args.run_name, timestamp])
    
    wandb.init(
        project=f"{args.exp_name}_{args.env_id}",
        config=config,
        name=run_name,
        group=args.run_name,
    )
    
    train_cfg = TrainConfig(
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_coef=config['clip_coef'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        minibatch_size=config['minibatch_size'],
        n_update_epochs=config['n_update_epochs'],
        n_envs=config['n_envs'],
        norm_adv=config['norm_adv'],
    )
    
    return config, train_cfg

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
        critic = CriticMLP(config['num_layers'], config['neurons_per_layer'])

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
        config: dataclass with run configuration details
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

        delta = reward + config.gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + config.gamma * config.gae_lambda * nextnonterminal * advantages
        
        # must return the next carry and output for scan to work
        return advantages, advantages
    
    next_value = critic_state.apply_fn(critic_state.params, next_obs).squeeze()

    advantages = jnp.zeros(next_done.shape, dtype=jnp.float32)
    dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
    values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
    _, advantages = jax.lax.scan(compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True)
    
    storage = storage.replace(
        advantages=advantages,
        returns=advantages + storage.values,
    )
    
    return storage

# Apparently, built-in types need to be declared as static to use @jax.jit
@partial(jax.jit, static_argnames=['config', 'is_discrete'])
def get_ppo_loss(config, actor_state, critic_state, actor_params, critic_params, is_discrete, batch):
    """
    Compute the policy loss, which is used to update action probabilities; 
    value loss, which measures the difference between the critic's state
    value and the reward obtained; and KL divergence from a single minibatch obs.
    Actor and critic parameters are explicitly passed for gradient calculations.
    
    NOTE: config must be a dataclass for jax.jit
    
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

    if config.norm_adv:
        mb_advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
    else:
        mb_advantages = batch.advantages

    pol_loss = -mb_advantages * ratio
    clipped_loss = -mb_advantages * jnp.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
    pol_loss = jnp.maximum(pol_loss, clipped_loss).mean()

    v_loss = 0.5 * ((new_val - batch.returns) ** 2).mean()

    # total loss includes policy loss, entropy bonus, and value loss
    loss = pol_loss - config.ent_coef * entropy + v_loss * config.vf_coef
    
    return loss, (pol_loss, v_loss, entropy, jax.lax.stop_gradient(approx_kl))

@partial(jax.jit, static_argnames=['config', 'is_discrete'])
def update_ppo(config, actor_state, critic_state, storage, key, is_discrete):
    """
    Train and update model parameters using one collected rollout (in storage).
    
    NOTE: config must be a dataclass for jax.jit
    """
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
            lambda x: prepare_data(x, perm, config.minibatch_size),
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
        length=config.n_update_epochs,
    )

    return actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

@partial(jax.jit, static_argnames=['is_discrete'])
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

def convert_to_discrete(params, is_discrete):
    """
    Helper function used in model evaluation to convert the optimized SDT to
    a D-SDT.

    Args:
        params: FrozenDict of SDT params
        is_discrete: True if the action space is discrete
        temperature: temperature used before choosing a feature to represent each node

    Returns:
        FrozenDict with discrete params
    """
    new_params = unfreeze(copy.deepcopy(params))
    sdt_params = new_params["params"]["sdt"]

    int_weights = sdt_params["internal"]["kernel"]
    int_bias = sdt_params["internal"]["bias"]
    
    # in the SDT, each internal node uses a weighted combination of each
    # of the observation dimensions
    # for the D-SDT, we only use the most influential feature
    chosen = jnp.argmax(int_weights, axis=0)
    one_hot_int = jax.nn.one_hot(chosen, num_classes=int_weights.shape[0]).T

    # since we are only using one weight, we need to preserve the original
    # decision boundary when adding the bias
    denom = int_weights[chosen, jnp.arange(int_weights.shape[1])]
    # safety check for dividing by very small weights
    denom = jnp.where(jnp.abs(denom) < 1e-8, 1.0, denom)

    norm_int_bias = int_bias / denom

    sdt_params["internal"]["kernel"] = one_hot_int
    sdt_params["internal"]["bias"] = norm_int_bias

    leaf_weights = sdt_params["leaves"]["kernel"]

    if is_discrete:
        # select the most probable action
        chosen_action = jnp.argmax(leaf_weights, axis=1)
        one_hot_leaf = jax.nn.one_hot(chosen_action, num_classes=leaf_weights.shape[1])
        sdt_params["leaves"]["kernel"] = one_hot_leaf

    else:
        # remove stochasticity
        sdt_params["log_std"]["kernel"] = jnp.zeros_like(sdt_params["log_std"]["kernel"])
        sdt_params["log_std"]["bias"] = jnp.zeros_like(sdt_params["log_std"]["bias"])

    return freeze(new_params)

def cohen_d(x, y):
    """Use x = SDT scores, y = D-SDT scores to match paper convention."""
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)

    pooled_std = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))

    if pooled_std < 1e-8:
        return np.nan

    return float((np.mean(x) - np.mean(y)) / pooled_std)

def evaluate_agent(actor_state, config, is_discrete, action_indices, is_final, seed=100):
    """NOTE: The only difference between SDT and D-SDT is this evaluation."""
    env = make_env(config["env_id"])

    def run_eval(eval_params, hard_tree=False):
        scores = []

        for ep_index in range(config["n_eval_episodes"]):
            obs, _ = env.reset(seed=seed + ep_index)
            done, trunc = False, False
            total_reward = 0.0

            while not (done or trunc):
                obs_batch = jnp.array([obs])

                pol = actor_state.apply_fn(
                    eval_params,
                    obs_batch,
                    max_path=hard_tree
                )

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

        return scores

    # evaluation
    if config["actor"] == 'sdt':
        scores = run_eval(actor_state.params, hard_tree=False)
    elif config["actor"] == 'd-sdt':
        dsdt_params = convert_to_discrete(actor_state.params, is_discrete)
        scores = run_eval(dsdt_params, hard_tree=True)

    # img_path = plot_dsdt_from_params(
    #     dsdt_params,
    #     config,
    #     out_path=f"dsdt_{step}"
    # )
    # wandb.log({"D-SDT": wandb.Image(img_path), "global_step": step})

    env.close()

    score_mean, score_std = float(np.mean(scores)), float(np.std(scores))

    if is_final:
        sdt_scores = run_eval(actor_state.params, hard_tree=False)
        d = cohen_d(sdt_scores, scores)
    else:
        d = np.nan

    return score_mean, score_std, d

def main(trial):
    config, train_config = setup()
    
    # fix issues with seeding
    seed = config["seed"] + (trial * 100)
    config["seed"] = seed
    wandb.config.update({"trial": trial, "seed": seed}, allow_val_change=True)
    
    raw_envs = [lambda: make_env(config['env_id']) for _ in range(config['n_envs'])]
    envs = gym.vector.AsyncVectorEnv(raw_envs)
    
    
    print(f'{envs.num_envs} environments were successfully created.')
    print(f'Obs. space: {envs.single_observation_space}')
    print(f'Action space: {envs.single_action_space}')
    
    action_dim, action_indices, is_discrete = get_action_mapping(config['env_id'], envs.single_action_space)
    
    print(f'Action space is disrete: {is_discrete}')
    print(f'Action mapping: {action_indices}')
    
    actor_state, critic_state = get_actor_and_critic_state(config, envs, action_dim, is_discrete)
    
    print(f"The {config['actor']} actor was created with arch.: {jax.tree_util.tree_map(lambda x: x.shape, actor_state.params)}")
    print(f"The {config['critic']} critic was created with arch.: {jax.tree_util.tree_map(lambda x: x.shape, critic_state.params)}")
    
    # prepare parameters for training loop
    next_obs, _ = envs.reset(seed=config['seed'])
    next_done = np.zeros(config['n_envs']).astype(bool)
    batch_size = int(config['n_envs'] * config['n_steps'])
    key = jax.random.PRNGKey(config["seed"])
    
    iteration = 1
    last_eval = 0
    global_step = 0
    # necessary to support arithmetic
    total_time = timedelta(0)
    
    avg_episodic_return_list = []
    eval, train = [], []
    
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(config['n_envs'], dtype=jnp.float32),
        episode_lengths=jnp.zeros(config['n_envs'], dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(config['n_envs'], dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(config['n_envs'], dtype=jnp.int32),
    )
    
    while global_step < config['total_steps']:
        start_time = datetime.now()
        print(f'Training iteration {iteration} started at {start_time}.')
        
        # storage for data in a single rollout
        action_dtype = jnp.int32 if is_discrete else jnp.float32
        storage = Storage(
                obs=jnp.zeros((config['n_steps'], config['n_envs']) + envs.single_observation_space.shape),
                actions=jnp.zeros((config['n_steps'], config['n_envs']) + envs.single_action_space.shape, dtype=action_dtype),
                log_probs=jnp.zeros((config['n_steps'], config['n_envs'])),
                dones=jnp.zeros((config['n_steps'], config['n_envs'])),
                values=jnp.zeros((config['n_steps'], config['n_envs'])),
                advantages=jnp.zeros((config['n_steps'], config['n_envs'])),
                returns=jnp.zeros((config['n_steps'], config['n_envs'])),
                rewards=jnp.zeros((config['n_steps'], config['n_envs'])),
        )
        
        episode_stats, next_obs, next_done, storage, key, global_step = rollout(
            config, actor_state, critic_state, action_indices, is_discrete, episode_stats, next_obs, next_done, storage, key, global_step, envs
        )
        
        print(f'rollout completed with reward mu: {float(jnp.mean(storage.rewards))}, \
              std: {float(jnp.std(storage.rewards))}, done count: {int(jnp.sum(storage.dones))}')
        
        storage = compute_gae(train_config, critic_state, next_obs, next_done, storage)
        
        print(f'gae: adv. mean={float(jnp.mean(storage.advantages))}, \
              std={float(jnp.std(storage.advantages))}, ret. mean={float(jnp.mean(storage.returns))}')
        
        actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            train_config,
            actor_state,
            critic_state,
            storage,
            key,
            is_discrete
        )
        
        print(f'PPO update: loss={float(jnp.mean(loss))}, pol_loss={float(jnp.mean(pg_loss))}, \
            v_loss={float(jnp.mean(v_loss))}, entropy={float(jnp.mean(entropy_loss))}, \
            approx_kl={float(jnp.mean(approx_kl))}')

        elapsed_time = datetime.now() - start_time
        total_time += elapsed_time

        avg_return = np.mean(np.array(episode_stats.returned_episode_returns))
        avg_episodic_return_list.append(avg_return)

        # evaluation "bucket"
        current_eval = global_step // config["eval_freq"]
        
        # determine evaluation criteria
        is_first = (iteration == 1)
        # entered a new evaluation "bucket"
        is_new = (current_eval > last_eval)
        is_final = (global_step + batch_size >= config["total_steps"])

        if is_first or is_new or is_final:
            last_eval = current_eval
            
            eval_score, eval_std, d = evaluate_agent(
                actor_state,
                config,
                is_discrete,
                action_indices,
                is_final
            )
                
            print(f"[eval] step={global_step} score={eval_score}")
            eval.append((global_step, eval_score))
            train.append((global_step, avg_return))
            
            wandb.log({
                f"test/{config['actor']}_avg_score": eval_score,
                "test/avg_std": eval_std,
                "global_step": global_step
            })
            
            if is_final:
                wandb.log({
                    "test/cohen_d": d
                })

        iteration += 1
        
        avg_episodic_return_100 = np.mean(avg_episodic_return_list[-100:])
        wandb.log({
            f"train/{config['actor']}_avg_episodic_return": avg_return,
            f"train/{config['actor']}_avg_episodic_return_100": avg_episodic_return_100,
            "global_step": global_step
        })
    
    return train, eval
        
if __name__ == '__main__':
    for trial in range(5):
        print(f"\n=== Starting trial {trial} ===\n")
        train, eval = main(trial)
        wandb.finish()
