import os 
import random
import datetime
import functools
import distrax
import graphviz
import wandb
import pickle
import jax
import optax
import optuna
import tyro
import flax
import gymnasium as gym
import numpy as np
import jax.numpy as jnp
import time


from dataclasses import dataclass, replace
from typing import Any, Literal
from distrax import Normal, MultivariateNormalDiag
from jax import lax
from flax import linen as nn
from functools import partial
from flax.training.train_state import TrainState
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip
from sklearn.tree import export_graphviz
from gymnasium.wrappers import FlattenObservation

from mlp import CriticMLP, DiscreteActorMLP, ContinuousActorMLP
from sadt import fit_state_action_dt
from utils import ActorTrainState, Storage, EpisodeStatistics, build_env, OBSERVATION_LABELS


# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
#os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
#os.environ["TF_CUDNN DETERMINISTIC"] = "1"

@dataclass
class Args:
    random_trials: int = 5
    """Number of random experiment runs """
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with wandb"""
    wandb_project_name: str = "mlp_sadt_sympol"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    render_env: bool = False
    """whether to render and save interpretables ike decision-tree plots"""
    render_each_eval: bool = False
    """whether to render artifacts at every evaluation instead of only the final one"""
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    actor: Literal["mlp", "stateActionDT"] = "mlp"
    """The type of actor to train."""
    total_timesteps: int = 1_000_000
    """Total training timesteps."""
    eval_freq: int = 50_000
    """Frequency of evaluation (in total timesteps)"""
    n_eval_episodes: int = 5
    """Number of episodes for evaluation"""
    sadt_max_depth: int = 4
    """Maximum depth for SA-DT distillation trees."""


# Values from configs.py in the SYMPOL repository (mlp best configs).
ENV_CONFIGS = {
    "CartPole-v1": {
        "learning_rate_actor": 0.00142,
        "learning_rate_critic": 0.002765,
        "n_envs": 13,
        "n_steps": 128,
        "minibatch_size": 256,
        "n_update_epochs": 7,
        "gamma": 0.999,
        "gae_lambda": 0.9,
        "ent_coef": 0.2,
        "vf_coef": 0.25,
        "max_grad_norm": 1.0,
        "norm_adv": 0,
        "clip_coef": 0.1,
        "num_layers": 2,
        "neurons_per_layer": 139,
    },
    "Acrobot-v1": {
        "learning_rate_actor": 0.0002193,
        "learning_rate_critic": 0.004594,
        "n_envs": 12,
        "n_steps": 512,
        "minibatch_size": 256,
        "n_update_epochs": 9,
        "gamma": 0.99,
        "gae_lambda": 0.9,
        "ent_coef": 0.0,
        "vf_coef": 0.50,
        "max_grad_norm": 1.0,
        "norm_adv": 1,
        "clip_coef": 0.1,
        "num_layers": 2,
        "neurons_per_layer": 185,
    },
    "LunarLander-v2": {
        "learning_rate_actor": 0.0005870,
        "learning_rate_critic": 0.0032635,
        "n_envs": 13,
        "n_steps": 512,
        "minibatch_size": 128,
        "n_update_epochs": 8,
        "gamma": 0.999,
        "gae_lambda": 0.9,
        "ent_coef": 0.1,
        "vf_coef": 0.50,
        "max_grad_norm": 0.5,
        "norm_adv": 0,
        "clip_coef": 0.1,
        "num_layers": 3,
        "neurons_per_layer": 46,
    },
    "Pendulum-v1": {
        "learning_rate_actor": 0.0003681,
        "learning_rate_critic": 0.001936,
        "n_envs": 8,
        "n_steps": 512,
        "minibatch_size": 128,
        "n_update_epochs": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.1,
        "vf_coef": 0.25,
        "max_grad_norm": 1000.0,
        "norm_adv": 1,
        "clip_coef": 0.1,
        "num_layers": 2,
        "neurons_per_layer": 75,
    },
    "MountainCarContinuous-v0": {
        "learning_rate_actor": 0.004786,
        "learning_rate_critic": 0.001164,
        "n_envs": 15,
        "n_steps": 512,
        "minibatch_size": 512,
        "n_update_epochs": 2,
        "gamma": 0.999,
        "gae_lambda": 0.95,
        "ent_coef": 0.1,
        "vf_coef": 0.25,
        "max_grad_norm": 0.1,
        "norm_adv": 1,
        "clip_coef": 0.1,
        "num_layers": 2,
        "neurons_per_layer": 240,
    },
    "MiniGrid-DoorKey-5x5-v0": {
        "learning_rate_actor": 0.0004126,
        "learning_rate_critic": 0.0004508,
        "n_envs": 8,
        "n_steps": 256,
        "minibatch_size": 256,
        "n_update_epochs": 7,
        "gamma": 0.9,
        "gae_lambda": 0.9,
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "max_grad_norm": 0.1,
        "norm_adv": 1,
        "clip_coef": 0.1,
        "num_layers": 1,
        "neurons_per_layer": 169,
    },
    "MiniGrid-LavaGapS5-v0": {
        "learning_rate_actor": 0.001808,
        "learning_rate_critic": 0.00304,
        "n_envs": 8,
        "n_steps": 512,
        "minibatch_size": 128,
        "n_update_epochs": 9,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.1,
        "vf_coef": 0.25,
        "max_grad_norm": 1.0,
        "norm_adv": 0,
        "clip_coef": 0.1,
        "num_layers": 1,
        "neurons_per_layer": 76,
    },
    "MiniGrid-Empty-Random-6x6-v0": {
        "learning_rate_actor": 0.00022006113628729703,
        "learning_rate_critic": 0.001016745520591446,
        "n_envs": 13,
        "n_steps": 512,
        "minibatch_size": 64,
        "n_update_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "max_grad_norm": 0.1,
        "norm_adv": 0,
        "clip_coef": 0.1,
        "num_layers": 3,
        "neurons_per_layer": 112,
    },
    "MiniGrid-LavaGapS7-v0": {
        "learning_rate_actor": 0.00030421380932650844,
        "learning_rate_critic": 0.0005672446115512332,
        "n_envs": 12,
        "n_steps": 128,
        "minibatch_size": 512,
        "n_update_epochs": 8,
        "gamma": 0.9,
        "gae_lambda": 0.95,
        "ent_coef": 0.1,
        "vf_coef": 0.75,
        "max_grad_norm": 0.5,
        "norm_adv": 1,
        "clip_coef": 0.1,
        "num_layers": 1,
        "neurons_per_layer": 28,
    },
    "MiniGrid-DistShift1-v0": {
        "learning_rate_actor": 0.0002579884429826318,
        "learning_rate_critic": 0.000950682421442667,
        "n_envs": 10,
        "n_steps": 128,
        "minibatch_size": 256,
        "n_update_epochs": 7,
        "gamma": 0.99,
        "gae_lambda": 0.99,
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "max_grad_norm": 0.1,
        "norm_adv": 1,
        "clip_coef": 0.1,
        "num_layers": 2,
        "neurons_per_layer": 158,
    },
    "MountainCar-v0": {
        "learning_rate_actor": 0.0001266,
        "learning_rate_critic": 0.007337,
        "n_envs": 10,
        "n_steps": 128,
        "minibatch_size": 512,
        "n_update_epochs": 6,
        "gamma": 0.999,
        "gae_lambda": 0.9,
        "ent_coef": 0.1,
        "vf_coef": 0.25,
        "max_grad_norm": 1000.0,
        "norm_adv": 1,
        "clip_coef": 0.1,
        "num_layers": 3,
        "neurons_per_layer": 144,
    },
}

def apply_env_defaults(args: Args) -> Args:
    if args.env_id in ENV_CONFIGS and args.actor in ["mlp", "stateActionDT"]:
        config = ENV_CONFIGS[args.env_id]
        for key, val in config.items():
            setattr(args, key, val)

    return args


def evaluate_mlp(env_id, actor, actor_params, n_episodes, is_discrete, seed=100,
                 render_env=False, render_now=False, capture_video=False, track=False):
    """Evaluate a MLP actor."""
    scores = []
    name_appendix = "_mlp"
    for ep in range(n_episodes):
        env = build_env(env_id, n_env=1)
        obs, _ = env.reset(seed=seed + ep)
        done, trunc = False, False
        total_reward = 0.0
        step_counter = 0
        frames = []
        while not done and not trunc:
            # The rendering functions/code is taken from the original paper and utils.py. See utils.py for detailed explanation
            if render_env and render_now:
                if capture_video:
                    frame = env.render()
                    image = Image.fromarray(frame)
                    draw = ImageDraw.Draw(image)
                    text_step = f'Step: {step_counter}'
                    font_size = frame.shape[0] // 20
                    draw.text((font_size, font_size*0.5), text_step, (200, 200, 200), font=ImageFont.load_default())
                    text_reward = f'Reward: {total_reward}'
                    draw.text((font_size, font_size*2.0), text_reward, (200, 200, 200), font=ImageFont.load_default())
                    frames.append(np.array(image))

            # discrete mlp case
            if is_discrete:
                logits = actor.apply(actor_params, np.array([obs]))
                action = int(jnp.argmax(logits, axis=-1)[0])
            else:
            # continuous mlp case
                mean, log_std = actor.apply(actor_params, np.array([obs]))
                action = np.array(mean[0])
            obs, reward, done, trunc, _ = env.step(action)
            total_reward += reward
            step_counter += 1

        if render_env and render_now and capture_video and frames:
            numpy_clip = np.transpose(np.array(frames), (0, 3, 1, 2)) 
            fps = 5 if 'MiniGrid' in env_id else 25
            if track:
                wandb.log({"gameplay" + name_appendix + '_trial_ep' + str(ep): wandb.Video(numpy_clip, fps=fps, format="mp4")}, commit=False)

        env.close()
        scores.append(total_reward)
    return scores


def evaluate_sadt(env_id, decision_tree, n_episodes, is_discrete, action_dim, seed=100,
                  render_env=False, render_now=False, capture_video=False, track=False):
    """Evaluates a fitted SA-DT decision tree."""
    scores = []
    name_appendix = "_sadt"
    for ep in range(n_episodes):
        env = build_env(env_id, n_env=1)
        obs, _ = env.reset(seed=seed + ep)
        done, trunc = False, False
        total_reward = 0.0
        step_counter = 0
        frames = []
        while not done and not trunc:
            if render_env and render_now:
                if capture_video:
                    frame = env.render()
                    image = Image.fromarray(frame)
                    draw = ImageDraw.Draw(image)
                    text_step = f'Step: {step_counter}'
                    font_size = frame.shape[0] // 20
                    draw.text((font_size, font_size*0.5), text_step, (200, 200, 200), font=ImageFont.load_default())
                    text_reward = f'Reward: {total_reward}'
                    draw.text((font_size, font_size*2.0), text_reward, (200, 200, 200), font=ImageFont.load_default())
                    frames.append(np.array(image))

            flat_obs = obs.reshape(1, -1)
            if is_discrete:
                action = decision_tree.predict(flat_obs)[0]
            elif action_dim == 1:
                action = decision_tree.predict(flat_obs)
            else:
                action = np.array([decision_tree[i].predict(flat_obs)[0] for i in range(action_dim)])
            obs, reward, done, trunc, _ = env.step(action)
            total_reward += reward
            step_counter += 1
            
        if render_env and render_now and capture_video and frames:
            numpy_clip = np.transpose(np.array(frames), (0, 3, 1, 2)) 
            fps = 5 if 'MiniGrid' in env_id else 25
            if track:
                wandb.log({"gameplay" + name_appendix + '_trial_ep' + str(ep): wandb.Video(numpy_clip, fps=fps, format="mp4")}, commit=False)

        env.close()
        scores.append(total_reward)
    return scores


def plot_state_action_dt(decision_tree, env_id, run_name, is_discrete, action_dim, track):
    """Save SA-DT plots"""
    video_folder = "videos/wandb"
    os.makedirs(video_folder, exist_ok=True)
    image_name = run_name + "-" + "-" + env_id
    for char in '<>:"/\\|?*':
        image_name = image_name.replace(char, '_')
    image_path = os.path.join(video_folder, image_name)
    obs_labels = OBSERVATION_LABELS.get(env_id, None)
    wrapped_obs_labels = None
    # i made the changes for these in order to actually make the tree readable, otherwise it becomes extremely hard to read
    # as the tree isn't pruned so each layer is full. this makes it so the text becomes unreadable when the depth gets far enough
    # (eg. 8)
    if obs_labels is not None:
        wrapped_obs_labels = [label.replace("angular_velocity", "angular\\nvelocity") for label in obs_labels]
    node_count = 0

    def render_tree_png(tree, plot_filename):
        dot = export_graphviz(
            tree,
            out_file=None,
            filled=False,
            rounded=False,
            impurity=False,
            proportion=False,
            feature_names=wrapped_obs_labels,
        )

        cleaned_lines = []
        for line in dot.splitlines():
            if '[label="' in line:
                prefix, label_suffix = line.split('label="', 1)
                label_text, suffix = label_suffix.split('"]', 1)
                label_parts = [part for part in label_text.split('\\n') if not part.startswith('samples = ')]
                line = prefix + 'label="' + '\\n'.join(label_parts) + '"]' + suffix
            cleaned_lines.append(line)

        dot = "\n".join(cleaned_lines)
        dot = dot.replace(
            'digraph Tree {',
            'digraph Tree {\ngraph [dpi="150", pad="0.2", nodesep="0.2", ranksep="0.3"] ;',
            1,
        )
        dot = dot.replace(
            'node [shape=box, color="black", fontname="helvetica"] ;',
            'node [shape=box, color="black", fontname="Helvetica", fontsize=24, margin="0.15,0.12"] ;',
            1,
        )
        dot = dot.replace(
            'edge [fontname="helvetica"] ;',
            'edge [fontname="Helvetica", fontsize=24] ;',
            1,
        )

        rendered_png = graphviz.Source(dot).unflatten(stagger=4).pipe(format="png")
        with open(plot_filename, "wb") as output_file:
            output_file.write(rendered_png)

    # cannot use utils.py, that structure for plotting trees is not for sk.learn DecisionTreeClassifier/Regressor, so need to use custom.
    if is_discrete or action_dim == 1:
        trees = [(None, decision_tree)]
    else:
        trees = [(index, decision_tree[index]) for index in range(action_dim)]

    for index, tree in trees:
        if index is None:
            plot_filename = image_path + "state_action_DT.png"
        else:
            plot_filename = image_path + "state_action_DT_reg" + str(index) + ".png"

        render_tree_png(tree, plot_filename)
        tree_node_count = tree.tree_.node_count
        node_count += tree_node_count

        if index is None:
            print(
                f"Tree saved to {plot_filename} "
                f"(nodes: {tree_node_count})"
            )
            if track:
                wandb.log({"state_action_DT": wandb.Image(plot_filename)})
        else:
            print(
                f"Tree {index} saved to {plot_filename} "
                f"(nodes: {tree_node_count})"
            )
            if track:
                wandb.log({"state_action_DT_" + str(index): wandb.Image(plot_filename)})

    if len(trees) > 1:
        print(f"Trees saved to {video_folder}/ (total nodes: {node_count})")

    return node_count

# took scaffolding/ppo steps from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py
# steps were taken from this repo but modified to match sympol's sa-dt and mlp actions
def run_trial(args: Args, random_trial: int = 1):
    # Derived values
    batch_size = int(args.n_envs * args.n_steps)
    minibatch_size = args.minibatch_size
    while batch_size // minibatch_size < 2:
        minibatch_size = minibatch_size // 2

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    env_seed = args.seed + (random_trial * 100)
    seed_training = env_seed
    key = jax.random.PRNGKey(seed_training)
    model_key = jax.random.PRNGKey(args.seed)
    model_key, actor_key, critic_key = jax.random.split(model_key, 3)

    # environment setup, used from util.py from sympol repo
    envs = build_env(args.env_id, n_env=args.n_envs)
    is_discrete = isinstance(envs.single_action_space, gym.spaces.Discrete)
    action_type = "discrete" if is_discrete else "continuous"
    if is_discrete:
        action_dim = envs.single_action_space.n
    else:
        action_dim = envs.single_action_space.shape[-1]

    print(f"Env: {args.env_id} | Obs: {envs.single_observation_space.shape} | "
          f"Actions: {action_dim} ({action_type}) | Actor: {args.actor}")

    # agent setup
    critic = CriticMLP(
        hidden_layers=args.num_layers,
        hidden_size=args.neurons_per_layer,
    )
    if is_discrete:
        actor = DiscreteActorMLP(
            out_features=action_dim,
            hidden_layers=args.num_layers,
            hidden_size=args.neurons_per_layer,
        )
    else:
        actor = ContinuousActorMLP(
            out_features=action_dim,
            hidden_layers=args.num_layers,
            hidden_size=args.neurons_per_layer,
        )

    sample_obs = jnp.array([envs.single_observation_space.sample()])

    actor_params = actor.init(actor_key, sample_obs)
    critic_params = critic.init(critic_key, sample_obs)

    # separate critic optimizer
    critic_state = TrainState.create(
        apply_fn=None,
        params=critic_params,
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=args.learning_rate_critic),
        ),
    )
    # separate actor optimizer
    actor_state = ActorTrainState.create(
        apply_fn=None,
        params=actor_params,
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(args.learning_rate_actor),
        ),
        grad_accum=jax.tree.map(jnp.zeros_like, actor_params),
        indices=None,
    )

    accumulate_gradients_every = 1

    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    # episode statistics tracker
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
    )

    @jax.jit
    def get_action_and_value(
        actor_state: ActorTrainState,
        critic_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        step: int,
        key: jax.random.PRNGKey,
    ):
        """Sample action, calculate value, logprob, and update storage."""
        if is_discrete:
            logits = actor.apply(actor_state.params, next_obs)
            action_dist = distrax.Categorical(logits=logits)
        else:
            mean, log_std = actor.apply(actor_state.params, next_obs)
            action_dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))

        value = critic.apply(critic_state.params, next_obs)
        key, subkey = jax.random.split(key)
        action = action_dist.sample(seed=subkey)
        logprob = action_dist.log_prob(action)

        storage = storage.replace(
            obs=storage.obs.at[step].set(next_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(action),
            logprobs=storage.logprobs.at[step].set(logprob),
            values=storage.values.at[step].set(value.squeeze()),
        )
        return storage, action, key

    @jax.jit
    def get_action_and_value2(
        actor_params: flax.core.FrozenDict,
        critic_params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """Calculate value, logprob of supplied action, and entropy."""
        if is_discrete:
            logits = actor.apply(actor_params, x)
            action_dist = distrax.Categorical(logits=logits)
        else:
            mean, log_std = actor.apply(actor_params, x)
            action_dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))

        value = critic.apply(critic_params, x).squeeze()
        logprob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return logprob, entropy, value

    # GAE with jax.lax.scan
    # https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
    def compute_gae_step(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone
        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_step = partial(compute_gae_step, gamma=args.gamma, gae_lambda=args.gae_lambda)
    
    '''
    Calculate the GAE estimate from the original SYMPOL paper.
    Scaffold from original paper's computation of the GAE estimate sicne they use a JAX optimization technique that is not specified in the paper.
    Does the same thing in hw2.py
    '''
    @jax.jit
    def compute_gae(
        critic_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        next_value = critic.apply(critic_state.params, next_obs).squeeze()
        advantages = jnp.zeros((args.n_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_step,
            advantages,
            (dones[1:], values[1:], values[:-1], storage.rewards),
            reverse=True,
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    # PPO loss (gradients w.r.t. actor and critic separately)
    @jax.jit
    def ppo_loss(actor_params, critic_params, x, a, logp, mb_advantages, mb_returns):
        newlogprob, entropy, newvalue = get_action_and_value2(
            actor_params, critic_params, x, a
        )
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8
            )

        # policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(
            ratio, 1 - args.clip_coef, 1 + args.clip_coef
        )
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, argnums=(0, 1), has_aux=True)

    # PPO update with double jax.lax.scan
    # the PPO loop. Directly reimplement because the authors do not 
    # explain or specify their optimizations using jax.lax.scan() in the paper
    # which affects results.
    @jax.jit
    def update_ppo(
        actor_state: ActorTrainState,
        critic_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, _):
            actor_state, critic_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])
                
            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                num_minibatches = int(np.floor(x.shape[0] / minibatch_size))
                size = num_minibatches * minibatch_size
                x = jax.random.permutation(subkey, x)[:size]
                x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
                return x   
            
            # https://docs.jax.dev/en/latest/_autosummary/jax.tree.map.html#jax.tree.map
            flatten_storage = jax.tree_map(flatten, storage)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

            def update_minibatch(carry, minibatch):
                actor_state, critic_state = carry
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), (
                    actor_grads,
                    critic_grads,
                ) = ppo_loss_grad_fn(
                    actor_state.params,
                    critic_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                )
                critic_state = critic_state.apply_gradients(grads=critic_grads)

                actor_grad_accum = jax.tree_util.tree_map(
                    lambda grad, accum: grad + accum,
                    actor_grads,
                    actor_state.grad_accum,
                )
                actor_state = actor_state.apply_gradients(grads=actor_grads)

                def update_fn():
                    grads = jax.tree_util.tree_map(
                        lambda grad: grad / accumulate_gradients_every,
                        actor_grad_accum,
                    )
                    return actor_state.apply_gradients(
                        grads=grads,
                        grad_accum=jax.tree_util.tree_map(jnp.zeros_like, grads),
                    )

                actor_state = jax.lax.cond(
                    actor_state.step % accumulate_gradients_every == 0,
                    lambda _: update_fn(),
                    lambda _: actor_state.replace(
                        grad_accum=actor_grad_accum,
                        step=actor_state.step + 1,
                    ),
                    None,
                )

                return (actor_state, critic_state), (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    approx_kl,
                )

            (actor_state, critic_state), (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
            ) = jax.lax.scan(
                update_minibatch, (actor_state, critic_state), shuffled_storage
            )
            return (actor_state, critic_state, key), (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
            )

        (actor_state, critic_state, key), (
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
        ) = jax.lax.scan(
            update_epoch,
            (actor_state, critic_state, key),
            (),
            length=args.n_update_epochs,
        )
        return (
            actor_state,
            critic_state,
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            key,
        )

    # Training Loop
    global_step = 0
    next_obs, _ = envs.reset(seed=env_seed)
    next_done = np.zeros(args.n_envs, dtype=bool)
    start_time = time.time()
    iteration = 1
    last_eval = 0
    avg_score_list = []
    n_steps = args.n_steps
    plotted_state_action_dt = False

    # History tracking
    history_train_return = []
    final_mlp_scores = None

    while global_step < args.total_timesteps:
        # Storage for this rollout
        action_shape = envs.single_action_space.shape
        storage = Storage(
            obs=jnp.zeros((n_steps, args.n_envs) + envs.single_observation_space.shape),
            actions=jnp.zeros((n_steps, args.n_envs) + action_shape, dtype=jnp.int32),
            logprobs=jnp.zeros((n_steps, args.n_envs)),
            dones=jnp.zeros((n_steps, args.n_envs)),
            values=jnp.zeros((n_steps, args.n_envs)),
            advantages=jnp.zeros((n_steps, args.n_envs)),
            returns=jnp.zeros((n_steps, args.n_envs)),
            rewards=jnp.zeros((n_steps, args.n_envs)),
        )

        # Rollout
        for step in range(n_steps):
            global_step += args.n_envs
            storage, action, key = get_action_and_value(
                actor_state, critic_state, next_obs, next_done, storage, step, key
            )
            action_np = np.array(action)
            next_obs, reward, next_done, trunc, info = envs.step(action_np)

            # Track episode statistics
            new_episode_return = episode_stats.episode_returns + reward
            new_episode_length = episode_stats.episode_lengths + 1
            episode_stats = EpisodeStatistics(
                episode_returns=(new_episode_return) * (1 - next_done) * (1 - trunc),
                episode_lengths=(new_episode_length) * (1 - next_done) * (1 - trunc),
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

        # GAE
        storage = compute_gae(critic_state, next_obs, next_done, storage)

        # PPO Update
        actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = (
            update_ppo(actor_state, critic_state, storage, key)
        )

        # Logging
        avg_episodic_return = np.mean(
            np.array(episode_stats.returned_episode_returns)
        )
        current_eval = global_step // args.eval_freq

        if iteration == 1 or current_eval > last_eval or global_step + batch_size >= args.total_timesteps:
            last_eval = current_eval
            elapsed = time.time() - start_time
            render_now = True if args.render_each_eval else True if global_step + batch_size >= args.total_timesteps else False

            # Evaluate MLP actor (train curve)
            mlp_scores = evaluate_mlp(
                args.env_id, actor, actor_state.params,
                args.n_eval_episodes, is_discrete, seed=env_seed,
                render_env=args.render_env, render_now=render_now,
                capture_video=args.capture_video, track=args.track
            )
            avg_mlp = np.mean(mlp_scores)
            std_mlp = np.std(mlp_scores)

            history_train_return.append(avg_episodic_return)

            # Compute avg_episodic_return_100 for wandb logging
            avg_episodic_return_100 = np.mean(
                np.array(history_train_return[-100:])
            ) if len(history_train_return) >= 1 else avg_episodic_return

            # Evaluate SA-DT if requested
            if args.actor == "stateActionDT":
                eval_env = build_env(args.env_id, n_env=1)

                dt = fit_state_action_dt(
                    eval_env,
                    actor.apply,
                    actor_state.params,
                    max_depth=args.sadt_max_depth,
                    n_episodes=25,
                    action_type=action_type,
                    action_dim=action_dim,
                    seed=args.seed,
                )
                eval_env.close()

                sadt_scores = evaluate_sadt(
                    args.env_id, dt, args.n_eval_episodes,
                    is_discrete, action_dim, seed=env_seed,
                    render_env=args.render_env, render_now=render_now,
                    capture_video=args.capture_video, track=args.track
                )
                avg_sadt = np.mean(sadt_scores)
                std_sadt = np.std(sadt_scores)

                if args.render_env and render_now:
                    plot_state_action_dt(
                        dt,
                        args.env_id,
                        run_name,
                        is_discrete,
                        action_dim,
                        args.track,
                    )
                    plotted_state_action_dt = True

                print(
                    f"step={global_step:>8d} | "
                    f"mlp={avg_mlp:>8.2f}±{std_mlp:.2f} | "
                    f"sa-dt={avg_sadt:>8.2f}±{std_sadt:.2f} | "
                    f"train_return={avg_episodic_return:>8.2f} | "
                    f"time={elapsed:.1f}s"
                )
                avg_score_list.append(avg_sadt)

                if args.track:
                    wandb.log({
                        "global_step": global_step,
                        "charts/avg_score_mlp": avg_mlp,
                        "charts/std_score_mlp": std_mlp,
                        "charts/avg_score_sadt": avg_sadt,
                        "charts/std_score_sadt": std_sadt,
                        "charts/avg_episodic_return_mlp": avg_episodic_return,
                        "charts/avg_episodic_return_100_mlp": avg_episodic_return_100,
                    })
            else:
                print(
                    f"step={global_step:>8d} | "
                    f"eval={avg_mlp:>8.2f}±{std_mlp:.2f} | "
                    f"train_return={avg_episodic_return:>8.2f} | "
                    f"time={elapsed:.1f}s"
                )
                avg_score_list.append(avg_mlp)

                if args.track:
                    wandb.log({
                        "global_step": global_step,
                        "charts/avg_score_mlp": avg_mlp,
                        "charts/std_score_mlp": std_mlp,
                        "charts/avg_episodic_return_mlp": avg_episodic_return,
                        "charts/avg_episodic_return_100_mlp": avg_episodic_return_100,
                    })
            # final test step like in original repo for whatever reason
            if global_step + batch_size >= args.total_timesteps:
                test_seed = 123456
                final_mlp_scores = evaluate_mlp(
                    args.env_id, actor, actor_state.params,
                    args.n_eval_episodes, is_discrete, seed=test_seed,
                    render_env=args.render_env, render_now=True,
                    capture_video=args.capture_video, track=args.track
                )
                avg_mlp_test = np.mean(final_mlp_scores)
                std_mlp_test = np.std(final_mlp_scores)

                print(
                    f"final_mlp_test={avg_mlp_test:>8.2f}±{std_mlp_test:.2f}"
                )

                if args.track:
                    wandb.log({
                        "charts/global_step": global_step,
                        "charts/mlp_test_mean_mlp": avg_mlp_test,
                        "charts/mlp_test_std_mlp": std_mlp_test,
                    })

        # Log losses every iteration
        if args.track:
            try:
                wandb.log({
                    "losses/value_loss": np.mean(v_loss[-1]),
                    "losses/policy_loss": np.mean(pg_loss[-1]),
                    "losses/entropy": np.mean(entropy_loss[-1]),
                    "losses/approx_kl": np.mean(approx_kl[-1]),
                    "losses/loss": np.mean(loss[-1]),
                    "charts/global_step": global_step,
                })
            except Exception:
                pass

        iteration += 1

    # test evaluation
    if final_mlp_scores is None:
        test_seed = 123456
        final_mlp_scores = evaluate_mlp(
            args.env_id, actor, actor_state.params,
            args.n_eval_episodes, is_discrete, seed=test_seed,
            render_env=args.render_env, render_now=True,
            capture_video=args.capture_video, track=args.track
        )
    print(f"\nFinal MLP Test Score: {np.mean(final_mlp_scores):.2f} "
          f"± {np.std(final_mlp_scores):.2f}")

    if args.actor == "stateActionDT":
        eval_env = build_env(args.env_id, n_env=1)

        dt_final = fit_state_action_dt(
            eval_env, actor.apply, actor_state.params,
            max_depth=args.sadt_max_depth, n_episodes=25,
            action_type=action_type, action_dim=action_dim, seed=args.seed,
        )
        eval_env.close()

        final_sadt_scores = evaluate_sadt(
            args.env_id, dt_final, args.n_eval_episodes,
            is_discrete, action_dim, seed=test_seed,
            render_env=args.render_env, render_now=True,
            capture_video=args.capture_video, track=args.track
        )
        print(f"Final SA-DT Test Score: {np.mean(final_sadt_scores):.2f} "
              f"± {np.std(final_sadt_scores):.2f}")

        if args.render_env and not plotted_state_action_dt:
            plot_state_action_dt(
                dt_final,
                args.env_id,
                run_name,
                is_discrete,
                action_dim,
                args.track,
            )

    if args.track:
        wandb.finish()
    envs.close()
    print("Done.")


if __name__ == "__main__":
    base_args = tyro.cli(Args)

    for trial_idx in range(base_args.random_trials):
        random_trial = trial_idx + 1
        trial_args = apply_env_defaults(replace(base_args))
        if base_args.random_trials > 1:
            print(
                f"\n=== Trial {trial_idx + 1}/{base_args.random_trials} "
                f"(seed={trial_args.seed}, env_seed={trial_args.seed + (random_trial * 100)}) ==="
            )
        run_trial(trial_args, random_trial=random_trial)
