'''
This code is adapted from the template of clean-rl's implementation of PPO and extends SYMPOL's implementation to fit our replication
The original code template can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy

The PPO advantage calculation, rollout helper function, and plotting script are taken directly from SYMPOL.
This is because the authors use specific JAX optimization techniques (e.g. jax.lax.scan) or other technical details that are not specified in the paper
and are largely different from clean-rl's PPO. Moreover, we use the same Storage and TrainState classes to ensure that our model 
and results are saved the same way to properly reproduce the plots. We note and explain these functions in our writeup and the comments below.
The original code can be found at https://github.com/s-marton/SYMPOL/blob/master/sympol.py 

Importantly, our contributions consist of the model definitions to fit our replicated model classes, fixing several issues in the original training script 
(e.g. unused values, invalid typing, improper updates; detailed in our writeup), and augmenting the ppo training logic to fit our replication experiments.

Note: If encountering issues with gymnasium[box2d] for LunarLander, try ```pip install box2d pygame```
'''

import os
import time
from dataclasses import dataclass
import tyro # Using Tyro, like the original codebase, for ease of implementation

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax.training.train_state import TrainState
import distrax
import optax
from optax_swag import swag

from PIL import Image, ImageDraw, ImageFont
import wandb
from utils import (
    build_env, # Creates our (sometimes parallel) environments with the correct wrappers for each corresponding environment
    ActorTrainState, # Defines separate TrainState for the SYMPOL actor as it requires us to track the node indices as a model parameter
    EpisodeStatistics, # Defines the training statistics to track episode returns/lengths to send to Wandb
    Storage, # Defines the training statistics to track episode obs/actions/etc. to send to Wandb
    plot_decision_tree, # Plots pruned/un-pruned decision trees for our Actor model using Graphviz backend and tree distillation logic
    OBSERVATION_LABELS # Hard-coded observation labels to help with decision tree interpretability
)

from sympol import SYMPOL
from mlp import CriticMLP

# We hardcode the optimal hyperparameters for each environment based on the original paper's experiments (from configs.py)
# Note that we can adjust these hyperparameters anytime using optional arguments
@dataclass
class Args:
    experiment_name: str = "SYMPOL"

    seed: int = 42
    total_steps: int = 1000000
    gpu_number: int = 0
    random_trials: int = 5
    n_eval_episodes: int = 5
    eval_freq: int = 50000
    normEnv: bool = True
    max_grad_norm: float = 1000
    clip_coef: float = 0.1
    render_env: bool = True
    render_each_eval: bool = True
    render_video: bool = False
    view_size: int = 3

    # The following hyperparameters are taken directly from the SYMPOL paper
    env_id: str = "CartPole-v1"
    ent_coef: float = 0.200
    gae_lambda: float = 0.950
    gamma: float = 0.990
    learning_rate_actor_split_values: float = 0.000222274485191996
    learning_rate_actor_split_idx_array: float = 0.025528008432059508
    learning_rate_actor_leaf_array: float = 0.019530943718321373
    learning_rate_actor_log_std: float = 0.0012313062437960766
    learning_rate_critic: float = 0.0013329992676131342
    n_envs: int = 7
    n_steps: int = 512
    n_update_epochs: int = 7
    norm_adv: bool = False
    reduce_lr: bool = True
    vf_coef: float = 0.500
    dropout: float = 0.000
    depth: int = 7
    minibatch_size: int = 64


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

args = tyro.cli(Args)

if args.env_id == "MiniGrid-Empty-Random-6x6-v0":
    args.ent_coef = 0.100
    args.gae_lambda = 0.990
    args.gamma = 0.900
    args.learning_rate_actor_split_values = 0.0009245327872865724
    args.learning_rate_actor_split_idx_array = 0.0006304100125886239
    args.learning_rate_actor_leaf_array = 0.002646152248961896
    args.learning_rate_actor_log_std = 0.04341143118701816
    args.learning_rate_critic = 0.0006610280521337505
    args.n_envs = 14
    args.n_steps = 128
    args.n_update_epochs = 8
    args.norm_adv = True
    args.reduce_lr = False
    args.vf_coef = 0.500
    args.dropout = 0.000
    args.depth = 7
    args.minibatch_size = 64

elif args.env_id == "MiniGrid-DoorKey-5x5-v0":
    args.ent_coef = 0.200
    args.gae_lambda = 0.950
    args.gamma = 0.990
    args.learning_rate_actor_split_values = 0.0012450034110784152
    args.learning_rate_actor_split_idx_array = 0.0005029536099891734
    args.learning_rate_actor_leaf_array = 0.0035778989299146166
    args.learning_rate_actor_log_std = 0.02121882708532112
    args.learning_rate_critic = 0.0008539673613239264
    args.n_envs = 14
    args.n_steps = 512
    args.n_update_epochs = 9
    args.norm_adv = True
    args.reduce_lr = True
    args.vf_coef = 0.500
    args.dropout = 0.000
    args.depth = 7
    args.minibatch_size = 64

elif args.env_id == "MiniGrid-LavaGapS5-v0":
    args.ent_coef = 0.100
    args.gae_lambda = 0.900
    args.gamma = 0.950
    args.learning_rate_actor_split_values = 0.005811380648459824
    args.learning_rate_actor_split_idx_array = 0.01225369992015828
    args.learning_rate_actor_leaf_array = 0.008676695448646759
    args.learning_rate_actor_log_std = 0.004742570909023367
    args.learning_rate_critic = 0.0006092740766519476
    args.n_envs = 16
    args.n_steps = 512
    args.n_update_epochs = 5
    args.norm_adv = True
    args.reduce_lr = True
    args.vf_coef = 0.250
    args.dropout = 0.000
    args.depth = 7
    args.minibatch_size = 64

elif args.env_id == "MiniGrid-LavaGapS7-v0":
    args.ent_coef = 0.100
    args.gae_lambda = 0.900
    args.gamma = 0.990
    args.learning_rate_actor_split_values = 0.0005838223729862216
    args.learning_rate_actor_split_idx_array = 0.0006590714932633344
    args.learning_rate_actor_leaf_array = 0.007946523254059177
    args.learning_rate_actor_log_std = 0.002205830616639246
    args.learning_rate_critic = 0.001127757835458702
    args.n_envs = 7
    args.n_steps = 128
    args.n_update_epochs = 4
    args.norm_adv = True
    args.reduce_lr = True
    args.vf_coef = 0.500
    args.dropout = 0.000
    args.depth = 7
    args.minibatch_size = 64

elif args.env_id == "MiniGrid-DistShift1-v0":
    args.ent_coef = 0.500
    args.gae_lambda = 0.950
    args.gamma = 0.999
    args.learning_rate_actor_split_values = 0.0002680425031090237
    args.learning_rate_actor_split_idx_array = 0.008701058712472901
    args.learning_rate_actor_leaf_array = 0.0005740321057491008
    args.learning_rate_actor_log_std = 0.03767558661659253
    args.learning_rate_critic = 0.0009300326937305064
    args.n_envs = 10
    args.n_steps = 512
    args.n_update_epochs = 5
    args.norm_adv = False
    args.reduce_lr = True
    args.vf_coef = 0.250
    args.dropout = 0.000
    args.depth = 7
    args.minibatch_size = 64

elif args.env_id == "Acrobot-v1":
    args.ent_coef = 0.000
    args.gae_lambda = 0.950
    args.gamma = 0.990
    args.learning_rate_actor_split_values = 0.00020085566411900057
    args.learning_rate_actor_split_idx_array = 0.05198040198477529
    args.learning_rate_actor_leaf_array = 0.005371878728382642
    args.learning_rate_actor_log_std = 0.0019814944246277504
    args.learning_rate_critic = 0.0003547997953897775
    args.n_envs = 8
    args.n_steps = 128
    args.n_update_epochs = 7
    args.norm_adv = False
    args.reduce_lr = True
    args.vf_coef = 0.250
    args.dropout = 0.000
    args.depth = 7
    args.minibatch_size = 64

elif args.env_id == "LunarLander-v2":
    args.ent_coef = 0.000
    args.gae_lambda = 0.900
    args.gamma = 0.999
    args.learning_rate_actor_split_values = 0.0006591868973696417
    args.learning_rate_actor_split_idx_array = 0.009966850522393832
    args.learning_rate_actor_leaf_array = 0.008588600717840487
    args.learning_rate_actor_log_std = 0.02140711489067244
    args.learning_rate_critic = 0.001771755240346081
    args.n_envs = 6
    args.n_steps = 512
    args.n_update_epochs = 7
    args.norm_adv = True
    args.reduce_lr = True
    args.vf_coef = 0.500
    args.dropout = 0.000
    args.depth = 7
    args.minibatch_size = 64

elif args.env_id == "MountainCarContinuous-v0":
    args.ent_coef = 0.500
    args.gae_lambda = 0.990
    args.gamma = 0.999
    args.learning_rate_actor_split_values = 0.0001160748504514767
    args.learning_rate_actor_split_idx_array = 0.0001015527526014825
    args.learning_rate_actor_leaf_array = 0.028465599628829257
    args.learning_rate_actor_log_std = 0.0942996760712889
    args.learning_rate_critic = 0.0020613382527496695
    args.n_envs = 5
    args.n_steps = 128
    args.n_update_epochs = 2
    args.norm_adv = False
    args.reduce_lr = True
    args.vf_coef = 0.500
    args.dropout = 0.000
    args.depth = 7
    args.minibatch_size = 64

elif args.env_id == "Pendulum-v1":
    args.ent_coef = 0.100
    args.gae_lambda = 0.800
    args.gamma = 0.999
    args.learning_rate_actor_split_values = 0.0002307526719494789
    args.learning_rate_actor_split_idx_array = 0.009862044169880627
    args.learning_rate_actor_leaf_array = 0.006414075616512551
    args.learning_rate_actor_log_std = 0.00015395109187787975
    args.learning_rate_critic = 0.00032866087550350426
    args.n_envs = 15
    args.n_steps = 128
    args.n_update_epochs = 7
    args.norm_adv = True
    args.reduce_lr = False
    args.vf_coef = 0.750
    args.dropout = 0.000
    args.depth = 7
    args.minibatch_size = 64

# To minimize variance from the original paper, we use the same n_steps exponential scaling as ppo.py. This ensures proper exploration by the agent.
args.n_steps = max(16, args.n_steps // 8)
initial_steps = args.n_steps
batch_size = int(args.n_envs * args.n_steps)
minibatch_size = args.minibatch_size
while batch_size // minibatch_size < 2:
    minibatch_size = minibatch_size // 2
n_iterations = args.total_steps // batch_size

# In order to replicate the same experiments, we have to index the same subset of actions as the original paper
def get_environment_bounds(envs):
    '''
    Obtains the observation and action indices corresponding to the environment

    Input:
        - envs: gym.Env or gym.vector.AsyncVectorEnv
    '''
    if args.n_envs > 1:
        args.obs_dim = envs.single_observation_space.shape[-1]
        if isinstance(envs.single_action_space, gym.spaces.Discrete):
            args.action_type = "discrete"
            if any(substring in args.env_id for substring in ['DistShift', 'Empty', 'LavaGap']):
                args.action_dim = 3
                args.action_indices = [0,1,2]
            elif "DoorKey" in args.env_id:
                args.action_dim = 5
                args.action_indices = [0,1,2,3,5]        
            else:
                args.action_dim = envs.single_action_space.n
                args.action_indices = [i for i in range(args.action_dim)]
        elif isinstance(envs.single_action_space, gym.spaces.Box):
            args.action_dim = envs.single_action_space.shape[-1]
            args.action_indices = [i for i in range(args.action_dim)]
            args.action_type = "continuous"
    else:
        args.obs_dim = envs.observation_space.shape[-1]
        if isinstance(envs.action_space, gym.spaces.Discrete):
            args.action_type = "discrete"
            if any(substring in args.env_id for substring in ['DistShift', 'Empty', 'LavaGap']):
                args.action_dim = 3
                args.action_indices = [0,1,2]
            elif "DoorKey" in args.env_id:
                args.action_dim = 5
                args.action_indices = [0,1,2,3,5]      
            else:
                args.action_dim = envs.action_space.n
                args.action_indices = [i for i in range(args.action_dim)]
        elif isinstance(envs.action_space, gym.spaces.Box):
            args.action_dim = envs.action_space.shape[-1]
            args.action_indices = [i for i in range(args.action_dim)]
            args.action_type = "continuous"

    print('Observation Dim:', args.obs_dim)
    print("Actions Dim:", args.action_dim)


# Helper function provided by original paper that maps a function to every node in the SYMPOL tree
def map_nested_fn(fn):
    '''
    Recursively apply `fn` to key-value pairs of a nested dict.
    '''
    def map_fn(nested_dict):
        return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()}
    return map_fn


def create_SYMPOL_agent(envs):
    '''
    Creates and returns the actor, critc, and their corresponding JAX TrainStates

    Input:
        - envs: gym.Env or gym.vector.AsyncVectorEnv
    Output:
        - actor: SYMPOL model
        - critic: MLP model
        - actorState: TrainState
        - criticState: TrainState
    '''
    # To minimize variance from original paper, we initialize our models and random keys using the same code
    model_key = jax.random.PRNGKey(args.seed)
    model_key, actor_key, critic_key = jax.random.split(model_key, 3)

    actor = SYMPOL(
        num_states=args.obs_dim,
        num_actions=args.action_dim,
        depth=args.depth,
        action_type=args.action_type,
    )
    actor_state = ActorTrainState.create(
        apply_fn=None,
        params=actor.init(
            actor_key, 
            jnp.array([envs.single_observation_space.sample()]) if args.n_envs > 1 else jnp.array([envs.observation_space.sample()])
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.multi_transform(
                {
                    'threshold_values': optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_split_values), 
                    'feature_assignments': optax.inject_hyperparams(optax.adamw)(args.learning_rate_actor_split_idx_array), 
                    'leaf_outputs': optax.inject_hyperparams(optax.adamw)(args.learning_rate_actor_leaf_array), 
                    'log_std_dev': optax.inject_hyperparams(optax.adamw)(args.learning_rate_actor_log_std),
                }, 
                map_nested_fn(lambda k, _: k)),
                swag(10, 2),
        ),
        grad_accum=jax.tree.map(
            jnp.zeros_like, 
            actor.init(
                actor_key, 
                jnp.array([envs.single_observation_space.sample()]) if args.n_envs > 1 else jnp.array([envs.observation_space.sample()])
            )
        ),
        indices=actor.init_indices(),
    )

    critic = CriticMLP()
    critic_state = TrainState.create(
        apply_fn=None,
        params=critic.init(
            critic_key, 
            jnp.array([envs.single_observation_space.sample()]) if args.n_envs > 1 else jnp.array([envs.observation_space.sample()])
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm), optax.adamw(learning_rate=args.learning_rate_critic)
        ),
    )

    return actor, actor_state, critic, critic_state


def evaluate_agent(actor_state, env_id, n_episodes, name_appendix, seed=100):
    '''
    Helper function taken from the original codebase which performs an evaluation episode in a single environment.
    Writes the pruned and complete policy trees to output files. Optionally renders the video replay of the agent in the episode.
    '''
    video_folder = 'videos/wandb'
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)    

    score = []
    node_count = 0
    
    for episode_index in range(n_episodes):
        temp_env = build_env(env_id, n_env=1, view_size=args.view_size)               
        image_path = os.path.join(video_folder, run_name + "-" + "-" + env_id)

        done, trunc = False, False
        obs, info = temp_env.reset(seed=seed + episode_index)
        running_reward = 0
        frames = []
        dones = False
        step_counter = 0
        while not done and not trunc:
            if args.render_env and render_now:
                frame = temp_env.render()
        
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                text_step = f'Step: {step_counter}'
                font_size = frame.shape[0] // 20
                draw.text((font_size, font_size*0.5), text_step, (200, 200, 200), font=ImageFont.load_default())
                text_reward = f'Reward: {running_reward}'
                draw.text((font_size, font_size*2.0), text_reward, (200, 200, 200), font=ImageFont.load_default())
                
                frames.append(np.array(image))
                
            actor_params = actor_state.params
            if args.action_type == 'discrete':
                action_logits = actor.apply(actor_params, np.array([obs]), indices=actor_state.indices)

                action = jnp.argmax(action_logits, axis=1)
                action = jnp.squeeze(action, axis=0)
            else:
                result = actor.apply(actor_params, np.array([obs]), indices=actor_state.indices)
                action_distribution = distrax.MultivariateNormalDiag(result[0], jnp.exp(result[1]))
                action = action_distribution.mean()
                action = jnp.squeeze(action, axis=0)

            if args.env_id == "MiniGrid-DoorKey-5x5-v0":
                action = np.array(args.action_indices[action], dtype=np.float64)
            else:
                action = np.array(action)

            next_obs, rewards, done, trunc, info = temp_env.step(action)
            
            running_reward += rewards

            obs = next_obs
            step_counter += 1
            
        score.append(running_reward)
        # The following visualization helper functions/code is taken from the original paper and utils.py. See utils.py for detailed explanation
        if (args.render_env and render_now):
            if args.render_video:
                frame = temp_env.render()
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                text_step = f'Step: {step_counter}'
                font_size = frame.shape[0] // 20
                draw.text((font_size, font_size*0.5), text_step, (200, 200, 200), font=ImageFont.load_default())
                text_reward = f'Reward: {running_reward}'
                draw.text((font_size, font_size*2.0), text_reward, (200, 200, 200), font=ImageFont.load_default())
                
                frames.append(np.array(image))

                numpy_clip = np.transpose(np.array(frames), (0, 3, 1, 2)) 
                fps = 5 if 'MiniGrid' in env_id else 25

                wandb.log({"gameplay" + name_appendix + '_trial_ep' + str(episode_index): wandb.Video(numpy_clip, fps=fps, format="mp4")}, commit=False)
            if episode_index==0:
                image_path, node_count = plot_decision_tree(
                                                split_values=actor_params['threshold_values'], 
                                                split_indices=actor_params['feature_assignments'], 
                                                leaf_values=actor_params['leaf_outputs'],
                                                features_by_estimator=jnp.arange(args.obs_dim),
                                                image_path=image_path,
                                                observation_labels=None if args.env_id not in OBSERVATION_LABELS.keys() else OBSERVATION_LABELS[args.env_id],
                                                filename_appendix = "",
                                                env=temp_env, 
                                                prune=True,
                                                continuous = args.action_type != 'discrete'
                                            )
                image_path_plot = image_path + '.png'    
                wandb.log({"DT_"+ name_appendix + '_trial' + str(episode_index) + '_estNumber': wandb.Image(image_path_plot)}, commit=False)                                 
                image_path_complete = image_path + '_COMPLETE'
                image_path_complete, _ = plot_decision_tree(
                                                split_values=actor_params['threshold_values'], 
                                                split_indices=actor_params['feature_assignments'], 
                                                leaf_values=actor_params['leaf_outputs'],
                                                features_by_estimator=jnp.arange(args.obs_dim),
                                                image_path=image_path_complete,
                                                observation_labels=None if args.env_id not in OBSERVATION_LABELS.keys() else OBSERVATION_LABELS[args.env_id],
                                                filename_appendix = "",
                                                env=temp_env, 
                                                prune=False,
                                                continuous = args.action_type != 'discrete'
                                            )
                    
                image_path_plot = image_path_complete + '.png'     
                wandb.log({"DT_COMPLETE"+ name_appendix + '_trial' + str(episode_index) + '_estNumber': wandb.Image(image_path_plot)}, commit=False)
        
        temp_env.close()

    return score, node_count


if __name__ == "__main__":
    start_time = time.time()

    for random_trial in range(1, args.random_trials + 1):
        model_identifier = str(args.seed)
        run_name = '-'.join([args.experiment_name, args.env_id, model_identifier, str(random_trial)])

        envs = build_env(args.env_id, n_env=args.n_envs, view_size=args.view_size)

        get_environment_bounds(envs=envs)

        wandb_run = wandb.init(
            project=f"{args.experiment_name}_{args.env_id}",
            group=args.experiment_name,
            tags=[],
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True, 
        )

        env_seed = args.seed + (random_trial * 100)
        key = jax.random.PRNGKey(env_seed)
        env_key = jax.random.PRNGKey(env_seed)

        actor, actor_state, critic, critic_state = create_SYMPOL_agent(envs=envs)


        @jax.jit
        def get_action_and_value(
            actor_state: TrainState,
            critic_state: TrainState,
            next_obs: np.ndarray,
            next_done: np.ndarray,
            storage: Storage,
            step: int,
            key: jax.random.PRNGKey,
        ):
            '''
            Sample action, calculate value, logprob, entropy, and update storage.
            Returns storage object, action, and random key
            '''
            if args.action_type == "discrete":
                logits = actor.apply(actor_state.params, next_obs, indices=actor_state.indices)
                dist = distrax.Categorical(logits=logits)
            elif args.action_type == "continuous":
                logits, log_std_dev = actor.apply(actor_state.params, next_obs, indices=actor_state.indices)
                dist = distrax.MultivariateNormalDiag(logits, jnp.exp(log_std_dev))

            key, subkey = jax.random.split(key)
            action = dist.sample(seed=subkey)
            logprob = dist.log_prob(action)

            value = critic.apply(critic_state.params, next_obs)

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
            actor_state_params: flax.core.FrozenDict,
            critic_state_params: flax.core.FrozenDict,
            x: np.ndarray,
            action: np.ndarray,
        ):
            '''
            Sample action, calculate value, logprob, entropy, and update storage.
            Returns logprob, entropy, and value
            '''
            if args.action_type == "discrete":
                logits = actor.apply(actor_state_params, x, indices=actor_state.indices)
                dist = distrax.Categorical(logits=logits)
            elif args.action_type == "continuous":
                logits, log_std_dev = actor.apply(actor_state_params, x, indices=actor_state.indices)
                dist = distrax.MultivariateNormalDiag(logits, jnp.exp(log_std_dev))

            value = critic.apply(critic_state_params, x).squeeze()
            logprob = dist.log_prob(action)
            entropy = dist.entropy()

            return logprob, entropy, value


        lr_scheduler = optax.contrib.reduce_on_plateau(patience=3, factor=0.5)
        lr_scheduler_state = lr_scheduler.init(actor_state.params)

        episode_stats = EpisodeStatistics(
            episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
            episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
            returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
        )


        @jax.jit
        def compute_gae(
            critic_state: TrainState,
            next_obs: np.ndarray,
            next_done: np.ndarray,
            storage: Storage,
        ):
            '''
            Helper function to calculate the GAE estimate from the original SYMPOL paper.
            We use the original paper's computation of the GAE estimate as they implement a JAX optimization technique that is not specified in their writeup.
            Functionally, they perform the same backwards calculation of advantages as in hw2.py
            '''
            def compute_gae_once(carry, inp):
                advantages = carry
                nextdone, nextvalues, curvalues, reward = inp
                nextnonterminal = 1.0 - nextdone

                delta = reward + args.gamma * nextvalues * nextnonterminal - curvalues
                advantages = delta + args.gamma * args.gae_lambda * nextnonterminal * advantages
                return advantages, advantages

            next_value = critic.apply(critic_state.params, next_obs).squeeze()

            advantages = jnp.zeros((args.n_envs,))
            dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
            values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
            _, advantages = jax.lax.scan(compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True)
            
            storage = storage.replace(
                advantages=advantages,
                returns=advantages + storage.values,
            )
            return storage
        

        @jax.jit
        def ppo_loss_base(actor_state_params, critic_state_params, x, a, logp, mb_advantages, mb_returns):
            '''
            Standard PPO loss calculation taken from Clean-RL's implementation of PPO at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
            '''
            newlogprob, entropy, newvalue = get_action_and_value2(actor_state_params, critic_state_params, x, a)
            logratio = newlogprob - logp
            ratio = jnp.exp(logratio)
            approx_kl = ((ratio - 1) - logratio).mean()

            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Value loss
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))


        def create_rollout(n_steps, envs):
            '''
            Helper function from original paper which generates rollout and updates Storage object with environment details.
            We adapt the function by fixing action index mapping
            '''
            def rollout_(actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step):
                for step in range(0, n_steps):
                    global_step += args.n_envs
                    storage, action, key = get_action_and_value(
                        actor_state, critic_state, next_obs, next_done, storage, step, key
                    )
                    
                    if args.env_id == "MiniGrid-DoorKey-5x5-v0":
                        action = np.array([args.action_indices[single_action] for single_action in action], dtype=np.float64)
                    else:
                        action = np.array(action)

                    next_obs, reward, next_done, trunc, info = envs.step(action)

                    new_episode_return = episode_stats.episode_returns + reward
                    new_episode_length = episode_stats.episode_lengths + 1
                    episode_stats = episode_stats.replace(
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
                return actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step       
            return rollout_


        global_step = 0
        next_obs, _ = envs.reset(seed=env_seed)
        next_done = np.zeros(args.n_envs).astype(bool)

        avg_score_list = []
        iteration = 1
        last_eval = 0
        n_steps_old = 0

        avg_episodic_return_list = []

        ppo_loss_base_grad_fn = jax.value_and_grad(ppo_loss_base, argnums=(0, 1), has_aux=True)


        @jax.jit
        def update_ppo(
            actor_state: TrainState,
            critic_state: TrainState,                
            storage: Storage,
            key: jax.random.PRNGKey,
        ):
            '''
            Helper function which performs the PPO update on the actor and critic networks.
            We re-use this function directly because the authors do not explain or specify their optimizations using jax.lax.scan(), which affects the implementation results.
            '''
            def update_epoch(carry, unused_inp):
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

                flatten_storage = jax.tree_util.tree_map(flatten, storage)
                shuffled_storage = jax.tree_util.tree_map(convert_data, flatten_storage)

                def update_minibatch(carry, minibatch):
                    actor_state, critic_state = carry
                    (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), (actor_grads, critic_grads) = ppo_loss_base_grad_fn(
                        actor_state.params,
                        critic_state.params,              
                        minibatch.obs,
                        minibatch.actions,
                        minibatch.logprobs,
                        minibatch.advantages,
                        minibatch.returns,
                    )
                    actor_state = actor_state.apply_gradients(grads=actor_grads)
                    critic_state = critic_state.apply_gradients(grads=critic_grads)
                    
                    return (actor_state, critic_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl)
                
                (actor_state, critic_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
                    update_minibatch, (actor_state, critic_state), shuffled_storage
                )

                return (actor_state, critic_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl)

            (actor_state, critic_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
                update_epoch, (actor_state, critic_state, key), (), length=args.n_update_epochs
            )

            return actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key


        while global_step < args.total_steps:       
            wandb_log = {}
            # To minimize variance from the original paper, we use the same n_steps exponential scaling as ppo.py. This ensures proper exploration by the agent.
            increase_factor = int(2**(np.ceil((((global_step+1)*8)/(1+args.total_steps)))-1))
            n_steps = initial_steps * increase_factor           
            batch_size = int(args.n_envs * n_steps)
            current_eval = global_step // args.eval_freq
            if n_steps != n_steps_old:               
                rollout = create_rollout(n_steps, envs)
                n_steps_old = n_steps         
            
            storage = Storage(
                obs=jnp.zeros(
                    (n_steps, args.n_envs) + envs.single_observation_space.shape if args.n_envs > 1 else (n_steps, args.n_envs) + envs.observation_space.shape
                ),
                actions=jnp.zeros(
                    (n_steps, args.n_envs) + envs.single_action_space.shape if args.n_envs > 1 else (n_steps, args.n_envs) + envs.action_space.shape, 
                    dtype=(jnp.int32 if args.action_type == "discrete" else jnp.float32)
                ),
                logprobs=jnp.zeros((n_steps, args.n_envs)),
                dones=jnp.zeros((n_steps, args.n_envs)),
                values=jnp.zeros((n_steps, args.n_envs)),
                advantages=jnp.zeros((n_steps, args.n_envs)),
                returns=jnp.zeros((n_steps, args.n_envs)),
                rewards=jnp.zeros((n_steps, args.n_envs)),
            )

            actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step = rollout(
                actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step
            )

            storage = compute_gae(critic_state, next_obs, next_done, storage)
            actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
                actor_state,
                critic_state,
                storage,
                key
            )
            
            avg_episodic_return = np.mean(np.array(episode_stats.returned_episode_returns))
            avg_episodic_return_list.append(avg_episodic_return)

            # Performs training evaluations at every (global_step + batch_size) bucket and at the first iteration
            if iteration == 1 or current_eval > last_eval or global_step + batch_size >= args.total_steps:
                last_eval = current_eval
                render_now = True if args.render_each_eval else True if global_step + batch_size >= args.total_steps else False

                end_time = time.time()
                elapsed_time = end_time - start_time

                score, node_count = evaluate_agent(
                    actor_state=actor_state,
                    env_id=args.env_id,
                    n_episodes=args.n_eval_episodes,
                    name_appendix="",
                    seed=env_seed
                )

                avg_score = np.mean(score).item()
                std_score = np.std(score).item()

                if args.reduce_lr:
                    _, lr_scheduler_state = lr_scheduler.update(
                        updates=actor_state.params, state=lr_scheduler_state, value=avg_score
                    )
                    actor_state.opt_state[1][0]['threshold_values'][0].hyperparams["learning_rate"] = args.learning_rate_actor_split_values * lr_scheduler_state.scale
                    actor_state.opt_state[1][0]['feature_assignments'][0].hyperparams["learning_rate"] = args.learning_rate_actor_split_idx_array * lr_scheduler_state.scale
                    actor_state.opt_state[1][0]['leaf_outputs'][0].hyperparams["learning_rate"] = args.learning_rate_actor_leaf_array * lr_scheduler_state.scale
                    actor_state.opt_state[1][0]['log_std_dev'][0].hyperparams["learning_rate"] = args.learning_rate_actor_log_std * lr_scheduler_state.scale

                end_time = time.time()
                elapsed_time = end_time - start_time
                start_time = end_time

                # The following wandb logging code is taken directly from the original codebase as it is not important to our replication
                print(f"Train: global_step={global_step}, avg_eval_episodic_return={avg_score} (Elapsed time: {elapsed_time} seconds)")

                # Train-time eval statistics
                wandb_log['train/avg_score'] = avg_score
                wandb_log['train/std_score'] = std_score
                wandb_log['train/score_list'] = score
            
                avg_score_list.append(avg_score)
                wandb_log['train/node_count'] = node_count

                # Performs testing evaluation for the policy at the final evaluation step
                if global_step + batch_size >= args.total_steps:
                    # Using the same final eval seed for reproducibility
                    test_seed = 123456
                            
                    score_test, node_count_test = evaluate_agent(
                        actor_state, 
                        args.env_id,
                        n_episodes=args.n_eval_episodes,
                        name_appendix="",
                        seed=test_seed
                    )
            
                    avg_score_test = np.mean(score_test).item()
                    std_score_test = np.std(score_test).item()

                    print(f"Test: global_step={global_step}, avg_eval_episodic_return={avg_score_test} (Elapsed time: {elapsed_time} seconds)")
                    # Final eval/testing statistics
                    wandb_log['test/avg_score_test'] = avg_score_test
                    wandb_log['test/std_score_test'] = std_score_test
                    wandb_log['test/score_list_test'] = score_test
                    wandb_log['test/node_count_test'] = node_count_test


            wandb_log['global_step'] = global_step

            # Training statistics
            wandb_log['train/avg_episodic_return'] = avg_episodic_return
            wandb_log['train/avg_episodic_return_10'] = np.mean(avg_episodic_return_list[-10:])
            wandb_log['train/avg_episodic_return_100'] = np.mean(avg_episodic_return_list[-100:])

            wandb.log(wandb_log)

            iteration = iteration + 1

        wandb_run.finish()
        envs.close()
