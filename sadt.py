import jax
import jax.numpy as jnp
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def collect_expert_trajectories(env, expert_apply_fn, expert_params, n_episodes: int, action_type: str, seed: int = 0, action_indices=None):
    """
    Collect state-action pairs by rolling out an expert policy in the environment.
    
    Simulates a trained expert policy (typically a neural network)
    to gather a dataset of (state, action) pairs. Pairs are then used to
    train decision tree baselines via supervised learning.
    
    Attributes:
        env: gymnasium environment instance
        expert_apply_fn: jax function to evaluate expert policy (returns logits/mean)
        expert_params: parameters of the expert neural network
        n_episodes (int): number of episodes to collect
        action_type (str): either 'discrete' or 'continuous' action space
        seed (int): random seed for reproducibility. default is 0.
        
    Returns:
        states (np.ndarray): array of collected states, shape (total_steps, state_dim)
        actions (np.ndarray): array of collected actions, shape (total_steps, action_dim) or (total_steps,)
    """
    states = []
    actions = []
    
    for episode in range(n_episodes):
        # reset environment for new episode with seeded randomness
        obs, info = env.reset(seed=seed + episode)
        done, trunc = False, False
        
        while not done and not trunc:
            if action_type == 'discrete':
                # expert outputs logits for discrete actions
                # argmax gives deterministic action selection
                logits = expert_apply_fn(expert_params, np.array([obs]))
                action = jnp.argmax(logits, axis=-1)[0]
            else:
                # expert outputs (mean, log_std) for continuous actions
                # use deterministic mean action for imitation learning dataset
                mean, log_std = expert_apply_fn(expert_params, np.array([obs]))
                # use deterministic mean action for the dataset
                action = mean[0]
                
            # store the state-action pair for supervised learning dataset
            states.append(obs)
            actions.append(action)
            
            # Remap action indices for environments with reduced action spaces (e.g. DoorKey)
            # The MLP outputs indices 0..N-1 for N actions, but the env expects the original action IDs
            env_action = action
            if action_indices is not None and action_type == 'discrete':
                env_action = action_indices[int(action)]
            
            # execute action and transition to next state
            obs, reward, done, trunc, info = env.step(np.array(env_action))
            
    return np.array(states), np.array(actions)

def fit_state_action_dt(env, expert_apply_fn, expert_params, max_depth: int, n_episodes: int = 25, action_type: str = 'discrete', action_dim: int = 1, seed: int = 0, action_indices=None):
    """
    Implements the state-action decision tree (sa-dt) baseline.
    
    Baseline extracts a symbolic policy by:
    1. collecting a dataset of state-action pairs from a trained expert neural network
    2. fitting a scikit-learn decision tree to learn the expert's policy via supervised learning
    
    Sa-dt is simpler than sympol but should suffer from information loss
    
    Attributes:
        env: gymnasium environment instance
        expert_apply_fn: jax function to evaluate expert policy
        expert_params: trained expert neural network parameters
        max_depth (int): maximum depth of the decision tree(s)
        n_episodes (int): number of episodes to collect expert data. default is 25.
        action_type (str): either 'discrete' or 'continuous'. default is 'discrete'.
        action_dim (int): dimensionality of continuous actions. default is 1.
        seed (int): random seed for reproducibility. default is 0.
    
    Returns:
        decisiontreeclassifier or decisiontreeregressor: fitted decision tree(s)
            - for discrete actions: single decisiontreeclassifier
            - for continuous actions: single decisiontreeregressor (action_dim=1)
                                      or list of decisiontreeregressor (action_dim>1)
    """
    # collect state-action pairs from expert demonstration
    X, y = collect_expert_trajectories(
        env, 
        expert_apply_fn, 
        expert_params, 
        n_episodes=n_episodes, 
        action_type=action_type, 
        seed=seed,
        action_indices=action_indices,
    )
    
    if action_type == 'discrete':
        # train single classifier for discrete action spaces
        # decision tree directly maps states to action classes
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
        dt.fit(X, y)
        return dt
    else:
        # continuous action spaces: train regressor(s)
        if action_dim == 1:
            # single output dimension: fit one regressor
            dt = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
            dt.fit(X, y)
            return dt
        else:
            # multi-dimensional output: fit one tree per action dimension
            # this allows each action dimension to be mapped independently by its own tree
            dts = [DecisionTreeRegressor(max_depth=max_depth, random_state=seed+i) for i in range(action_dim)]
            for i in range(action_dim):
                # train regressor to predict i-th action dimension from states
                dts[i].fit(X, y[:, i])
            return dts
