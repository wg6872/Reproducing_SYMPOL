import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import numpy as np

class Critic_MLP(nn.Module):
    """
    Critic for MLP that estimates the state value function V(s).
    
    Attributes:
        hidden_layers (int): number of hidden dense layers. default is 2.
        hidden_size (int): number of neurons in each hidden layer. default is 256.
    """
    hidden_layers: int = 2
    hidden_size: int = 256

    def setup(self):
        # initialize
        # fully connected hidden layers with ReLU activation function
        self.dense_layers = [nn.Dense(self.hidden_size) for _ in range(self.hidden_layers)]

        # output layer: single neuron for scalar value estimate
        self.value_head = nn.Dense(1)

    def __call__(self, x):
        # forward pass through
        # pass through hidden layers with ReLU activation function
        for layer in self.dense_layers:
            x = layer(x)
            x = nn.relu(x)
        # output scalar value estimate
        v = self.value_head(x)
        return v

class DiscreteActorMLP(nn.Module):
    """
    Actor network for discrete action spaces. Outputs logits for each possible action. 
    
    Attributes:
        out_features (int): number of discrete actions
        hidden_layers (int): number of hidden dense layers. default is 2.
        hidden_size (int): number of neurons in each hidden layer. default is 256.
    """
    out_features: int  # number of discrete actions
    hidden_layers: int = 2
    hidden_size: int = 256

    def setup(self):
        # initialize
        # fully connected hidden layers with ReLU activation function
        self.dense_layers = [nn.Dense(self.hidden_size) for _ in range(self.hidden_layers)]

        # output layer: logits for each discrete action
        # no softmax, will do in ppo.py probably
        self.action_head = nn.Dense(self.out_features)

    def __call__(self, x):
        # forward pass through
        # pass through hidden layers with ReLU activation function
        for layer in self.dense_layers:
            x = layer(x)
            x = nn.relu(x)
        # output raw logits
        logits = self.action_head(x)
        return logits

class ContinuousActorMLP(nn.Module):
    """
    Actor network for continuous action spaces. Outputs the mean (mu) of 
    a Gaussian policy and log standard deviation for continuous actions.
    Uses orthogonal weight initialization like the original for stability.
    
    The policy is N(mu, exp(log_sigma)), allowing the agent to explore while
    learning deterministic action mappings for efficient exploitation.
    
    Attributes:
        out_features (int): dimensionality of continuous action space
        hidden_layers (int): number of hidden dense layers. default is 2.
        hidden_size (int): number of neurons in each hidden layer. default is 256.
    """
    out_features: int  # dimensions of continuous action space
    hidden_layers: int = 2
    hidden_size: int = 256

    def setup(self):
        # init
        # orthogonal initialization with scaling factor sqrt(2) for hidden layers
        # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        self.dense_layers = [
            nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0)
            ) for _ in range(self.hidden_layers)
        ]
        
        # output later: for action mean with small scale orthogonal init
        # Small scale (0.01) is for policy output layers to start with
        # near-zero mean actions for more stability
        # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        self.mu_head = nn.Dense(
            self.out_features,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )

        # log standard deviation as learnable parameter
        # Initialized to zeros (sigma=1), can be optimized to adjust exploration
        self.log_sigma = self.param(
            "log_sigma", 
            nn.initializers.zeros,
            (self.out_features,)
        )

    def __call__(self, x):
        # forward pass through
        # pass through hidden layers with ReLU activation function
        for layer in self.dense_layers:
            x = layer(x)
            x = nn.relu(x)

        # compute action mean
        mu = self.mu_head(x)
        # return both mean and log-std for sampling Gaussian distributions
        return mu, self.log_sigma