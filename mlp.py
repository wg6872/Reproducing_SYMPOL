import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import numpy as np

class CriticMLP(nn.Module):
    "critic to estimate value function"
    hidden_layers: int = 2
    hidden_size: int = 256

    def setup(self):
        self.dense_layers = [nn.Dense(self.hidden_size) for _ in range(self.hidden_layers)]
        self.value_head = nn.Dense(1)

    def __call__(self, x):
        for layer in self.dense_layers:
            x = layer(x)
            x = nn.relu(x)
        v = self.value_head(x)
        return v

class DiscreteActorMLP(nn.Module):
    "actor network for discrete action spaces"
    out_features: int # discrete actions
    hidden_layers: int = 2
    hidden_size: int = 256

    def setup(self):
        self.dense_layers = [nn.Dense(self.hidden_size) for _ in range(self.hidden_layers)]
        self.action_head = nn.Dense(self.out_features)

    def __call__(self, x):
        for layer in self.dense_layers:
            x = layer(x)
            x = nn.relu(x)
        logits = self.action_head(x)
        return logits

class ContinuousActorMLP(nn.Module):
    """actor network for continuous action spaces"""
    out_features: int # dimensions
    hidden_layers: int = 2
    hidden_size: int = 256

    def setup(self):
        # orthogonal init (common)
        self.dense_layers = [
            nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0)
            ) for _ in range(self.hidden_layers)
        ]
            
        self.mu_head = nn.Dense(
            self.out_features,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )

        self.log_sigma = self.param(
            "log_sigma", 
            nn.initializers.zeros,
            (self.out_features,)
        )

    def __call__(self, x):
        for layer in self.dense_layers:
            x = layer(x)
            x = nn.relu(x)

        mu = self.mu_head(x)

        return mu, self.log_sigma
