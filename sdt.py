# sdt.py
# Author(s): Evan Soper
# Implementation of Soft Decision Tree (SDT) core logic and Actor and
# Critic wrappers

import jax.numpy as jnp
import flax.linen as nn

"""
Core implementation logic for SDTs. 

Attributes:
    out_dim: output dimension 
    depth: depth of the SDT
    is_discrete: True if the action space is discrete
"""
class SDT(nn.Module):
    out_dim: int
    depth: int
    is_discrete: bool
    """
    Store parameters to compute probabilistic routing for each internal
    node and final policy distribution.
    """
    def setup(self):
        self.internal = nn.Dense(2 ** self.depth - 1)
        self.leaves = nn.Dense(self.out_dim)
        
        # for continuous action spaces
        if not self.is_discrete:
            self.log_std = nn.Dense(self.out_dim)
    
    def _calc_leaf_probs(self, p_left, p_right, node_prob, node=0, depth=0):
        """
        Helper to compute the probability of reaching each leaf.

        Args:
            p_left:     (batch_size, num_internal_nodes)
            p_right:    (batch_size, num_internal_nodes)
            node:       node index (heap-ordered)
            node_prob:  probability of reaching this node (batch_size,)
            depth:      current tree depth

        Returns:
            probability of reaching each leaf (batch_size, num_leaves_in_subtree)
        """
        if depth == self.depth:
            return jnp.expand_dims(node_prob, axis=1)

        p_left_child = node_prob * p_left[:, node]
        p_right_child = node_prob * p_right[:, node]

        left = 2 * node + 1
        right = 2 * node + 2

        left_leaves = self._calc_leaf_probs(p_left, p_right, p_left_child, left, depth+1)
        right_leaves = self._calc_leaf_probs(p_left, p_right, p_right_child, right, depth+1)

        return jnp.concatenate([left_leaves, right_leaves], axis=1)

    def __call__(self, obs):
        p_left = nn.sigmoid(self.internal(obs))
        p_right = 1 - p_left
        
        # probability of starting at the root is always 1
        root_prob = jnp.ones((obs.shape[0],))
        leaf_probs = self._calc_leaf_probs(p_left, p_right, root_prob)
        
        y_pred = self.leaves(leaf_probs)
        
        if not self.is_discrete:
            log_std = self.log_std(leaf_probs)
            # clip to prevent training collapse
            clipped_log_std = jnp.clip(log_std, -20, 2)
            y_pred = (y_pred, clipped_log_std)
        
        return y_pred
        
"""Critic wrapper for the SDT."""
class Critic_SDT(nn.Module):
    depth: int

    def setup(self):
        self.sdt = SDT(out_dim=1, depth=self.depth, is_discrete=True)

    def __call__(self, x):
        return self.sdt(x)

"""Actor wrapper for the SDT."""
class Actor_SDT(nn.Module):
    action_dim: int
    depth: int
    is_discrete: bool
    
    def setup(self):
        self.sdt = SDT(out_dim=self.action_dim, 
                       depth=self.depth, 
                       is_discrete=self.is_discrete)

    def __call__(self, x):
        return self.sdt(x)
