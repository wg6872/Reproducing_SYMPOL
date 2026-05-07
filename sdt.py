# sdt.py
# Author(s): Evan Soper
# Implementation of Soft Decision Tree (SDT) core logic and Actor and
# Critic wrappers

import jax
import jax.numpy as jnp
import flax.linen as nn

def entmoid15(x, temperature=1.0):
    """Binary entmax gate."""
    logits = jnp.stack([x / temperature, jnp.zeros_like(x)], axis=-1)
    return entmax15JAX(logits)[..., 0]

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
    
    def _calc_leaf_probs(self, p_left, p_right, node_prob, node=0, depth=0, max_path=False):
        """
        Helper to compute the probability of reaching each leaf.

        Args:
            p_left:     (batch_size, num_internal_nodes)
            p_right:    (batch_size, num_internal_nodes)
            node:       node index (heap-ordered)
            node_prob:  probability of reaching this node (batch_size,)
            depth:      current tree depth
            max_path:   route all prob. mass to the higher-prob. child

        Returns:
            probability of reaching each leaf (batch_size, num_leaves_in_subtree)
        """
        if depth == self.depth:
            return jnp.expand_dims(node_prob, axis=1)
        
        if max_path:
            go_left = (p_left[:, node] >= p_right[:, node]).astype(node_prob.dtype)
            # for hard routing, probability is 1 if the left child's routing 
            # probability was higher in the soft tree
            p_left_child = node_prob * go_left
            p_right_child = node_prob * (1.0 - go_left)
        else:
            p_left_child = node_prob * p_left[:, node]
            p_right_child = node_prob * p_right[:, node]

        left = 2 * node + 1
        right = 2 * node + 2

        left_leaves = self._calc_leaf_probs(p_left, p_right, p_left_child, left, depth+1, max_path)
        right_leaves = self._calc_leaf_probs(p_left, p_right, p_right_child, right, depth+1, max_path)

        return jnp.concatenate([left_leaves, right_leaves], axis=1)

    def __call__(self, obs, max_path=False):
        p_left = entmoid15(self.internal(obs))    
        p_right = 1 - p_left
        
        # probability of starting at the root is always 1
        root_prob = jnp.ones((obs.shape[0],))
        leaf_probs = self._calc_leaf_probs(p_left, p_right, root_prob, max_path=max_path)
        
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

    def __call__(self, x, max_path=False):
        return self.sdt(x, max_path)

"""Actor wrapper for the SDT."""
class Actor_SDT(nn.Module):
    action_dim: int
    depth: int
    is_discrete: bool
    
    def setup(self):
        self.sdt = SDT(out_dim=self.action_dim, 
                       depth=self.depth, 
                       is_discrete=self.is_discrete)

    def __call__(self, x, max_path=False):
        return self.sdt(x, max_path)

"""
The rest of this file contains helper functions for entmax. I initially opted
to use sigmoid, but the performance difference between SDT and D-SDT was 
jarring without any sparsification.

Code taken from: https://github.com/deep-spin/entmax/blob/master/entmax/activations.py
Authored By: Ben Peters, Vlad Niculae 
Author Contact: vlad@vene.ro
"""

def top_k_over_axisJAX(inputs, k, axis=-1, **kwargs):
    with jax.named_scope("top_k_along_axis"):
        if axis == -1:
            return jax.lax.top_k(inputs, k)

        perm_order = list(range(inputs.shape.ndims))
        perm_order.append(perm_order.pop(axis))
        inv_order = [perm_order.index(i) for i in range(len(perm_order))]

        input_perm = jnp.transpose(inputs, perm_order)
        input_perm_sorted, sort_indices_perm = jax.lax.top_k(input_perm, k=k, **kwargs)

        input_sorted = jnp.transpose(input_perm_sorted, inv_order)
        sort_indices = jnp.transpose(sort_indices_perm, inv_order)
    return input_sorted, sort_indices


def _make_ix_like(inputs, axis=-1):
    """creates indices 0, ... , input[axis] unsqueezed to input dimensios"""
    assert jnp.ndim(inputs) is not None
    rho = jnp.arange(1, inputs.shape[axis] + 1, dtype=jnp.float32)
    view = [1] * jnp.ndim(inputs)
    view[axis] = -1
    return jnp.reshape(rho, view)


def jax_gather_nd(params, indices):
    tuple_indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    return params[tuple_indices]


def gather_over_axisJAX(values, indices, gather_axis):
    assert jnp.ndim(indices) is not None
    assert jnp.ndim(indices) == jnp.ndim(values)

    ndims = jnp.ndim(indices)
    gather_axis = gather_axis % ndims
    shape = jnp.shape(indices)

    selectors = []
    for axis_i in range(ndims):
        if axis_i == gather_axis:
            selectors.append(indices)
        else:
            index_i = jnp.arange(shape[axis_i])
            index_i = jnp.reshape(index_i, [-1 if i == axis_i else 1 for i in range(ndims)])
            index_i = jnp.tile(index_i, [shape[i] if i != axis_i else 1 for i in range(ndims)])
            selectors.append(index_i)
    return jax_gather_nd(values, jnp.stack(selectors, axis=-1))


def entmax_threshold_and_supportJAX(inputs, axis=-1):
    """
    Computes clipping threshold for entmax1.5 over specified axis
    :param inputs: (entmax1.5 inputs - max) / 2
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    """

    with jax.named_scope("entmax_threshold_and_supportJAX"):
        num_outcomes = inputs.shape[axis]

        inputs_sorted, _ = top_k_over_axisJAX(inputs, k=num_outcomes, axis=axis, sorted=True)

        rho = _make_ix_like(inputs, axis=axis)

        mean = jnp.cumsum(inputs_sorted, axis=axis) / rho

        mean_sq = jnp.cumsum(jnp.square(inputs_sorted), axis=axis) / rho
        delta = (1 - rho * (mean_sq - jnp.square(mean))) / rho

        delta_nz = jax.nn.relu(delta)
        tau = mean - jnp.sqrt(delta_nz)

        support_size = jnp.sum(jnp.less_equal(tau, inputs_sorted), axis=axis, keepdims=True)

        tau_star = gather_over_axisJAX(tau, support_size - 1, axis)
    return tau_star, support_size


def entmax15JAX(inputs, axis=-1):
    """
    This particular function from: https://github.com/deep-spin/entmax/tree/master/entmax.
    
    :param inputs: similar to softmax logits, but for entmax1.5
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    :return: entmax activations of same shape as inputs
    """

    @jax.custom_gradient
    def _entmax_inner(inputs):
        with jax.named_scope("entmax"):
            inputs = inputs / 2  # divide by 2 so as to solve actual entmax
            inputs -= jnp.max(inputs, axis, keepdims=True)  # subtract max for stability

            threshold, _ = entmax_threshold_and_supportJAX(inputs, axis)
            outputs_sqrt = jax.nn.relu(inputs - threshold)
            outputs = jnp.square(outputs_sqrt)

        def grad_fn(d_outputs):
            with jax.named_scope("entmax_grad"):
                d_inputs = d_outputs * outputs_sqrt
                q = jnp.sum(d_inputs, axis=axis, keepdims=True)
                q = q / jnp.sum(outputs_sqrt, axis=axis, keepdims=True)
                d_inputs -= q * outputs_sqrt
                return d_inputs

        return outputs, grad_fn

    return _entmax_inner(inputs)
