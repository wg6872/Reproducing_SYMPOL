import jax
import jax.numpy as jnp
from flax import struct

# Note: By defining the SYMPOL class as a flax.struct, we can apply automatic differentiation to the tree
# Source: https://flax.readthedocs.io/en/stable/api_reference/flax.struct.html
@struct.dataclass(frozen=False)
class SYMPOL:
    # This is necessary to resolve bugs regarding immutability of the SYMPOL object
    num_states: int = struct.field(pytree_node=False)
    num_actions: int = struct.field(pytree_node=False)
    depth: int = struct.field(pytree_node=False)
    action_type: str = struct.field(pytree_node=False)

    def init(self, random_key, *args):
        # Note: To minimize variance from the original paper, we split the key the same way despite not explicitly using multiple estimators in our code
        # Note: log_std_dev_PRNG_key is only necessary to estimate the distribution of continuous-action leaves
        (
            _, 
            threshold_PRNG_key, 
            feature_index_PRNG_key, 
            leaf_outputs_PRNG_key, 
            log_std_dev_PRNG_key
        ) = jax.random.split(random_key, 5)

        self.num_leaves = 2**self.depth
        self.num_internal_nodes = 2**self.depth - 1

        # Initialize all tree parameters w.r.t. Normal distribution
        threshold_values = 0.00 + 0.05 * jax.random.normal(
            key=threshold_PRNG_key,
            shape=[self.num_internal_nodes, self.num_states],
            dtype=jnp.float32
        )
        feature_assignments = 0.00 + 0.05 * jax.random.normal(
            key=feature_index_PRNG_key,
            shape=[self.num_internal_nodes, self.num_states],
            dtype=jnp.float32
        )
        leaf_outputs = 0.00 + 0.05 * jax.random.normal(
            key=leaf_outputs_PRNG_key,
            shape=[self.num_leaves, self.num_actions],
            dtype=jnp.float32
        )
        log_std_dev = 0.00 + 0.05 * jax.random.normal(
            key=log_std_dev_PRNG_key,
            shape=[self.num_actions,],
            dtype=jnp.float32
        )

        # Note: Like in the original paper, we package the parameters to be compatible with flax.linen.init()
        # Source: https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/init_apply.html
        parameters = {
            "threshold_values": threshold_values,
            "feature_assignments": feature_assignments,
            "leaf_outputs": leaf_outputs,
            "log_std_dev": log_std_dev
        }

        return parameters 
    

    def init_indices(self):
        """
        Pre-computes the branch to get to each leaf node

        Calculation equations are taken from the original paper at https://github.com/s-marton/SYMPOL/blob/master/sympol.py
        # Authors: Sascha Marton, Tim Grams, Florian Vogt, Stefan Ludtke, Christian Bartelt, Heiner Stuckenschmidt
        # License: MIT

        Output:
        - leaf_decisions: [num_leaves, depth]
            For each leaf, lists the decision at every level to reach the leaf (0 = left, 1 = right)
        - leaf_internal_nodes: [num_leaves, depth]
            For each leaf, lists the node visited at every level to reach the leaf (pre-order indexing of internal nodes) 
        """

        leaf_decisions = jnp.zeros((self.num_leaves, self.depth), dtype=jnp.float32)
        leaf_internal_nodes = jnp.zeros((self.num_leaves, self.depth), dtype=jnp.int32)

        for i in range(self.num_leaves):
            for d in range(1, self.depth + 1):
                leaf_decisions = leaf_decisions.at[i, d - 1].set(
                    jnp.floor(i / (2 ** (self.depth - d))) % 2
                )
                leaf_internal_nodes = leaf_internal_nodes.at[i, d - 1].set(
                    (2 ** (d - 1) + jnp.floor(i / (2 ** (self.depth - (d - 1)))) - 1).astype(jnp.int32)
                )

        indices = {
            "leaf_decisions": leaf_decisions,
            "leaf_internal_nodes": leaf_internal_nodes
        }

        return indices


    @jax.jit
    def apply(self, params, inputs, indices):
        """
        Forward pass

        As with the original paper, we employ the same einsum syntax:
        - b is the batch size
        - l is num_leaves
        - i is num_internal_nodes
        - d is depth
        - s is num_states
        - a is num_actions

        Input: 
        - params: 
            tree parameters
        - inputs: [num_leaves,]
            vector of feature assignments 
        Output
        - policy: [num_leaves, num_actions]
        """
        threshold_values = params["threshold_values"]
        feature_assignments = params["feature_assignments"]
        leaf_outputs = params["leaf_outputs"]
        log_std_dev = params["log_std_dev"]

        leaf_decisions = indices["leaf_decisions"]
        leaf_internal_nodes = indices["leaf_internal_nodes"]

        # Straight-Through estimator excludes hardmax from backpropagation calculation --> differentiates entmax approximation
        # feature_assignments is the hardmax over all num_states
        # Source: https://docs.jax.dev/en/latest/higher-order.html#straight-through-estimator-using-stop-gradient
        feature_entmax = entmax15JAX(feature_assignments)
        feature_hardmax = jax.nn.one_hot(
            jnp.argmax(feature_assignments, axis=-1), num_classes=self.num_states
        )

        # Shape: [i, s]
        feature_assignments = feature_entmax - jax.lax.stop_gradient(feature_entmax - feature_hardmax)

        internal_node_thresholds = jnp.einsum("is,is->i", threshold_values, feature_assignments)
        observed_state_values = jnp.einsum("bs,is->bi", inputs, feature_assignments)

        # The original codebase uses soft_sign(), which is equivalent
        internal_node_results_sigmoid = jax.nn.sigmoid(internal_node_thresholds - observed_state_values)
        internal_node_results_sigmoid_round = jnp.round(internal_node_results_sigmoid)

        # Shape: [b, i]
        # Straight-Through estimator excludes rounding from backpropagation calculation --> differentiates sigmoid approximation
        # Source: https://docs.jax.dev/en/latest/higher-order.html#straight-through-estimator-using-stop-gradient
        internal_node_results = internal_node_results_sigmoid - jax.lax.stop_gradient(internal_node_results_sigmoid - internal_node_results_sigmoid_round)

        # Shape: [b, l, d]
        internal_node_results_to_leaves = internal_node_results[:, leaf_internal_nodes]

        # Shape: [b, l]
        indicators = jnp.prod(
            ((1 - leaf_decisions) * internal_node_results_to_leaves + leaf_decisions * (1 - internal_node_results_to_leaves)),
            axis=2,
        )

        # To get the final vector of actions, we sum across all the leaf outputs
        action = jnp.einsum("la,bl->ba", leaf_outputs, indicators)
        if self.action_type == "continuous":
            action = [action, log_std_dev]

        return action


# Note: We inherit the entmax implementation from the original paper as the implementation is not specified in their methodology or specifically for SYMPOL
# Taken from: https://github.com/s-marton/SYMPOL/blob/master/sympol.py
# Cleared by Professor Henderson, Noted in our writeup
"""
Taken from: https://github.com/deep-spin/entmax/blob/master/entmax/activations.py

An implementation of entmax (Peters et al., 2019). See
https://arxiv.org/pdf/1905.05702 for detailed description.

This builds on previous work with sparsemax (Martins & Astudillo, 2016).
See https://arxiv.org/pdf/1602.02068.
"""

# Author: Ben Peters
# Author: Vlad Niculae <vlad@vene.ro>
# License: MIT

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
    """creates indices 0, ... , input[axis] unsqueezed to input dimensions"""
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
    NOTE this implementation uses the same heuristic as
    the original code: https://tinyurl.com/pytorch-entmax-line-203
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

    # Implementation taken from: https://github.com/deep-spin/entmax/tree/master/entmax

    """
    Entmax 1.5 implementation, heavily inspired by
     * paper: https://arxiv.org/pdf/1905.05702.pdf
     * pytorch code: https://github.com/deep-spin/entmax
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
