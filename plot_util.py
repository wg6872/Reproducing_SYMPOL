import numpy as np
import graphviz


# ---------------------------
# OPTIONAL: nicer feature names
# ---------------------------
OBSERVATION_LABELS = {
    "CartPole-v1": [
        "cart_position",
        "cart_velocity",
        "pole_angle",
        "pole_angular_velocity",
    ],
    "Pendulum-v1": [
        "cos(theta)",
        "sin(theta)",
        "theta_dot",
    ],
}


# ---------------------------
# Build tree structure
# ---------------------------
def convert_to_child_representation(split_values, split_indices, leaf_values):
    num_internal_nodes = split_values.shape[0]

    def build(node_id):
        if node_id >= num_internal_nodes:
            leaf_idx = node_id - num_internal_nodes
            dist = leaf_values[leaf_idx]

            return {
                "type": "leaf",
                "distribution": dist.tolist(),
            }

        feat_idx = np.argmax(split_indices[node_id])
        threshold = split_values[node_id, feat_idx]

        return {
            "type": "internal",
            "split_index": int(feat_idx),
            "split_value": float(threshold),
            "left": build(2 * node_id + 1),
            "right": build(2 * node_id + 2),
        }

    return build(0)


# ---------------------------
# Plot with Graphviz
# ---------------------------
def plot_tree(tree, path, obs_labels=None):
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB")

    def traverse(node):
        node_id = str(id(node))

        if node["type"] == "leaf":
            action = int(np.argmax(node["distribution"]))
            label = f"Action: {action}"
            dot.node(node_id, label, shape="box")
            return node_id

        feat = node["split_index"]
        thresh = node["split_value"]

        if obs_labels:
            name = obs_labels[feat]
        else:
            name = f"x{feat}"

        label = f"{name} <= {thresh:.3f}?"
        dot.node(node_id, label)

        left_id = traverse(node["left"])
        right_id = traverse(node["right"])

        dot.edge(node_id, left_id, label="True")
        dot.edge(node_id, right_id, label="False")

        return node_id

    traverse(tree)
    dot.render(path, format="png", cleanup=True)
    return path + ".png"


# ---------------------------
# MAIN FUNCTION YOU CALL
# ---------------------------
def plot_dsdt_from_params(params, config, out_path="tree"):
    """
    params = actor_state.params AFTER convert_to_discrete(...)
    """

    sdt = params["params"]["sdt"]

    kernel = np.array(sdt["internal"]["kernel"])   # (obs_dim, nodes)
    bias = np.array(sdt["internal"]["bias"])       # (nodes,)
    leaves = np.array(sdt["leaves"]["kernel"])     # (leaves, actions)

    # IMPORTANT: match their logic
    split_indices = kernel.T
    split_values = (kernel.T * bias[:, None])

    tree = convert_to_child_representation(
        split_values,
        split_indices,
        leaves,
    )

    obs_labels = OBSERVATION_LABELS.get(config["env_id"], None)

    return plot_tree(tree, out_path, obs_labels)