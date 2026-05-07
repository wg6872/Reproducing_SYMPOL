# Reproducing_SYMPOL
This repository contains our submission for Track 1 of the final project for COS435 (Introduction to Reinforcement Learning). 

We replicate "Mitigating Information Loss in Tree-Based Reinforcement Learning via Direct Optimization" Marton et al. (2025).

## Requirements
### SYMPOL
**Note:** ```ppo_sympol.py``` requires both ```python==3.11.4``` and the package versions in ```requirements_sympol.txt```.

To run an environment using our version of SYMPOL, use the ```env_id``` argument for ```ppo_sympol.py```.

```bash
python ppo_sympol.py --env_id CartPole-v1
```

### MLP and SA-DT
`ppo_mlp_sadt.py` uses `requirements_sympol.txt`.

This version uses `jax==0.4.31` and not the CUDA-enabled jax package.

To run the MLP baseline, choose an environment with `--env-id` and set `--actor mlp`. The script automatically applies the environment-specific hyperparameters defined in `ppo_mlp_sadt.py`.

```bash
python ppo_mlp_sadt.py --env-id CartPole-v1 --actor mlp
```

To train the MLP and then distill it into a state-action decision tree, run the same script with `--actor sadt`.

```bash
python ppo_mlp_sadt.py --env-id CartPole-v1 --actor sadt
```

You can also override optional settings such as the number of random trials, evaluation frequency, tree depth, rendering, video capture, and wandb tracking:

```bash
python ppo_mlp_sadt.py --env-id CartPole-v1 --actor sadt --random-trials 1 --sadt-max-depth 4 --render-env --track
```

## References
```
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}

@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Yash Katariya and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}

@article{marton2024sympol,
  title={SYMPOL: Symbolic Tree-Based On-Policy Reinforcement Learning},
  author={Marton, Sascha and Grams, Tim and Vogt, Florian and L{\"u}dtke, Stefan and Bartelt, Christian and Stuckenschmidt, Heiner},
  journal={arXiv preprint arXiv:2408.08761},
  year={2024}
}

@misc{wandb,
title = {Experiment Tracking with Weights and Biases},
year = {2020},
note = {Software available from wandb.com},
url={https://www.wandb.com/},
author = {Biewald, Lukas},
}
```
