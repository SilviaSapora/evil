<h1 align="center">EvIL: Evolution Strategies for Generalisable Imitation Learning</h1>

<p align="center">
      <img src="https://img.shields.io/badge/python-3.8_%7C_3.9-blue" />
      <a href= "https://github.com/psf/black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
      <a href= "https://github.com/FLAIROx/jaxirl/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
       
</p>

[**Installation**](#install) | [**Setup**](#setup) | [**Algorithms**](#algorithms) | [**Citation**](#citation)

## Evolution Strategies for Generalisable Imitation Learning

Often times in imitation learning (IL), the environment we collect expert demonstrations in and the environment we want to deploy our learned policy in aren't exactly the same (e.g. demonstrations collected in simulation but deployment in the real world). Compared to policy-centric approaches to IL like behavioural cloning, reward-centric approaches like **Inverse Reinforcement Learning** (IRL) often better replicate expert behaviour in new environments. 

This transfer is usually performed by optimising the recovered reward under the dynamics of the target environment. However:
1. we find that modern deep IL algorithms frequently recover rewards which induce policies far weaker than the expert, even in the same environment the demonstrations were collected in
2. these rewards are often quite poorly shaped, necessitating extensive environment interaction to optimise effectively.

We provide simple and scalable fixes to both of these concerns. 
1. We find that **reward model ensembles** combined with a slightly different training objective significantly improves re-training and transfer performance.
2. To improve the poorly shaped rewards, we propose a novel **evolution-strategies** based method **EvIL** to optimise for a reward-shaping term that speeds up re-training in the target environment

On a suite of continuous control tasks, we are able to re-train policies in target (and source) environments more interaction-efficiently than prior work.

<div class="collage">
    <div class="column" align="center">
        <div class="row" align="center">
            <img src="https://github.com/SilviaSapora/evil/blob/main/images/hopper.jpg" alt="hopper" width="40%">
            <img src="https://github.com/SilviaSapora/evil/blob/main/images/humanoid.jpg" alt="humanoid" width="40%">
        </div>
        <div class="row" align="center">
            <img src="https://github.com/SilviaSapora/evil/blob/main/images/hopper_irl.jpg" alt="hopper_irl" width="40%">
            <img src="https://github.com/SilviaSapora/evil/blob/main/images/humanoid_irl.jpg" alt="humanoid_irl" width="40%">
        </div>
        <div class="row" align="center">
            <img src="https://github.com/SilviaSapora/evil/blob/main/images/hopper_transfer.jpg" alt="hopper_transfer" width="40%">
            <img src="https://github.com/SilviaSapora/evil/blob/main/images/humanoid_transfer.jpg" alt="humanoid_transfer" width="40%">
        </div>
    </div>
</div>

## Why JAX?
JAX is a game-changer in the world of machine learning, empowering researchers and developers to train models with unprecedented efficiency and scalability. Here's how it sets a new standard for performance:

- GPU Acceleration: JAX harnesses the full power of GPUs by JIT compiling code in XLA. Executing environments directly on the GPU, we eliminate CPU-GPU bottlenecks due to data transfer. This results in remarkable speedups compared to traditional frameworks like PyTorch.
- Parallel Training at Scale: JAX effortlessly scales to multi-environment and multi-agent training scenarios, enabling **efficient parallelization** for massive performance gains.

All our code can be used with `jit`, `vmap`, `pmap` and `scan` inside other pipelines. 
This allows you to:
- 🎲 Efficiently run tons of seeds in parallel on one GPU
- 💻 Perform rapid hyperparameter tuning

We support the following brax environments:
- humanoid
- hopper
- walker
- ant

and classic control environments:
- cartpole
- pendulum
- reacher
- gridworld

## Setup

The high-level structure of this repository is as follows:
```
├── evil  # package folder
│   ├── configs # standard configs for inner and outer loop
│   ├── envs # extra envs
│   ├── irl # main scripts that implement Imitation Learning and IRL algorithms
│   ├── ├── bc.py # Code for standard Behavioural Cloning
│   ├── ├── irl.py # Code implementing basic IRL algorithm
│   ├── ├── irl_plus.py # Code implementing our IRL++ version of the algorithm
│   ├── ├── gail_discriminator.py # Used by irl.py to implement IRL algorithm
│   ├── ├── evil.py # Runs the shaping on both real and fake rewards
│   ├── ├── run_irl.py # Code used to run the IRL training, save the various metrics and retrain the agent on the recovered reward
|   ├── training # generated expert demos
│   ├── ├── ppo_v2_cont_irl.py # PPO implementation for continuous action envs
│   ├── ├── ppo_v2_irl.py # PPO implementation for discrete action envs
│   ├── ├── supervised.py # Standard supervised training implementation
│   ├── ├── wrappers.py # Utility wrappers for training
│   ├── utils # utility functions
├── experts # expert policies
├── scripts # scripts to reproduce results
├── plotting # plotting code to reproduce plots
```
### Install 
```
conda create -n evil python=3.10.8
conda activate evil
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=evil:$PYTHONPATH
```
> [!IMPORTANT]
> All scripts should be run from under ```evil/```. 

## Algorithms

Our IRL implementation is the [moment matching](https://arxiv.org/abs/2103.03236) version. 
This includes implementation tricks to make learning more stable, including decay on the discriminator and learner learning rates and gradient penalties on the discriminator.

## Reproduce Results
Simply run the following commands to reproduce our results.
To run standard IRL and generate a reward: 
```
python3 evil/scripts/run_irl.py --env env_name -sd 1
```
For the IRL++ version run: 
```
python3 evil/scripts/run_irl.py --env env_name -sd 1 --irl_plus
```
The script will print the name of the file it saves the reward parameters to. Then run the following to shape the recovered reward:
```
python3 evil/scripts/run_evil.py --env env_name -sd 1 --reward-filename <REWARD_FILENAME>
```
or the following to shape the original reward:
```
python3 evil/scripts/run_evil.py --env env_name -sd 1 --real
```
For all scripts, the default parameters in ```outer_training_configs.py``` and the trained experts in ```experts/``` will be used.
To run a sweep for irl++, you can use the ```evil/scripts/run_sweep.py``` script.

## Citation

If you find this code useful in your research, please cite:
```bibtex
@misc{sapora2024evil,
      title={EvIL: Evolution Strategies for Generalisable Imitation Learning}, 
      author={Silvia Sapora and Gokul Swamy and Chris Lu and Yee Whye Teh and Jakob Nicolaus Foerster},
      year={2024},
}
```

## See Also 🙌
Our work reused code, tricks and implementation details from the following libraries, we encourage you to take a look!

- [JAXIRL](https://github.com/FLAIROx/jaxirl): JAX implementation of basic IRL algorithms
- [FastIRL](https://github.com/gkswamy98/fast_irl): PyTorch implementation of moment matching IRL and FILTER algorithms.
- [PureJaxRL](https://github.com/luchris429/purejaxrl): JAX implementation of PPO, and demonstration of end-to-end JAX-based RL training.
