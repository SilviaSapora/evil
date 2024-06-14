import os
import pickle
import time
from evil.utils.utils import get_expert_obsv_and_actions
from evil.irl.irl_multi_discr import IRLPlus
from evil.utils.parser import get_parser
from evil.utils.plot import plot
import evosax

import wandb
from evil.utils.utils import (
    get_irl_config,
    get_plot_filename,
    is_irl,
)
from evil.irl.irl import IRL
from evil.irl.bc import BC
from evil.irl.rl import RL
from evil.utils.env_utils import get_env, get_test_params, is_brax_env
import jax.numpy as jnp
from evil.utils.utils import LossType, RewardWrapper, generate_config
import matplotlib.pyplot as plt
from evil.utils.utils import (
    RewardNetwork,
    get_observation_size,
    RewardNetworkPessimistic,
    RewardNetworkPenalty,
)

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import jax

print("Visible devices", jax.devices())

from evil.training.ppo_v2_cont_irl import (
    make_train as make_train_cont,
    eval as eval_cont,
)
from evil.training.ppo_v2_irl import make_train, eval
from evil.utils.utils import RewardNetwork


def wandb_callback(gen, popsize, fitness, returned_ep_real_returns, ep_real_return):
    metrics = {
        "fitness": fitness,
        "avg_fitness": fitness.mean(),
        "avg_last_return": returned_ep_real_returns.reshape(popsize, -1)
        .mean(axis=-1)
        .mean(),
        "max_last_return": returned_ep_real_returns.reshape(popsize, -1)
        .mean(axis=-1)
        .max(),
        "hist_last_return": wandb.Histogram(
            returned_ep_real_returns.reshape(popsize, -1).mean(axis=-1)
        ),
        "avg_ep_return": ep_real_return.reshape(popsize, -1).mean(-1).mean(),
        "max_ep_return": ep_real_return.reshape(popsize, -1).mean(-1).max(),
        "hist_ep_return": wandb.Histogram(ep_real_return.reshape(popsize, -1).mean(-1)),
    }
    wandb.log(step=gen, data=metrics)


def save_to_file(gen, rewards_path, cur_shaping, cur_state):
    if gen % 5 == 0:
        with open(rewards_path, "wb") as f:
            pickle.dump(
                {
                    "reward_avg": cur_shaping,
                    "state": cur_state,
                },
                f,
            )


def main(es_config=None):
    run = None
    if es_config is None:
        run = wandb.init(project="EvIL")
        es_config = wandb.config
        print("wandb run initialized: ", run.name)
    env, env_params, original_training_config = get_env(
        es_config["env"], es_config["backend"]
    )
    es_config, es_training_config = get_irl_config(es_config, original_training_config)
    if es_config["wandb_log"]:
        total_config = {**original_training_config, **es_config}
        print(original_training_config)
        if run is None:
            run = wandb.init(project="EvIL", config=total_config)
            print("wandb run initialized with all configs: ", run.name)
        else:
            print("wandb run updated")
            run.config.update(total_config)
        # create dir for saving rewards
        rewards_dir = f"{os.getcwd()}/shaped_rewards/{es_config['env']}"
        save_dir = None
        if es_config["save_to_file"]:
            if not os.path.exists(rewards_dir):
                os.mkdir(rewards_dir)
            save_dir = f"{os.getcwd()}/shaped_rewards/{es_config['env']}/{str(run.name).replace(' ', '_')}_recover_NN"
            print("save dir:", save_dir)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        es_config["save_dir"] = save_dir
    print("ORIG UPDATES", es_training_config["ORIG_NUM_UPDATES"])
    if original_training_config["DISCRETE"]:
        make_train_fn, eval_fn = make_train, eval
    else:
        make_train_fn, eval_fn = make_train_cont, eval_cont

    if es_training_config["DISCRETE"]:
        observation_shape = env.observation_space(env_params).shape[0]
        action_shape = env.action_space(env_params).n
    else:
        observation_shape = get_observation_size(env, env_params)[0]
        action_shape = env.action_size
    true_env = RewardWrapper(
        env=env,
        env_params=env_params,
        reward_network=None,
        rew_network_params=None,
        shaping_network=None,
        shap_network_params=None,
        training_config=original_training_config,
    )
    training_class = IRLPlus(
        env=env,
        env_params=env_params,
        training_config=es_training_config,
        es_config=es_config,
        logging_run=run,
        expert_data=(
            jnp.zeros((1, observation_shape)),
            jnp.zeros((1, action_shape)),
        ),
        shaping_net=None,
    )
    basic_reward_net = training_class._reward_network.network
    # if es_config["reward_net_ensemble_type"] == "avg":
    #     reward_net = RewardNetworkPenalty(basic_reward_net, 0.0)
    # elif es_config["reward_net_ensemble_type"] == "min":
    #     reward_net = RewardNetworkPessimistic(basic_reward_net)
    shaping_net = RewardNetwork(
        hsize=es_config["reward_net_hsize"],
        activation_fn=es_config["reward_net_activation_fn"],
        sigmoid=es_config["reward_net_sigmoid"],
    )
    if is_brax_env(es_config["env"]):
        expert_file = (
            f"{os.getcwd()}/experts/{es_config['env']}_{es_config['backend']}_1.pkl"
        )
    else:
        expert_file = f"{os.getcwd()}/experts/{es_config['env']}_1.pkl"
    if es_config["reward_filename"]:
        with open(f"{os.getcwd()}/{es_config['reward_filename']}", "rb") as f:
            saved_info = pickle.load(f)
            if es_config["reward_net_ensemble_params_type"] == "across_agents":
                rew_net_params = jax.tree_map(
                    lambda x: x[:, es_config["seed"] - 1],
                    saved_info.get("rew_net_params", None),
                )
            elif es_config["reward_net_ensemble_params_type"] == "same_agent":
                rew_net_params = jax.tree_map(
                    lambda x: x[es_config["seed"] - 1],
                    saved_info.get("rew_net_params", None),
                )
            data_stats = (
                saved_info.get("data_stats", None)[0],
                saved_info.get("data_stats", None)[1],
            )
            print(
                "data stats shape: ",
                data_stats[0].shape,
                "var: ",
                data_stats[1].shape,
            )
            print(
                "data stats mean: ",
                jnp.mean(data_stats[0]),
                "var: ",
                jnp.mean(data_stats[1]),
            )
    else:
        if es_config["obs_normalize"]:
            with open(expert_file, "rb") as f:
                expert_info = pickle.load(f)
                data_mean = expert_info["norm_mean"]
                data_var = expert_info["norm_var"]
                data_stats = (data_mean, data_var)
        else:
            data_stats = (
                jnp.zeros(observation_shape),
                jnp.zeros(observation_shape),
            )

    retrain_config = es_training_config.copy()
    retrain_config["NUM_UPDATES"] = es_config["num_updates_two_step"]

    def train_multi_seed(
        rng,
        rew_net_params,
        data_stats,
        shaping_params,
    ):
        multi_irl_reward_env = RewardWrapper(
            env=env,
            env_params=env_params,
            reward_network=reward_net,
            rew_network_params=rew_net_params,
            shaping_network=shaping_net,
            shap_network_params=shaping_params,
            include_action=False,
            training_config=retrain_config,
            invert_reward=True,
            data_stats=data_stats,
        )
        print("retraining config", retrain_config)
        multi_irl_agent_train = make_train_fn(
            config=retrain_config,
            env=multi_irl_reward_env,
            env_params=env_params,
            runner_state_start=None,
            log_timestep_returns=True,
            return_obs_and_actions=False,
        )
        return multi_irl_agent_train(rng)["metrics"]

    def es_loss(metrics):
        if is_irl(es_config):
            print("Optimizing for IRL AUC")
            if es_config["loss_last_only"]:
                return metrics["last_irl_return"].mean()
            else:
                # return metrics["avg_episode_irl_return"].mean()
                return metrics["irl_auc"].mean()
        else:
            print("Optimizing for Real AUC")
            print(es_config["loss_last_only"])
            # return metrics["avg_episode_real_return"].mean()
            if es_config["loss_last_only"]:
                return metrics["last_real_return"].mean()
            else:
                return metrics["real_auc"].mean()

    num_gpus = len(jax.devices())
    in_features = observation_shape
    shap_params = shaping_net.init(jax.random.PRNGKey(0), jnp.zeros(in_features))
    param_reshaper = evosax.ParameterReshaper(shap_params)
    rng_init = jax.random.PRNGKey(es_config["seed"])
    es_num_params = param_reshaper.total_params
    strategy = evosax.OpenES(
        popsize=es_config["popsize"],
        lrate_init=es_config["lrate_init"],
        num_dims=es_num_params,
        opt_name="adam",
        centered_rank=True,
        maximize=True,
    )
    es_params = strategy.default_params
    vmap_train_multi_seed = jax.vmap(train_multi_seed, (0, None, None, 0))
    vmap_loss = jax.vmap(es_loss)
    if num_gpus > 1:
        vmap_train_multi_seed = jax.pmap(
            vmap_train_multi_seed,
            in_axes=(0, None, None, 0),
            out_axes=0,
            devices=jax.devices(),
        )
        vmap_loss = jax.pmap(vmap_loss, devices=jax.devices())

    if not is_irl(es_config):
        rew_net_params = None
        reward_net = None

    def es_step(carry, unused):
        cur_state, rng, gen = carry
        rng_pop, rng_iter = jax.random.split(rng, 2)
        if es_config["train_rng"] == "SAME":
            rng_train = jnp.tile(jax.random.PRNGKey(0), reps=(es_config["popsize"], 1))
        elif es_config["train_rng"] == "DIFFERENT_IN_PAIRS":
            rng_pop = jax.random.split(rng_pop, es_config["popsize"] // 2)
            rng_train = jnp.tile(rng_pop, reps=(2, 1))
        else:
            raise ValueError("Invalid train rng")

        network_params, cur_state = strategy.ask(rng_iter, cur_state, es_params)
        # get shaping or reward parameters, for popsize
        network_params_tree = param_reshaper.reshape(network_params)
        if num_gpus > 1:
            rng_train = rng_train.reshape(num_gpus, -1, *rng_train.shape[1:])

        shaped_agents_out_metrics = vmap_train_multi_seed(
            rng_train, rew_net_params, data_stats, network_params_tree
        )

        pmap_fitness = vmap_loss(shaped_agents_out_metrics)
        fitness = pmap_fitness.reshape(-1)
        cur_state = strategy.tell(network_params, fitness, cur_state, es_params)
        return_ep_real_return = shaped_agents_out_metrics[
            "returned_episode_real_returns"
        ]
        avg_episode_real_return = shaped_agents_out_metrics["real_auc"]
        jax.debug.print(
            "Gen {g}, fit mean={f}, last mean={last}, ep mean={ep}",
            g=cur_state.gen_counter,
            f=fitness.mean(),
            last=return_ep_real_return.mean(),
            ep=avg_episode_real_return.mean(),
        )
        jax.debug.callback(
            wandb_callback,
            gen,
            es_config["popsize"],
            fitness,
            return_ep_real_return,
            avg_episode_real_return,
        )
        rewards_path = f"{os.getcwd()}/rewards/{es_config['env']}/{wandb.run.name}_{cur_state.gen_counter}.pkl"
        jax.debug.callback(
            save_to_file,
            cur_state.gen_counter,
            rewards_path,
            cur_state.mean,
            cur_state,
        )
        return (cur_state, rng, gen + 1), None

    rng_init_strategy, rng_init = jax.random.split(rng_init, 2)
    if es_config["restart_filename"]:
        state_path = f"{os.getcwd()}/rewards/{es_config['env']}/{es_config['restart_filename']}_{es_config['restart_gen']}.pkl"
        with open(state_path, "rb") as f:
            contents = pickle.load(f)
            init_es_state = contents["state"]
            start_gen = es_config["restart_gen"]
            new_opt = strategy.optimizer.initialize(es_params.opt_params)
            init_es_state = init_es_state.replace(opt_state=new_opt)
            print(
                "restarting from ", start_gen, "with state: ", init_es_state.gen_counter
            )
            print("opt lr", strategy.lrate_init, "es params", es_params)
    else:
        start_gen = 0
        init_es_state = strategy.initialize(rng_init_strategy, es_params)
    for gen in range(start_gen, es_config["generations"]):
        (init_es_state, rng_init, next_gen), _ = es_step(
            (init_es_state, rng_init, gen), None
        )

    wandb.finish()
    return
