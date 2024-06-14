import os
import pickle
from evil.irl.irl_multi_discr import IRLPlus
from evil.utils.parser import get_parser
from evil.utils.plot import plot

import wandb
from evil.utils.utils import (
    get_expert_obsv_and_actions,
    get_irl_config,
    get_plot_filename,
)
from evil.irl.irl import IRL
from evil.irl.rl import RL
from evil.utils.env_utils import get_env

from evil.utils.utils import LossType, RewardWrapper, generate_config
from evil.training.wrappers import BraxGymnaxWrapper

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import jax
import jax.numpy as jnp

print("Visible devices", jax.devices())

from evil.training.ppo_v2_cont_irl import (
    make_train as make_train_cont,
    eval as eval_cont,
)
from evil.training.ppo_v2_irl import make_train, eval
import seaborn as sns
import matplotlib.pyplot as plt

x_vel_dims = {
    "hopper": 5,
    "walker2d": 8,
    "ant": 13,
    "humanoid": 22,
}


def mutual_info(x, y):
    """
    Calculates the mutual information between two discrete datasets.

    Args:
        x: A 1D numpy array containing the first dataset.
        y: A 1D numpy array containing the second dataset.

    Returns:
        The mutual information between x and y.
    """
    _x_value, x_counts = jnp.unique(x, return_counts=True)
    _y_value, y_counts = jnp.unique(y, return_counts=True)
    _xy_value, xy_counts = jnp.unique(jnp.vstack([x, y]), axis=1, return_counts=True)

    n_samples = jnp.sum(x_counts)
    p_x = x_counts / n_samples
    p_y = y_counts / n_samples
    p_xy = xy_counts / n_samples

    entropy_x = -jnp.sum(p_x * jnp.log2(p_x + 1e-8))
    entropy_y = -jnp.sum(p_y * jnp.log2(p_y + 1e-8))
    entropy_xy = -jnp.sum(p_xy * jnp.log2(p_xy + 1e-8))

    mi = entropy_x + entropy_y - entropy_xy
    return mi


def run_irl(es_config=None):
    run = None
    if es_config is None:
        run = wandb.init(project="EvIL")
        es_config = wandb.config
        print("wandb run initialized: ", run.name)
    env, env_params, original_training_config = get_env(
        es_config["env"], es_config["backend"]
    )
    es_config, es_training_config = get_irl_config(es_config, original_training_config)
    if es_config["irl_plus"]:
        irl_method = IRLPlus
    else:
        irl_method = IRL
        es_config["discr_l2_loss"] = 0.0
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
        rewards_dir = f"{os.getcwd()}/rewards"
        save_dir = None
        if es_config["save_to_file"]:
            if not os.path.exists(rewards_dir):
                os.mkdir(rewards_dir)
            save_dir = (
                f"{os.getcwd()}/rewards/{str(run.name).replace(' ', '_')}_recover_NN"
            )
            print("save dir:", save_dir)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        es_config["save_dir"] = save_dir
    # train base agent
    print("ORIG UPDATES", es_training_config["ORIG_NUM_UPDATES"])
    print("ES TRAINING", es_training_config)
    if original_training_config["DISCRETE"]:
        make_train_fn, eval_fn = make_train, eval
    else:
        make_train_fn, eval_fn = make_train_cont, eval_cont

    # TRAIN EXPERT AGENT
    true_env = RewardWrapper(
        env=env,
        env_params=env_params,
        reward_network=None,
        rew_network_params=None,
        shaping_network=None,
        shap_network_params=None,
        training_config=original_training_config,
    )

    # GET EXPERT DATA
    (
        expert_obsv,
        expert_actions,
        _,
        original_returns,
        last_return,
        train_expert_norm_stats,
    ) = get_expert_obsv_and_actions(
        true_env,
        env_params,
        es_config,
        make_train_fn,
        eval_fn,
        original_training_config,
    )
    print("Expert last return", last_return)
    rng = jax.random.PRNGKey(es_config["seed"])

    if LossType[es_config["loss"]] == LossType.IRL:
        training_class = irl_method(
            env=env,
            env_params=env_params,
            training_config=es_training_config,
            es_config=es_config,
            logging_run=run,
            expert_data=(expert_obsv, expert_actions, train_expert_norm_stats),
            shaping_net=None,
        )
        if jax.device_count() > 1:
            rng = jax.random.split(rng, jax.device_count())
            print("IRL training with pmap", rng.shape)
            (
                (last_runner_state, last_discr, data_stats),
                irl_train_metrics,
                buffer_state,
            ) = jax.pmap(training_class.train, in_axis=(0, None, None, None))(
                rng, env_params, None, True
            )
            last_discr = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), last_discr)
            irl_train_metrics = jax.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]), irl_train_metrics
            )
            data_stats = train_expert_norm_stats
            buffer_state = jax.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]), buffer_state
            )
        else:
            (
                (last_runner_state, last_discr, data_stats),
                irl_train_metrics,
                buffer_state,
            ) = training_class.train(rng=rng, env_params=env_params, return_buffer=True)
        if es_config["plot"]:
            sz = (
                es_config["num_updates_inner_loop"]
                * es_training_config["NUM_STEPS"]
                * es_training_config["NUM_ENVS"],
            )
            steps = es_config["irl_generations"]
            filename_plot = get_plot_filename(es_config)
            plot(
                es_config["env"],
                irl_train_metrics["returned_episode_real_returns"],
                last_return,
                steps,
                sz,
                filename_plot,
            )
        rew_net_params = last_discr.params
        reward_net = training_class._reward_network
    else:
        raise NotImplementedError("Loss not supported")

    if es_config["save_to_file"]:
        print(f"saving reward net and data stats to {save_dir}/reward_net_params.pkl")
        with open(f"{save_dir}/reward_net_params.pkl", "wb") as f:
            pickle.dump(
                {
                    "rew_net_params": rew_net_params,
                    "data_stats": data_stats,
                    # "data_stats_runner_state": (
                    #     last_runner_state[1].mean,
                    #     last_runner_state[1].var,
                    # ),
                    "seeds": es_config["seeds"],
                },
                f,
            )
    rew_net_params = jax.tree_map(lambda x: x, rew_net_params)
    buffer_x_pos = buffer_state.buffer_obsv.reshape(-1, expert_obsv.shape[-1])[
        ::50, x_vel_dims[es_config["env"]]
    ].reshape(-1)
    expert_x_pos = expert_obsv.reshape(-1, expert_obsv.shape[-1])[
        ::50, x_vel_dims[es_config["env"]]
    ].reshape(-1)
    exp_rew_across = -jax.vmap(reward_net.apply, (1, None), 0)(
        rew_net_params,
        (expert_obsv.reshape(-1, expert_obsv.shape[-1])[::50] - data_stats[0])
        / jnp.sqrt(data_stats[1] + 1e-8),
    )
    buffer_rew_across = -jax.vmap(reward_net.apply, (1, None), 0)(
        rew_net_params,
        (
            buffer_state.buffer_obsv.reshape(-1, expert_obsv.shape[-1])[::50]
            - data_stats[0]
        )
        / jnp.sqrt(data_stats[1] + 1e-8),
    )
    exp_rew_same = -jax.vmap(reward_net.apply, (0, None), 0)(
        rew_net_params,
        (expert_obsv.reshape(-1, expert_obsv.shape[-1])[::50] - data_stats[0])
        / jnp.sqrt(data_stats[1] + 1e-8),
    )
    buffer_rew_same = -jax.vmap(reward_net.apply, (0, None), 0)(
        rew_net_params,
        (
            buffer_state.buffer_obsv.reshape(-1, expert_obsv.shape[-1])[::50]
            - data_stats[0]
        )
        / jnp.sqrt(data_stats[1] + 1e-8),
    )
    exp_rew_across = exp_rew_across.reshape(es_config["seeds"], -1)
    buffer_rew_across = buffer_rew_across.reshape(es_config["seeds"], -1)
    exp_rew_same = exp_rew_same.reshape(es_config["seeds"], -1)
    buffer_rew_same = buffer_rew_same.reshape(es_config["seeds"], -1)
    correlation_coeff_across = jnp.zeros(es_config["seeds"])
    correlation_coeff_same = jnp.zeros(es_config["seeds"])
    correlation_exp_across = jnp.zeros(es_config["seeds"])
    correlation_exp_same = jnp.zeros(es_config["seeds"])
    correlation_buffer_across = jnp.zeros(es_config["seeds"])
    correlation_buffer_same = jnp.zeros(es_config["seeds"])
    mi_across = jnp.zeros(es_config["seeds"])
    mi_same = jnp.zeros(es_config["seeds"])
    for i in range(es_config["seeds"]):
        buffer_same_corr_coef = jnp.corrcoef(buffer_rew_same[i], buffer_x_pos)[0, 1]
        exp_same_corr_coef = jnp.corrcoef(exp_rew_same[i], expert_x_pos)[0, 1]
        buffer_across_corr_coef = jnp.corrcoef(buffer_rew_across[i], buffer_x_pos)[0, 1]
        exp_across_corr_coef = jnp.corrcoef(exp_rew_across[i], expert_x_pos)[0, 1]
        same_rew = jnp.concatenate((buffer_rew_same[i], exp_rew_same[i]), axis=0)
        across_rew = jnp.concatenate((buffer_rew_across[i], exp_rew_across[i]), axis=0)
        x_pos = jnp.concatenate((buffer_x_pos, expert_x_pos), axis=0)
        all_same_corr_coef = jnp.corrcoef(same_rew, x_pos)[0, 1]
        all_across_corr_coef = jnp.corrcoef(across_rew, x_pos)[0, 1]
        wandb.log(
            {
                f"buffer_same_corr_coef_{i}": buffer_same_corr_coef,
                f"exp_same_corr_coef_{i}": exp_same_corr_coef,
                f"buffer_across_corr_coef_{i}": buffer_across_corr_coef,
                f"exp_across_corr_coef_{i}": exp_across_corr_coef,
                f"all_same_corr_coef_{i}": all_same_corr_coef,
                f"all_across_corr_coef_{i}": all_across_corr_coef,
            }
        )
        correlation_coeff_across = correlation_coeff_across.at[i].set(
            all_across_corr_coef
        )
        correlation_coeff_same = correlation_coeff_same.at[i].set(all_same_corr_coef)
        correlation_exp_across = correlation_exp_across.at[i].set(exp_across_corr_coef)
        correlation_exp_same = correlation_exp_same.at[i].set(exp_same_corr_coef)
        correlation_buffer_across = correlation_buffer_across.at[i].set(
            buffer_across_corr_coef
        )
        correlation_buffer_same = correlation_buffer_same.at[i].set(
            buffer_same_corr_coef
        )
        # calculate MI
        expert_x_bin = jnp.digitize(expert_x_pos, jnp.linspace(-10, 10, 20))
        buffer_x_bin = jnp.digitize(buffer_x_pos, jnp.linspace(-10, 10, 20))
        expert_same_rew_bin = jnp.digitize(exp_rew_same[i], jnp.linspace(0, 1, 20))
        buffer_same_rew_bin = jnp.digitize(buffer_rew_same[i], jnp.linspace(0, 1, 20))
        expert_across_rew_bin = jnp.digitize(exp_rew_across[i], jnp.linspace(0, 1, 20))
        buffer_across_rew_bin = jnp.digitize(
            buffer_rew_across[i], jnp.linspace(0, 1, 20)
        )
        all_bin_x = jnp.concatenate((expert_x_bin, buffer_x_bin), axis=0)
        all_same_rew = jnp.concatenate(
            (expert_same_rew_bin, buffer_same_rew_bin), axis=0
        )
        all_across_rew = jnp.concatenate(
            (expert_across_rew_bin, buffer_across_rew_bin), axis=0
        )

        exp_across_mi = mutual_info(expert_across_rew_bin, expert_x_bin)
        buffer_across_mi = mutual_info(buffer_across_rew_bin, buffer_x_bin)
        exp_same_mi = mutual_info(expert_same_rew_bin, expert_x_bin)
        buffer_same_mi = mutual_info(buffer_same_rew_bin, buffer_x_bin)
        all_same_mi = mutual_info(all_same_rew, all_bin_x)
        all_across_mi = mutual_info(all_across_rew, all_bin_x)
        wandb.log(
            {
                f"exp_across_mi_{i}": exp_across_mi,
                f"buffer_across_mi_{i}": buffer_across_mi,
                f"exp_same_mi_{i}": exp_same_mi,
                f"buffer_same_mi_{i}": buffer_same_mi,
                f"all_same_mi_{i}": all_same_mi,
                f"all_across_mi_{i}": all_across_mi,
            }
        )
        mi_across = mi_across.at[i].set(all_across_mi)
        mi_same = mi_same.at[i].set(all_same_mi)

    wandb.log(
        {
            f"avg_all_same_corr_coef": jnp.mean(correlation_coeff_same),
            f"min_all_same_corr_coef": jnp.min(correlation_coeff_same),
            f"max_all_same_corr_coef": jnp.max(correlation_coeff_same),
            f"avg_all_across_corr_coef": jnp.mean(correlation_coeff_across),
            f"min_all_across_corr_coef": jnp.min(correlation_coeff_across),
            f"max_all_across_corr_coef": jnp.max(correlation_coeff_across),
            "avg_correlation_exp_across": jnp.mean(correlation_exp_across),
            "min_correlation_exp_across": jnp.min(correlation_exp_across),
            "max_correlation_exp_across": jnp.max(correlation_exp_across),
            "avg_correlation_exp_same": jnp.mean(correlation_exp_same),
            "min_correlation_exp_same": jnp.min(correlation_exp_same),
            "max_correlation_exp_same": jnp.max(correlation_exp_same),
            "avg_correlation_buffer_across": jnp.mean(correlation_buffer_across),
            "min_correlation_buffer_across": jnp.min(correlation_buffer_across),
            "max_correlation_buffer_across": jnp.max(correlation_buffer_across),
            "avg_correlation_buffer_same": jnp.mean(correlation_buffer_same),
            "min_correlation_buffer_same": jnp.min(correlation_buffer_same),
            "max_correlation_buffer_same": jnp.max(correlation_buffer_same),
            f"avg_mi_same": jnp.mean(mi_same),
            f"min_mi_same": jnp.min(mi_same),
            f"max_mi_same": jnp.max(mi_same),
            f"avg_mi_across": jnp.mean(mi_across),
            f"min_mi_across": jnp.min(mi_across),
            f"max_mi_across": jnp.max(mi_across),
        }
    )

    es_training_config["NUM_UPDATES"] = int(
        original_training_config["ORIG_NUM_UPDATES"]
    )
    # RETRAIN ON REWARD FROM SCRATCH
    irl_returns = jnp.zeros((es_config["seeds"], es_training_config["NUM_UPDATES"]))
    if es_config["retrain_from_scratch"]:
        es_config["generations"] = es_training_config["NUM_UPDATES"]
        env = BraxGymnaxWrapper(
            env_name=es_config["env"], backend=es_config["backend_test"]
        )
        rl_training_class = RL(
            env,
            training_config=es_training_config,
            outer_config=es_config,
            logging_run=None,
            env_params=env_params,
            reward_net=reward_net,
            shaping_net=None,
        )
        if (
            es_config["reward_net_ensemble_params_type"] == "across_agents"
            or es_config["reward_net_ensemble_params_type"] == "both"
        ):
            rng = jax.random.PRNGKey(es_config["seed"])
            across_agents_rng = jax.random.split(rng, es_config["num_discr"])
            last_runner_state, metrics = jax.jit(
                jax.vmap(rl_training_class.train, (0, 1, 0, None), axis_name="i")
            )(
                across_agents_rng,
                rew_net_params,
                None,
                data_stats,
            )
            irl_returns = metrics["timestep_returned_episode_returns"]
            if es_config["wandb_log"]:
                for t in range(0, irl_returns.shape[1], 2):
                    metrics = {}
                    metrics["avg_irl_training_across_agents"] = jnp.mean(
                        irl_returns[:, t]
                    )
                    metrics["std_err_up_irl_training_across_agents"] = jnp.mean(
                        irl_returns[:, t]
                    ) + (
                        jnp.std(irl_returns[:, t], axis=0)
                        / jnp.sqrt(irl_returns.shape[0])
                    )
                    metrics["std_err_down_irl_training_across_agents"] = jnp.mean(
                        irl_returns[:, t]
                    ) - (
                        jnp.std(irl_returns[:, t], axis=0)
                        / jnp.sqrt(irl_returns.shape[0])
                    )
                    wandb.log(data=metrics)
        if (
            es_config["reward_net_ensemble_params_type"] == "same_agent"
            or es_config["reward_net_ensemble_params_type"] == "both"
        ):
            rng = jax.random.PRNGKey(es_config["seed"])
            multi_seed_rng = jax.random.split(rng, es_config["seeds"])
            last_runner_state, metrics = jax.jit(
                jax.vmap(rl_training_class.train, (0, 0, 0, None), axis_name="i")
            )(
                multi_seed_rng,
                rew_net_params,
                None,
                data_stats,
            )
            irl_returns = metrics["timestep_returned_episode_returns"]
            if es_config["wandb_log"]:
                for t in range(0, irl_returns.shape[1], 2):
                    metrics = {}
                    metrics["avg_irl_training_same_agent"] = jnp.mean(irl_returns[:, t])
                    metrics["std_err_up_irl_training_same_agent"] = jnp.mean(
                        irl_returns[:, t]
                    ) + (
                        jnp.std(irl_returns[:, t], axis=0)
                        / jnp.sqrt(irl_returns.shape[0])
                    )
                    metrics["std_err_down_irl_training_same_agent"] = jnp.mean(
                        irl_returns[:, t]
                    ) - (
                        jnp.std(irl_returns[:, t], axis=0)
                        / jnp.sqrt(irl_returns.shape[0])
                    )
                    wandb.log(data=metrics)

    wandb.log(
        data={
            "return_and_retrain": jnp.mean(irl_returns[:, -1])
            + jnp.mean(irl_train_metrics["returned_episode_real_returns"][:, -1])
        }
    )

    del training_class.buffer
    del training_class
    del rew_net_params
    del expert_obsv
    del expert_actions
    del original_returns
    del last_return
    del train_expert_norm_stats
    wandb.finish()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    for x in args.seed:
        es_config = generate_config(args, int(x))
        run_irl(es_config)
