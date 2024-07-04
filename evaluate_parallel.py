import os.path
import logging

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import joblib

import ray
from ray.rllib.algorithms.algorithm import Algorithm

from environment import SecretKeyEnv
from comparison import (
    constant_power_allocation,
    calculate_constant_power_level_balance,
    calculate_min_budget,
    max_power_when_below,
    adaptive_power_conditional_expectation,
    adaptive_power_budget_difference,
)
from rayleigh import pdf_skg_rate_rayleigh, sf_skg_rate_rayleigh
from ultimate_ruin_prob import calculate_ultimate_ruin_mixed

from util import to_decibel, db_to_linear, export_results

LOGGER = logging.getLogger(__name__)

def load_algorithms(env_config, **kwargs):
    algorithms = {}

    initial_state = None
    model_path = kwargs.get("model_path")
    if model_path is not None:
        ray.init(num_cpus=63, num_gpus=0)
        checkpoint_path = model_path
        LOGGER.info(f"Loading model from path: {checkpoint_path}")
        alg = Algorithm.from_checkpoint(checkpoint_path)
        alg.restore(checkpoint_path)
        policy = alg.get_policy()
        policy.config['explore'] = False
        if policy.is_recurrent():
            initial_state = policy.get_initial_state()
        #algorithms['model'] = policy.compute_single_action
        algorithms['model'] = alg.compute_single_action

    _power_level, _power_action = calculate_constant_power_level_balance(env_config)
    algorithms["constant"] = lambda state: constant_power_allocation(state, _power_action)
    algorithms["constant10dB"] = lambda state: constant_power_allocation(state, 10/env_config["max_power_db"])
    algorithms["constant30"] = lambda state: constant_power_allocation(state, to_decibel(30)/env_config["max_power_db"])
    algorithms["adaptiveCondExpect"] = lambda state: adaptive_power_conditional_expectation(state, env_config)
    algorithms["fullPower"] = lambda state: constant_power_allocation(state, 1)
    algorithms["lowPower"] = lambda state: constant_power_allocation(state, 0)
    return algorithms, initial_state

def main(test_config: dict):
    LOGGER.info("Updating config from training...")
    from configs import env_config
    env_config.update(test_config)
    env_config['init_budget'] = 70

    snr_bob_db = env_config.pop("snr_bob_db")
    snr_eve_db = env_config.pop("snr_eve_db")
    snr_bob_lin = db_to_linear(snr_bob_db)
    snr_eve_lin = db_to_linear(snr_eve_db)
    print(f"Main: {snr_bob_lin}, {snr_eve_lin}")
    env_config["rv_bob"] = stats.expon(scale=snr_bob_lin)
    env_config["rv_eve"] = stats.expon(scale=snr_eve_lin)
    LOGGER.info("Done.")

    LOGGER.info("Loading all algorithms...")
    model_path = env_config.pop("model_path")
    algorithms, initial_state = load_algorithms(env_config, model_path=model_path)
    LOGGER.info("Done.")

    LOGGER.info("Constructing objects...")
    num_runs = env_config.pop("num_runs")
    num_timesteps = env_config.pop("num_timesteps")
    timeline = np.arange(num_timesteps)

    plot = env_config.pop("plot")
    export = env_config.pop("export")
    LOGGER.info("Done.")


    def inner_loop(run, algorithm, num_timesteps, state=None):
        LOGGER.info(f"Run: {run+1:d}/{num_runs:d}")
        _rewards = np.nan*np.zeros(num_timesteps)
        _budgets = np.nan*np.zeros(num_timesteps)
        _actions = np.nan*np.zeros(num_timesteps)
        _powers = np.nan*np.zeros(num_timesteps)
        _alert_outage_probs = np.nan*np.zeros(num_timesteps)
        env = SecretKeyEnv(env_config, seed_id=run)
        obs, _ = env.reset(seed=run)
        prev_action = [1]
        prev_reward = 0.
        for time_slot in range(num_timesteps):
            # action = alg.compute_single_action(state)
            if state is None:
                action = algorithm(obs)
            else:
                action, state, _ = algorithm(obs, state, prev_action=prev_action,
                                             prev_reward=prev_reward)
            obs, reward, terminated, truncated, info = env.step(action)
            prev_action = action
            prev_reward = reward
            budget = obs["budget"][0]
            alert_outage_prob = info['alert_outage_prob'][0]
            _rewards[time_slot] = reward
            _budgets[time_slot] = budget
            _actions[time_slot] = action[0]
            if obs["is_transmission"]:
                _power = np.nan
            else:
                _power = 10**(action[0]*env_config["max_power_db"]/10.)
            _powers[time_slot] = _power
            _alert_outage_probs[time_slot] = alert_outage_prob
            if terminated or truncated:
                break
        return _rewards, _budgets, _actions, _powers, _alert_outage_probs


    rewards = {}
    budgets = {}
    actions = {}
    powers = {}
    average_powers = {}
    alert_outage_probs = {}
    for _name, _algorithm in algorithms.items():
        LOGGER.info(f"Algorithm: {_name}")
        #_rewards, _budgets, _actions, _powers, _alert_outage_probs = joblib.Parallel(n_jobs=joblib.cpu_count()//2)(
        if _name == "model":
            _results = [inner_loop(run, _algorithm, num_timesteps, state=initial_state)
                        for run in range(num_runs)]
        else:
            _backend = "threads" if _name == "model" else None
            _results = joblib.Parallel(n_jobs=joblib.cpu_count()//2, prefer=_backend)(
                    joblib.delayed(inner_loop)(run, _algorithm, num_timesteps)
                    for run in range(num_runs))
        _results = np.array(_results)
        rewards[_name] = _results[:, 0]
        budgets[_name] = _results[:, 1]
        actions[_name] = _results[:, 2]
        powers[_name] = _results[:, 3]
        alert_outage_probs[_name] = _results[:, 4]
    #print(budgets)

    #powers = 10 ** (actions * env_config["max_power_db"] / 10.0)
    # powers_db = actions*env_config['max_power_db']
    average_powers = {k: np.nancumsum(v, axis=1)/(np.arange(num_timesteps)+1-np.cumsum(np.isnan(v), axis=1))
                      for k, v in powers.items()}
    
    alert_outage_prob_violations = {k: np.nan_to_num(v, nan=1) > env_config['outage_prob_alert']
                                    for k, v in alert_outage_probs.items()}

    plots = ({"title": "Budget",
                "data": budgets,
                "xlabel": "Time $t$",
                "ylabel": "SK Budget",
                "yscale": "linear",
                "nan": 0.,
              "fname_prefix": "budget",
                },
                {"title": "Outage Probability when entering alert mode",
                "data": alert_outage_probs,
                "xlabel": "Time $t$",
                "ylabel": "Outage Probability",
                "yscale": "log",
                "nan": 1.,
              "fname_prefix": "alert_outage_prob",
                },
                {"title": "Resilience Outage Probability",
                "data": alert_outage_prob_violations,
                "xlabel": "Time $t$",
                "ylabel": "Resilience Outage Probability $\\alpha$",
                "yscale": "log",
                "nan": 1.,
              "fname_prefix": "resilience_outage_prob",
                },
                {"title": "Average Power",
                "data": average_powers,
                "xlabel": "Time $t$",
                "ylabel": "Average Transmit Power",
                "yscale": "log",
                "nan": np.nan,
              "fname_prefix": "avg_power",
                },
                {"title": "Power",
                "data": powers,
                "xlabel": "Time $t$",
                "ylabel": "Inst. Transmit Power",
                "yscale": "log",
                "nan": np.nan,
              "fname_prefix": "power",
                },
                )
    if plot:
        cmap = plt.colormaps["Set1"]

        max_power = 10 ** (env_config["max_power_db"] / 10.0)

        fig, axs = plt.subplots()
        axs.set_xscale("log")
        axs.set_yscale("log")
        for _idx, _name in enumerate(algorithms):
            _color = cmap(_idx)
            axs.plot(np.mean(average_powers[_name][:, -1]), np.mean(alert_outage_prob_violations[_name][:, -1]), 'o', label=_name, color=_color)
        axs.legend()
        axs.set_xlabel("Average Power")
        axs.set_ylabel("Resilience Outage Probability $\\alpha$")
        axs.set_xlim([1e-4, 1e4])
        axs.set_ylim([1e-5, 1])

        for _plot in plots:
            fig, axs = plt.subplots()
            axs.set_xlabel(_plot['xlabel'])
            axs.set_ylabel(_plot['ylabel'])
            axs.set_title(_plot['title'])
            axs.set_yscale(_plot['yscale'])
            values = _plot['data']
            for _idx, _name in enumerate(algorithms):
                _color = cmap(_idx)
                _values = values[_name]
                _values = np.nan_to_num(_values, nan=_plot.get("nan", 0))
                _values_mean = np.nanmean(_values, axis=0)
                axs.plot(timeline, _values_mean, c=_color, label=_name)
            axs.legend()

    if export:
        for _data_var in plots:
            _results = {k: np.nanmean(np.nan_to_num(v, nan=_data_var.get("nan", 0)), axis=0)
                        for k, v in _data_var["data"].items()}
            _results['time'] = timeline
            _fname = "{prefix}-p{prob_tx:.2f}-B{init_budget:d}-T{num_timesteps:d}.dat".format(prefix=_data_var['fname_prefix'], num_timesteps=num_timesteps, **env_config)
            export_results(_results, _fname)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-l", "--message_length", type=float, default=5)
    parser.add_argument("-b", "--snr_bob_db", type=float, default=10)
    parser.add_argument("-e", "--snr_eve_db", type=float, default=0)
    parser.add_argument("-p", "--prob_tx", type=float, default=0.2)
    #parser.add_argument("--max_power_db", type=float, default=30)
    parser.add_argument("-r", "--num_runs", type=int, default=100)
    parser.add_argument("-t", "--num_timesteps", type=int, default=1000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument(
        "-v", "--verbosity", action="count", default=0, help="Increase output verbosity"
    )
    args = vars(parser.parse_args())
    verb = args.pop("verbosity")
    logging.basicConfig(
        format="%(asctime)s - [%(levelname)8s]: %(message)s",
        handlers=[
            logging.FileHandler("main.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    loglevel = logging.WARNING - verb * 10
    LOGGER.setLevel(loglevel)
    main(args)
    ray.shutdown()
    plt.show()
