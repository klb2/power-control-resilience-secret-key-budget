import numpy as np
from scipy import optimize

from rayleigh import (
    expectation_skg_rate_rayleigh,
    expectation_skg_rate_rayleigh_conditional_bob,
)

from util import to_decibel, db_to_linear


def calculate_constant_power_level_balance(env_config):
    rv_bob = (
        env_config.get("rv_bob")
        if "rv_bob" in env_config
        else float(env_config.get("snr_bob_db"))
    )
    rv_eve = (
        env_config.get("rv_eve")
        if "rv_eve" in env_config
        else float(env_config.get("snr_eve_db"))
    )
    lam_bob = (
        1.0 / db_to_linear(rv_bob)
        if isinstance(rv_bob, (float, int))
        else 1.0 / rv_bob.std()
    )
    lam_eve = (
        1.0 / db_to_linear(rv_eve)
        if isinstance(rv_eve, (float, int))
        else 1.0 / rv_eve.std()
    )
    prob_tx = env_config["prob_tx"]
    message_length = env_config["message_length"]
    max_power_db = env_config["max_power_db"]
    max_power = 10 ** (max_power_db / 10.0)
    target_expect = prob_tx / (1 - prob_tx) * message_length

    def func_root_expect(power, lam_bob, lam_eve, target):
        return expectation_skg_rate_rayleigh(power, lam_bob, lam_eve) - target

    sol = optimize.root_scalar(
        func_root_expect,
        args=(lam_bob, lam_eve, target_expect),
        bracket=(0, max_power),
        x0=1,
    )
    power = sol.root
    power_action = to_decibel(power) / max_power_db
    return power, power_action


def adaptive_power_conditional_expectation(state, env_config):
    rv_eve = (
        env_config.get("rv_eve")
        if "rv_eve" in env_config
        else float(env_config.get("snr_eve_db"))
    )
    lam_eve = (
        1.0 / db_to_linear(rv_eve)
        if isinstance(rv_eve, (float, int))
        else 1.0 / rv_eve.std()
    )
    prob_tx = env_config["prob_tx"]
    message_length = env_config["message_length"]
    max_power_db = env_config["max_power_db"]
    max_power = 10 ** (max_power_db / 10.0)
    # target_expect = prob_tx/(1-prob_tx)*message_length
    channel_bob = state["channel_bob"][0]
    budget = state["budget"][0]
    min_budget = (
        env_config["rv_alert_duration"].ppf(1 - env_config["outage_prob_alert"])
        * env_config["message_length"]
    )
    budget_buffer = budget - min_budget
    if budget_buffer <= 0:
        power_action = 1
    else:

        def func_min_target(power, channel_bob, lam_eve, max_power, budget_buffer):
            # _part1 = expectation_skg_rate_rayleigh_conditional_bob(power, channel_bob, lam_eve)/expectation_skg_rate_rayleigh_conditional_bob(max_power, channel_bob, lam_eve)
            # _part2 = (power/max_power)**(1/budget_buffer)
            _part1 = np.log(
                expectation_skg_rate_rayleigh_conditional_bob(
                    power, channel_bob, lam_eve
                )
            )
            _part2 = 0.002 * budget_buffer * np.log(power)
            return -(_part1 - _part2)

        try:
            sol = optimize.minimize(
                func_min_target,
                args=(channel_bob, lam_eve, max_power, budget_buffer),
                bounds=((1, max_power),),
                x0=0.5 * max_power,
            )
            # power_db = sol.root
            # print(sol)
            power = sol.x[0]
            power_db = to_decibel(power)
            power_action = power_db / max_power_db
        except ValueError:
            power_action = -1
    action = np.array([power_action])
    return action


def constant_power_allocation(state, power_action):
    action = np.array([power_action])
    return action


def calculate_min_budget(env_config, safety_margin=0.2):
    rv_alert_duration = env_config["rv_alert_duration"]
    outage_prob_alert = env_config["outage_prob_alert"]
    message_length = env_config["message_length"]
    min_budget = rv_alert_duration.ppf(1 - outage_prob_alert) * message_length
    min_budget = (1 + safety_margin) * min_budget  # + message_length
    return min_budget


def max_power_when_below(state, min_budget):
    if state["budget"][0] < min_budget:
        action = np.array([1])
    else:
        action = np.array([-1])
    return action


def adaptive_power_budget_difference(state, min_budget, scale):
    budget_surplus = state["budget"][0] - min_budget
    action = np.clip(-budget_surplus / scale, -1, 1)
    action = np.array([action])
    return action
