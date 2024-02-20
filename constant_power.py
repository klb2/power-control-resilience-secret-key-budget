import os
import logging
from typing import Iterable

import numpy as np
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt

from rayleigh import (
    sf_skg_rate_rayleigh,
    pdf_skg_rate_rayleigh,
    logpdf_skg_rate_rayleigh,
    expectation_skg_rate_rayleigh,
)
from mc_stopping_time import mc_stopping_time
from ide_stopping_time import ide_conv
from ultimate_ruin_prob import calculate_ultimate_ruin_mixed

from util import (
    export_results,
    find_closest_element_idx,
    db_to_linear,
    capacity,
    clip_ruined_budget,
    calculate_alert_outage_probability,
    to_decibel,
)

LOGGER = logging.getLogger(__name__)


RESULTS_DIR = "results"


class RVNetClaim(stats.rv_continuous):
    def __init__(self, lam_x, lam_y, message_length, prob_tx, **kwargs):
        super().__init__(**kwargs)
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.message_length = message_length
        self.prob_tx = prob_tx

    def _cdf(self, x):
        _part1 = self.prob_tx * np.heaviside(x - self.message_length, 1)
        _part2 = (1 - self.prob_tx) * (sf_skg_rate_rayleigh(-x, self.lam_x, self.lam_y))
        return _part1 + _part2


class RVNetClaimFast(stats.rv_continuous):
    def __init__(self, rv_bob, rv_eve, message_length, prob_tx, **kwargs):
        super().__init__(**kwargs)
        self.rv_bob = rv_bob
        self.rv_eve = rv_eve
        self.message_length = message_length
        self.prob_tx = prob_tx

    def _rvs(self, *args, size=None, random_state=None):
        samples_x = self.rv_bob.rvs(size=size)
        samples_y = self.rv_eve.rvs(size=size)

        rate_sum = capacity(samples_x + samples_y)
        rate_eve = capacity(samples_y)
        samples_skg = -(rate_sum - rate_eve)
        selected_block = np.random.choice(
            2, p=[1 - self.prob_tx, self.prob_tx], size=size
        )
        samples_net_claims = np.where(selected_block, self.message_length, samples_skg)
        return samples_net_claims


def cdf_sum_conv(
    s,
    pdf_skg,
    sf_skg,
    max_s,
    message_length,
    prob_tx,
    num_timesteps: int = 100,
    num_points=2**16,
):
    z, dz = np.linspace(-max_s, 2 * max_s, num_points, retstep=True)
    zz = z
    num_z1 = len(z)
    first_idx = int(np.abs(min(zz)) / dz)

    pdf_tx = np.zeros(num_z1)
    pdf_tx[find_closest_element_idx(z, message_length)] = 1.0
    # _pdf_skg = np.exp(logpdf_skg_rate_rayleigh(-z, lam_x, lam_y))
    _pdf_skg = pdf_skg(-z)
    _factor_pdf_skg = 1.0 / (dz * sum(_pdf_skg))
    pdf_z1 = prob_tx * pdf_tx + (1 - prob_tx) * _factor_pdf_skg * dz * _pdf_skg
    cdf_z2 = prob_tx * np.heaviside(z - message_length, 1) + (1 - prob_tx) * sf_skg(-z)

    idx_s = find_closest_element_idx(zz, s)
    cdf_sum = [cdf_z2[idx_s]]
    for i in range(1, num_timesteps):
        LOGGER.debug(f"Calculating netsum CDF for time step {i+1:d}/{num_timesteps:d}")
        cdf_z2 = signal.fftconvolve(pdf_z1, cdf_z2, mode="full")  # /len(s)
        cdf_z2 = cdf_z2[first_idx : first_idx + num_z1]
        cdf_sum.append(cdf_z2[idx_s])
    cdf_sum = np.array(cdf_sum)
    cdf_sum = np.ravel(cdf_sum)
    return cdf_sum


def main(
    init_budget: float,
    power_tx_db: float = 0,
    snr_bob_db: float = 10,
    snr_eve_db: float = 0,
    prob_tx: float = 0.5,
    message_length: float = 5,
    target_alert_outage_prob: float = 0.1,
    mean_alert_duration: int = 5,
    num_timesteps: int = 150,
    num_budgets: int = 200,
    num_samples: int = int(1e6),
    skip_mc: bool = False,
    plot: bool = False,
    export: bool = False,
):
    LOGGER.info("Starting simulation...")
    LOGGER.debug(f"Number of MC samples: {num_samples:E}")
    LOGGER.debug(f"Number of timesteps: {num_timesteps:d}")
    LOGGER.debug(f"Number of budgets: {num_budgets:d}")

    power_tx = db_to_linear(power_tx_db)
    snr_bob_lin = db_to_linear(snr_bob_db)
    snr_eve_lin = db_to_linear(snr_eve_db)
    snr_bob_lin = snr_bob_lin * power_tx
    snr_eve_lin = snr_eve_lin * power_tx
    rv_bob = stats.expon(scale=snr_bob_lin)
    rv_eve = stats.expon(scale=snr_eve_lin)
    LOGGER.info(f"Tx Power: {power_tx_db:.1f} dB")
    LOGGER.info(f"Avg. Total SNR Bob: {to_decibel(snr_bob_lin):.1f} dB")
    LOGGER.info(f"Avg. Total SNR Eve: {to_decibel(snr_eve_lin):.1f} dB")

    LOGGER.debug("Determining the density of the claims")
    _num_samples_rv = int(1e5)  # 1e5 # for KDE
    rv_net_claims_fast = RVNetClaimFast(rv_bob, rv_eve, message_length, prob_tx)
    samples_net_claims = rv_net_claims_fast.rvs(size=_num_samples_rv)
    mean_claim = np.mean(samples_net_claims)
    LOGGER.info(f"Average claim (samples): {mean_claim:.3f} bit")

    # _hist = np.histogram(samples_net_claims, bins=300)
    # rv = stats.rv_histogram(_hist)
    # rv = RVNetClaim(1.0 / snr_bob_lin, 1.0 / snr_eve_lin, message_length, prob_tx)
    mean_skg_calc = expectation_skg_rate_rayleigh(
        power_tx, power_tx / snr_bob_lin, power_tx / snr_eve_lin
    )
    crit_prob_tx = mean_skg_calc / (mean_skg_calc + message_length)
    mean_claim_calc = -(1 - prob_tx) * mean_skg_calc + prob_tx * message_length
    LOGGER.info(f"Average claim (analytical): {mean_claim_calc:.3f} bit")
    LOGGER.info(f"Critical p: {crit_prob_tx:.3f}")

    rv_alert_duration = stats.poisson(mu=mean_alert_duration)
    _min_budget_alert = (
        rv_alert_duration.ppf(1 - target_alert_outage_prob) * message_length
    )
    _budget_gap = init_budget - _min_budget_alert
    LOGGER.info(f"Budget gap for resilience: {_budget_gap:.3f}")

    if not skip_mc:
        LOGGER.info("Working on the Monte Carlo simulation...")
        cdf_ruin_mc, acc_claims_mc = mc_stopping_time(
            rv_net_claims_fast,
            budget=init_budget,
            num_samples=num_samples,
            num_timesteps=num_timesteps,
        )
        acc_budgets_mc = init_budget - acc_claims_mc
        acc_budgets_mc = clip_ruined_budget(acc_budgets_mc)
        alert_outage_prob_mc = calculate_alert_outage_probability(
            acc_budgets_mc, message_length, rv_alert_duration
        )
        _sort_acc_claims = np.sort(acc_claims_mc, axis=0)
        prob_acc_claims_budget_gap = np.apply_along_axis(
            np.searchsorted, 0, _sort_acc_claims, _budget_gap
        ) / len(_sort_acc_claims)
    else:
        LOGGER.info("Skipping Monte Carlo simulation...")
        # budget_mc = np.linspace(0, max_budget, num_budgets)
        cdf_ruin_mc = np.zeros(num_timesteps)
        # alert_outage_prob = np.ones((num_timesteps, num_budgets))
    LOGGER.info("Finished the Monte Carlo simulation...")

    timeline = np.arange(num_timesteps)

    LOGGER.info("Calculating the ruin probabilities...")
    # _pdf_skg = lambda s: pdf_skg_rate_rayleigh(-s, 1.0 / snr_bob_lin, 1.0 / snr_eve_lin)
    _pdf_skg = lambda s: np.exp(
        logpdf_skg_rate_rayleigh(s, 1.0 / snr_bob_lin, 1.0 / snr_eve_lin)
    )
    _sf_skg = lambda s: sf_skg_rate_rayleigh(s, 1.0 / snr_bob_lin, 1.0 / snr_eve_lin)
    budget_ruin, prob_ruin_calc = ide_conv(
        _pdf_skg,
        prob_tx,
        message_length,
        max_x=3 * init_budget,
        num_points=2**20,
        num_timesteps=num_timesteps,
    )
    prob_ruin_calc = prob_ruin_calc[:, np.argmin((budget_ruin - init_budget) ** 2)]
    prob_sum_budget = cdf_sum_conv(
        _budget_gap,
        _pdf_skg,
        _sf_skg,
        np.maximum(np.abs(mean_claim_calc) * num_timesteps, 1.5 * _budget_gap),
        message_length=message_length,
        prob_tx=prob_tx,
        num_timesteps=num_timesteps,
    )
    LOGGER.info("Calculating the probability of ultimate ruin")
    if mean_claim_calc >= 0:  # prob_tx >= crit_prob_tx:
        ult_ruin_prob_calc = 1.0
    else:
        budget_ult_ruin, ult_ruin_prob_calc = calculate_ultimate_ruin_mixed(
            _pdf_skg,
            _sf_skg,
            message_length=message_length,
            prob_tx=prob_tx,
            max_b=3 * init_budget,
            num_points=9000,
        )
        # max_b=5*init_budget, num_points=5000)
        ult_ruin_prob_calc = ult_ruin_prob_calc[
            np.argmin((budget_ult_ruin - init_budget) ** 2)
        ]
        assert ult_ruin_prob_calc >= 0
        ult_ruin_prob_calc = np.clip(ult_ruin_prob_calc, 0, 1)
    LOGGER.info(f"Probability of ultimate ruin: {ult_ruin_prob_calc:E}")
    LOGGER.info("Finished all calculations.")

    if plot:
        fig, axs = plt.subplots()
        axs.set_xlabel("Time Step")
        axs.semilogy(timeline, cdf_ruin_mc, label="Ruin Probability -- Monte Carlo")
        axs.semilogy(
            timeline,
            alert_outage_prob_mc,
            label="Alert Outage Probability -- Monte Carlo",
        )
        axs.semilogy(timeline, prob_ruin_calc, label="Ruin Probability -- IDE")
        axs.semilogy(
            [min(timeline), max(timeline)],
            ult_ruin_prob_calc * np.ones(2),
            ls="--",
            label="Ultimate Ruin Probability",
        )
        axs.legend()

    if export:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results = {
            "time": timeline,
            "ruinMC": cdf_ruin_mc,
            "outageMC": alert_outage_prob_mc,
            "ruinIDE": prob_ruin_calc,
            "ultimateRuin": ult_ruin_prob_calc * np.ones_like(timeline),
        }
        fname = f"results-p{prob_tx:.2f}-P{power_tx_db:.1f}-b{init_budget:.1f}-e{target_alert_outage_prob:E}.dat"
        export_results(results, os.path.join(RESULTS_DIR, fname))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--init_budget", type=float)
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=int(1e6),
        help="Number of MC samples used for the simulation",
    )
    parser.add_argument("-T", "--num_timesteps", type=int, default=150)
    parser.add_argument("-P", "--power_tx_db", type=float, default=0)
    parser.add_argument("-p", "--prob_tx", type=float, default=0.5)
    parser.add_argument("-mu", "--mean_alert_duration", type=int, default=5)
    parser.add_argument("--skip_mc", action="store_true")
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
    main(**args)
    plt.show()
