import logging
from typing import Union, Iterable

import numpy as np
from scipy import stats

LOGGER = logging.getLogger(__name__)


def estimate_stop_times_samples(start_budget: float, acc_claims: Iterable):
    surplus = start_budget - acc_claims
    stop_times = np.argmax(surplus < 0, axis=1)
    num_timesteps = np.shape(acc_claims)[1]
    stop_times[np.logical_and(stop_times == 0, start_budget > 0)] = num_timesteps
    hist, bin_edges = np.histogram(
        stop_times, bins=np.arange(-0.5, num_timesteps + 1.5, 1), density=True
    )
    return bin_edges + 0.5, hist


def mc_stopping_time(
    rv: Union[stats.rv_continuous, stats.rv_discrete],
    num_samples: int = 100000,
    num_timesteps: int = 100,
    budget: float = 60.0,
):
    LOGGER.debug(f"Generating {num_samples:E} MC samples")
    samples_y = rv.rvs(size=(num_samples, num_timesteps - 1))
    LOGGER.debug(f"Generated all {num_samples:E} MC samples")
    samples_y = np.hstack((np.zeros((num_samples, 1)), samples_y))
    acc_claims = np.cumsum(samples_y, axis=1)

    _tau, _pdf_tau = estimate_stop_times_samples(budget, acc_claims)
    pdf = _pdf_tau[:-1]
    cdf = np.cumsum(pdf)
    return cdf, acc_claims
