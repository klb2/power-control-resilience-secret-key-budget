import logging
from typing import Callable

import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt

import mc_stopping_time

LOGGER = logging.getLogger(__name__)


def calculate_ultimate_ruin_mixed(
    pdf_skg,
    sf_skg,
    message_length: float,
    prob_tx: float,
    max_b: float,
    num_points: int = 100,
):
    # t, dt = np.linspace(0, max_b, num_points, retstep=True)
    idx_offset = int(np.ceil(message_length * (num_points + 1) / max_b))
    dt = message_length / idx_offset
    t = np.arange(0, max_b + dt, step=dt)
    if not np.isclose(idx_offset, round(idx_offset)):
        raise ValueError(
            "The message length should be an integer multiple of the stepsize"
        )
    num_points = len(t)
    t_mat = t - np.reshape(t, (-1, 1))
    # t_mat = np.reshape(t, (-1, 1)) - t  # b-s

    kernel_func = pdf_skg  # rv_skg.pdf
    kernel_mat = kernel_func(t_mat)
    kernel_mat[:, 1:-1] *= 2  # Trapezoidal rule
    kernel_mat = dt / 2 * kernel_mat

    offset_matrix = np.eye(num_points, k=-idx_offset)

    inhomog = 1.0 - (
        (1 - prob_tx) * sf_skg(-t) + prob_tx * np.heaviside(t - message_length, 1)
    )

    solve_matrix = (
        np.eye(num_points) - (1 - prob_tx) * kernel_mat - prob_tx * offset_matrix
    )

    outage_prob = scipy.linalg.solve(solve_matrix, inhomog)
    return t, outage_prob


if __name__ == "__main__":
    from rayleigh import pdf_skg_rate_rayleigh, sf_skg_rate_rayleigh

    rv_skg = stats.expon(scale=1)
    # rv_skg = stats.gamma(3)
    # rv_skg = stats.expon(scale=.5)
    # rv_skg = stats.norm(loc=0)
    # rv_skg = stats.norm(loc=0.1)
    pdf = rv_skg.pdf
    sf = rv_skg.sf
    lam_x = 10 ** (-10.0 / 10)
    lam_y = 10 ** (-0 / 10.0)
    pdf = lambda x: pdf_skg_rate_rayleigh(x, lam_x, lam_y)
    sf = lambda x: sf_skg_rate_rayleigh(x, lam_x, lam_y)
    message_length = 5
    prob_tx = 0.7
    b, outage_prob = calculate_ultimate_ruin_mixed(
        pdf, sf, message_length, prob_tx, num_points=2000, max_b=200
    )
    if False:
        rv = RVNetClaim(lam_x, lam_y, message_length, prob_tx)
        _b, _cdf = mc_stopping_time.mc_stopping_time(
            rv, num_samples=1000, num_timesteps=10, num_budgets=2000, max_budget=200
        )
        print(_cdf)
    else:
        _b = [0, 200]
        _cdf = [[1, 1]]
    # rv = RVNetClaim(lam_x, lam_y, message_length, prob_tx)
    # _samples = rv.rvs(size=10000)
    # plt.hist(_samples, bins=100, density=True)
    # s = np.linspace(min(_samples), 0, 200)
    # plt.plot(s, (1-prob_tx)*pdf_skg_rate_rayleigh(-s, lam_x, lam_y))
    plt.figure()
    plt.plot(b, outage_prob)
    plt.plot(_b, _cdf[-1])
    plt.ylim([-0.05, 1.05])
    plt.show()
