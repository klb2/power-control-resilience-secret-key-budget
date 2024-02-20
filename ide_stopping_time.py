from typing import Callable, Union, Iterable
import logging

import numpy as np
from scipy import signal

from util import find_closest_element_idx

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def ide_conv(
    pdf_skg: Callable,
    prob_tx: float,
    message_length: float,
    max_x: float = 60,
    num_timesteps: int = 25,
    num_points: int = 2**16,
):
    z, dz = np.linspace(-2 * max_x, 2 * max_x, num_points, retstep=True)
    zz = z
    num_z1 = len(z)
    first_idx = int(np.abs(min(zz)) / dz)
    _idx_budget = np.where(np.logical_and(0 <= z, z <= max_x))[0]

    pdf_tx = np.zeros(num_z1)
    pdf_tx[find_closest_element_idx(z, message_length)] = 1.0
    # _pdf_skg = np.exp(logpdf_skg_rate_rayleigh(-z, lam_x, lam_y))
    _pdf_skg = pdf_skg(-z)
    _factor_pdf_skg = 1.0 / (dz * sum(_pdf_skg))
    pdf_z1 = prob_tx * pdf_tx + (1 - prob_tx) * _factor_pdf_skg * dz * _pdf_skg
    cdf_z2 = np.heaviside(z, 0)

    results = [cdf_z2[_idx_budget]]
    for i in range(1, num_timesteps):
        LOGGER.debug(f"Calculating IDE for time step {i:d}/{num_timesteps:d}")
        cdf_z2 = signal.fftconvolve(pdf_z1, cdf_z2, mode="full")  # /len(s)
        cdf_z2 = cdf_z2[first_idx : first_idx + num_z1]
        cdf_z2 = cdf_z2 * np.heaviside(z, 0)
        cdf_z2 = np.clip(cdf_z2, 0, 1)
        results.append(cdf_z2[_idx_budget])
    results = np.array(results)
    return z[_idx_budget], 1.0 - results


if __name__ == "__main__":
    from rayleigh import pdf_skg_rate_rayleigh
    import matplotlib.pyplot as plt

    lam_x = 0.5
    lam_y = 1
    pdf = lambda s: pdf_skg_rate_rayleigh(-s, lam_x, lam_y)
    prob_tx = 0.3
    message_length = 5
    b, outage_prob = ide_fft(pdf, prob_tx, message_length, num_timesteps=5, max_x=120)
    # b, outage_prob = ide_conv(pdf, prob_tx, message_length, num_timesteps=25)
    for time_slot, outage_time in enumerate(outage_prob):
        plt.plot(b, outage_time, label=f"{time_slot:d}")
    plt.legend()
    plt.show()
