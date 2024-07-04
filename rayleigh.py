import numpy as np
from scipy import special


def pdf_skg_rate_rayleigh(s, lam_x, lam_y):
    _denom = (lam_y + lam_x * (2**s - 1)) ** 2
    _factor = np.log(2) * lam_x * lam_y
    _part1 = 2**s * np.exp(lam_x * (1 - 2**s))
    _part2 = 1 + lam_y + lam_x * (2**s - 1)
    _pdf = _factor * _part1 * _part2 / _denom
    _pdf = np.where(s < 0, 0., _pdf)
    _pdf = np.maximum(_pdf, 0)
    return _pdf

def logpdf_skg_rate_rayleigh(s, lam_x, lam_y):
    _denom = 2*np.log(lam_y + lam_x * (2**s - 1))
    _factor = np.log(np.log(2)) + np.log(lam_x) + np.log(lam_y)
    _part1 = s*np.log(2) + (lam_x * (1 - 2**s))
    _part2 = np.log(1 + lam_y + lam_x * (2**s - 1))
    _pdf = _factor + _part1 + _part2 - _denom
    _pdf = np.where(s < 0, -np.infty, _pdf)
    #_pdf = np.maximum(_pdf, 0)
    return _pdf

def cdf_skg_rate_rayleigh(s, lam_x, lam_y):
    _denom = lam_y + lam_x * (2**s - 1)
    _enum = lam_y * np.exp(lam_x*(1-2**s))
    _cdf = 1 - _enum/_denom
    _cdf = np.where(s <= 0, 0., _cdf)
    _cdf = np.clip(_cdf, 0, 1)
    return _cdf

def sf_skg_rate_rayleigh(s, lam_x, lam_y):
    return 1.-cdf_skg_rate_rayleigh(s, lam_x, lam_y)


def expectation_skg_rate_rayleigh(power, lam_bob, lam_eve):
    if power <= 0:
        return 0
    if lam_bob == lam_eve:
        expect = (power + np.exp(lam_eve / power) * special.expi(-lam_eve / power)) / (
            power * np.log(2)
        )
    else:
        _part1 = np.exp(lam_bob / power) * special.expi(-lam_bob / power)
        _part2 = np.exp(lam_eve / power) * special.expi(-lam_eve / power)
        _denom = (lam_bob - lam_eve) * np.log(2)
        expect = lam_eve * (_part1 - _part2) / _denom
    return expect

def expectation_skg_rate_rayleigh_conditional_bob(power, channel_bob, lam_eve):
    if power <= 0:
        return 0
    _part1 = np.exp(lam_eve/power) * special.expi(-lam_eve/power) 
    _part2 = -np.exp(lam_eve*(channel_bob + 1/power)) * special.expi(-(lam_eve + channel_bob*lam_eve*power)/power)
    _part3 = np.log(1 + channel_bob*power)
    conditional_expect = (_part1 + _part2 + _part3)/np.log(2)
    return conditional_expect


if __name__ == "__main__":
    from scipy import stats
    import matplotlib.pyplot as plt

    num_samples = 40000
    power_db = 2
    lam_x = 10**(-(10+power_db)/10.)
    lam_y = 10**(-power_db/10.)
    prob_tx = .3


    cond_expect = expectation_skg_rate_rayleigh_conditional_bob(10, 5, 1)
    assert np.round(cond_expect, 2) == 3.01

    x = stats.expon.rvs(scale=1./lam_x, size=num_samples)
    y = stats.expon.rvs(scale=1./lam_y, size=num_samples)
    skg = np.log2((1+x+y)/(1+y))
    s = np.linspace(-2, max(skg)*1.1, 200)
    #s = np.linspace(-2, 100, 200)

    from ultimate_ruin_prob import calculate_ultimate_ruin_mixed
    b, out = calculate_ultimate_ruin_mixed(
            lambda s: np.exp(logpdf_skg_rate_rayleigh(s, lam_x, lam_y)),
            lambda s: sf_skg_rate_rayleigh(s, lam_x, lam_y),
            message_length=5,
            prob_tx=prob_tx,
            max_b=210,
            num_points=9000)
    plt.figure()
    plt.plot(b, out)
    plt.show()
