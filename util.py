import numpy as np
import pandas as pd

def find_closest_element_idx(array, value):
    closest_element = min(array, key=lambda x: abs(x-value))
    idx = np.where(array == closest_element)[0]
    return idx

def to_decibel(power):
    return 10*np.log10(power)

def linear_to_db(value):
    return 10*np.log10(value)

def db_to_linear(value):
    return 10**(np.array(value)/10.)

def capacity(snr):
    snr = np.array(snr)
    return np.log2(1+snr)

def export_results(results, filename):
    df = pd.DataFrame.from_dict(results)
    df.to_csv(filename, sep='\t', index=False)

def clip_ruined_budget(budgets):
    """
    Parameters
    ----------
    budgets: array (N x T)
      Array where each row represents one sample of :math:`T` time slots.
    """
    time_ruin = np.argmin(budgets > 0, axis=1)
    budgets = np.copy(budgets)
    for _sample, _time_of_ruin in zip(budgets, time_ruin):
        if _time_of_ruin == 0:
            continue
        _sample[_time_of_ruin:] = 0
    return budgets

def calculate_alert_outage_probability(budgets, message_length, rv_alert_duration):
    alert_outage_prob = rv_alert_duration.sf(budgets/message_length)
    alert_outage_prob[np.where(budgets == 0)] = 1
    alert_outage_prob = np.mean(alert_outage_prob, axis=0)
    return alert_outage_prob
