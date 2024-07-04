import os.path

from scipy import stats


RESULTS_DIR = os.path.join(os.path.expanduser("~"), "ray_results")

env_config = {"message_length": 5,
              "rv_bob": stats.expon(scale=10),
              "rv_eve": stats.expon(scale=1),
              "prob_tx": .35,
              "rv_alert_duration": stats.poisson(mu=5),
              "outage_prob_alert": 1e-1,
              "max_power_db": 30,
              "init_budget": 70,
             }
