import os.path
from datetime import datetime

import numpy as np
from scipy import stats

import ray
from ray import air
from ray import tune
from ray import train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

from environment import SecretKeyEnv

if __name__ == "__main__":
    from configs import env_config
    from configs import RESULTS_DIR

    ray.shutdown()
    ray.init(num_cpus=63, num_gpus=0)

    algorithm = "PPO"
    tuner = tune.Tuner(
        algorithm,
        tune_config=tune.TuneConfig(
            metric="episode_reward_min",
            mode="max",
            num_samples=42,
            max_concurrent_trials=63,
        ),
        param_space={
            "env": SecretKeyEnv,
            "env_config": env_config,
            "model": {
                "free_log_std": True,
                "use_lstm": False,
            },
        },
        run_config=train.RunConfig(
            stop={"training_iteration": 5000},
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=2, checkpoint_at_end=True, checkpoint_frequency=4
            ),
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result()
    best_checkpoint_mean = best_result.get_best_checkpoint("episode_reward_mean", "max")
    best_checkpoint_min = best_result.get_best_checkpoint("episode_reward_min", "max")
    _dirname = "{}-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M"), algorithm)
    best_checkpoint_mean.to_directory(os.path.join(RESULTS_DIR, _dirname))
    best_checkpoint_min.to_directory(os.path.join(RESULTS_DIR, f"{_dirname}-min"))
    ray.shutdown()
