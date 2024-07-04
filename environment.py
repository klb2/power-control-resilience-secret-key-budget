import numpy as np
import gymnasium as gym
import ray
from ray.rllib.utils import check_env
from ray.rllib.env.env_context import EnvContext

from util import to_decibel, db_to_linear


class SecretKeyEnv(gym.Env):
    def __init__(self, env_config: EnvContext, seed_id=None):
        self.seed_id = seed_id
        if seed_id is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng([seed_id, 0x8c3c010cb4754c905776b])
        #self.action_space = gym.spaces.Box(low=0, high=np.inf, shape=(1,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Dict(
            {
                "budget": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
                "channel_bob": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
                "is_transmission": gym.spaces.Discrete(2),
                "net_income": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            }
        )

        self.max_power_db = env_config.get("max_power_db", 30)
        self.message_length = int(env_config["message_length"])
        assert self.message_length > 0

        self.rv_bob = env_config["rv_bob"]
        self.rv_eve = env_config["rv_eve"]
        self.prob_tx = env_config["prob_tx"]

        self.time_step = 0
        self.time_limit = int(env_config.get("time_limit", 5000))
        self.max_episode_steps = int(env_config.get("time_limit", 5000))

        self.rv_alert_duration = env_config['rv_alert_duration']
        self.outage_prob_alert = env_config['outage_prob_alert']
        self.min_budget = self.rv_alert_duration.ppf(1-self.outage_prob_alert)*self.message_length

        self.init_budget = env_config.get("init_budget", 1.*self.min_budget)
        self.budget = 1.*self.init_budget

        self.resilience_outage_counter = 0
        self.avg_power = 0
        self.powers = []
        self.number_skg = 0

    def get_system_state(self):
        is_transmission = int(self.rng.binomial(1, self.prob_tx))
        channel_gain_bob = self.rv_bob.rvs(random_state=self.rng)
        state = {
            "is_transmission": is_transmission,
            "channel_bob": np.array([channel_gain_bob], dtype=np.float32),
        }
        channel_gain_eve = np.float32(self.rv_eve.rvs(random_state=self.rng))
        return state, channel_gain_eve

    def reset(self, *, seed=None, options=None):
        #super().reset(seed=seed, options=options)
        if seed is not None:
            self.rng = np.random.default_rng([seed, 0x8c3c010cb4754c905776b])
        elif self.seed_id is not None:
            self.rng = np.random.default_rng([self.seed_id, 0x8c3c010cb4754c905776b])
        #np.random.seed(seed)
        reset_obs = {
            "budget": np.array([self.init_budget], dtype=np.float32),
            "channel_bob": np.array([self.rv_bob.rvs(random_state=self.rng)], dtype=np.float32),
            "is_transmission": 0,
            "net_income": np.array([0.], dtype=np.float32),
        }
        self.budget = self.init_budget
        self.time_step = 0
        self.number_skg = 0
        self.resilience_outage_counter = 0
        self.avg_power = 0
        self.powers = []
        info = {}
        return reset_obs, info

    def step(self, action):
        self.time_step = self.time_step + 1

        action = action[0]
        state, channel_eve = self.get_system_state()

        is_transmission = state['is_transmission']
        channel_bob = state['channel_bob']
        if is_transmission:
            #power_tx = (2**self.message_length-1)/channel_bob
            net_income = np.array([-self.message_length])
        else:
            power_tx = 10**(action*self.max_power_db/10.)
            channel_eve = self.rv_eve.rvs(random_state=self.rng)
            net_income = np.log2((1 + power_tx*channel_bob + power_tx*channel_eve)/(1+power_tx*channel_eve))
            self.number_skg = self.number_skg + 1
            #self.avg_power = self.avg_power + (power_tx - self.avg_power)/self.number_skg
            self.powers.append(power_tx)
            self.avg_power = np.mean(self.powers[-10:])
            #self.avg_power = np.mean(self.powers[-20:])
            #self.avg_power = np.mean(self.powers[-100:])
        self.budget = self.budget + net_income

        terminated = False
        #if self.budget < self.min_budget:
        if self.budget <= 0:
            terminated = True
            self.budget = np.array([0], dtype=np.float32)

        _obs_update = {
            "budget": self.budget,
            "net_income": net_income,
        }
        state.update(_obs_update)
        obs = state

        # Info
        alert_outage_prob = self.rv_alert_duration.sf(self.budget/self.message_length)
        alert_outage_prob_log = self.rv_alert_duration.logsf(self.budget/self.message_length)
        info = {"budget": self.budget,
                "alert_outage_prob": alert_outage_prob,
                "alert_outage_prob_log": alert_outage_prob_log,
                "is_transmission": is_transmission,
               }

        if alert_outage_prob > self.outage_prob_alert:
            self.resilience_outage_counter = self.resilience_outage_counter + 1


        if is_transmission:
            reward = 1 # positive as incentive to stay alive # 0
        else:
            power_reward = -50*self.avg_power / db_to_linear(self.max_power_db)
            outage_reward = -10*np.log10(self.resilience_outage_counter/self.time_step) if self.resilience_outage_counter > 0 else 10
            alert_reward = -np.log10(np.clip(alert_outage_prob, 1e-10, 1))
            budget_reward = 10*np.minimum(1.-self.min_budget/self.budget, 0)
            reward = power_reward + outage_reward + budget_reward + alert_reward

            reward = reward[0]
            if terminated:
                reward = -100000*self.time_limit/self.time_step
        info["reward"] = reward

        truncated = False
        if self.time_step >= self.time_limit:
            truncated = True
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    from scipy import stats
    env_config = {"message_length": 5,
                  "rv_bob": stats.expon(scale=2),
                  "rv_eve": stats.expon(scale=1),
                  "prob_tx": .1,
                  "rv_alert_duration": stats.poisson(mu=5),
                  "outage_prob_alert": 1e-3,
                }
    env = SecretKeyEnv(env_config)
    check_env(env)
