"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():

    def plot1(self, eg_estimates, num_trials):
        plt.figure(figsize=(10, 6))
        for i in range(eg_estimates.shape[1]):
            plt.plot(range(num_trials), eg_estimates[:, i], label=f'Arm {i + 1}')
        plt.title("Epsilon-Greedy: Estimated Reward per Arm (Linear Scale)")
        plt.xlabel("Trials")
        plt.ylabel("Estimated Value")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        for i in range(eg_estimates.shape[1]):
            plt.plot(range(num_trials), eg_estimates[:, i], label=f'Arm {i + 1}')
        plt.xscale('log')
        plt.title("Epsilon-Greedy: Estimated Reward per Arm (Log Scale)")
        plt.xlabel("Trials (log scale)")
        plt.ylabel("Estimated Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot2(self, eg_results, ts_results):
        cum_reward_eg = np.cumsum(eg_results['rewards'])
        cum_reward_ts = np.cumsum(ts_results['rewards'])
        cum_regret_eg = np.cumsum(eg_results['regrets'])
        cum_regret_ts = np.cumsum(ts_results['regrets'])

        plt.figure(figsize=(10, 5))
        plt.plot(cum_reward_eg, label="Epsilon Greedy")
        plt.plot(cum_reward_ts, label="Thompson Sampling")
        plt.title("Cumulative Rewards Comparison")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(cum_regret_eg, label="Epsilon Greedy")
        plt.plot(cum_regret_ts, label="Thompson Sampling")
        plt.title("Cumulative Regrets Comparison")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.grid(True)
        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, p, num_trials=20000, seed=None):
        self.p = np.array(p)
        self.num_trials = num_trials
        self.n = len(p)
        self.counts = np.zeros(self.n)
        self.values = np.zeros(self.n)
        self.trials = 0
        self.rng = np.random.default_rng(seed)
        if np.all((self.p >= 0) & (self.p <= 1)):
            self.reward_model = 'bernoulli'
        else:
            self.reward_model = 'gaussian'
        self.noise_std = 1.0

        self.rewards = np.zeros(self.num_trials)
        self.regrets = np.zeros(self.num_trials)
        self.chosen_arms = np.zeros(self.num_trials)
        self.estimate_hist = np.zeros((self.num_trials, self.n))

        self.opt_mean = float(np.max(self.p))

    def __repr__(self):
        return f"EpsilonGreedy(p={self.p}, trials={self.num_trials})"

    def pull(self, arm):
        if self.reward_model == 'bernoulli':
            return (self.rng.random() < self.p[arm])
        else:
            return float(self.rng.normal(loc=self.p[arm], scale=self.noise_std))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def experiment(self):
        for t in range(1, self.num_trials + 1):
            eps = 1/t

            if self.rng.random() < eps:
                arm = int(self.rng.integers(0, self.n))
            else:
                max_val = np.max(self.values)
                candidates = np.flatnonzero(np.isclose(self.values, max_val))
                arm = int(self.rng.choice(candidates))
            reward = self.pull(arm)
            self.update(arm, reward)

            idx = t - 1
            self.rewards[idx] = reward
            self.regrets[idx] = self.opt_mean - reward
            self.chosen_arms[idx] = arm
            self.estimate_hist[idx, :] = self.values
            self.trials = t

        return {
            'rewards': self.rewards.copy(),
            'regrets': self.regrets.copy(),
            'chosen_arms': self.chosen_arms.copy(),
            'estimate_history': self.estimate_hist.copy(),
        }
    def report(self):
        cum_reward = float(np.sum(self.rewards))
        cum_regret = float(np.sum(self.regrets))
        logger.info(f"EpsilonGreedy cumulative reward: {cum_reward:.4f}")
        logger.info(f"EpsilonGreedy cumulative regret: {cum_regret:.4f}")
        return cum_reward, cum_regret
#--------------------------------------#

class ThompsonSampling(Bandit):
    def __init__(self, p, num_trials=20000, seed=None, known_precision=1):
        self.p = np.array(p)
        self.num_trials = num_trials
        self.n = len(p)
        self.rng = np.random.default_rng(seed)

        # I assumes Bernoulli bandits as discussed during our lessons
        self.reward_model = 'bernoulli'

        # Priors
        self.alpha = np.ones(self.n)
        self.beta = np.ones(self.n)

        self.rewards = np.zeros(self.num_trials)
        self.chosen_arms = np.zeros(self.num_trials)
        self.regrets = np.zeros(self.num_trials)

        self.opt_mean = float(np.max(self.p))

    def __repr__(self):
        return f"Thompson Sampling (p = {self.p}, trials = {self.num_trials})"

    def pull(self, arm):
        return 1 if self.rng.random() < self.p[arm] else 0

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

    def experiment(self):
        for t in range(1, self.num_trials + 1):
            sampled_means = self.rng.beta(self.alpha, self.beta)
            arm = int(np.argmax(sampled_means))
            reward = self.pull(arm)
            self.update(arm, reward)

            idx = t - 1
            self.rewards[idx] = reward
            self.regrets[idx] = self.opt_mean - reward
            self.chosen_arms[idx] = arm

        return {
            'rewards' : self.rewards.copy(),
            'regrets' : self.regrets.copy(),
            'chosen_arms' : self.chosen_arms.copy(),
        }

    def report(self):
        cum_reward = float(np.sum(self.rewards))
        cum_regret = float(np.sum(self.regrets))
        logger.info(f"Thompson Sampling cumulative reward: {cum_reward:.4f}")
        logger.info(f"Thompson Sampling cumulative regret: {cum_regret:.4f}")
        return cum_reward, cum_regret


def comparison():
    p = [0.1, 0.3, 0.5, 0.8]
    num_trials = 20000

    eg = EpsilonGreedy(p, num_trials=num_trials)
    ts = ThompsonSampling(p, num_trials=num_trials)

    results_eg = eg.experiment()
    results_ts = ts.experiment()

    eg.report()
    ts.report()

    df_eg = pd.DataFrame({
        'Bandit': results_eg['chosen_arms'],
        'Reward': results_eg['rewards'],
        'Algorithm': ['EpsilonGreedy'] * num_trials
    })

    df_ts = pd.DataFrame({
        'Bandit': results_ts['chosen_arms'],
        'Reward': results_ts['rewards'],
        'Algorithm': ['ThompsonSampling'] * num_trials
    })

    df = pd.concat([df_eg, df_ts])
    df.to_csv("bandit_results.csv", index=False)
    logger.info("Results saved to bandit_results.csv")

    vis = Visualization()
    vis.plot1(eg.estimate_hist, num_trials)
    vis.plot2(results_eg, results_ts)

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    logger.info("Starting EpsilonGreedy and ThompsonSampling comparison...")
    try:
        comparison()  # This will run your algorithms
        logger.info("Experiment completed successfully.")
    except Exception as e:
        logger.exception(f"Error during experiment: {e}")

    logger.info("End of experiment log.")