import random
import numpy as np


class CoinToss:
    def __init__(self, head_probs, max_episode_steps=30):
        """
        构造函数
        :param head_probs: 数组参数，用于指定哥枚硬币正面朝上的概率
        :param max_episode_steps: 硬币的投掷次数，到达这个数值后，游戏结束
        """
        self.head_probs = head_probs
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    def step(self, action):
        """投掷硬币"""
        final = self.max_episode_steps - 1
        if self.toss_count > final:
            raise Exception("This step count exceeded maximum. Please reset env.")
        else:
            done = True if self.toss_count == final else False

        if action >= len(self.head_probs):
            raise Exception("The No. {} coin doesn't exist.")
        else:
            head_prob = self.head_probs[action]
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0
            self.toss_count += 1
            return reward, done


class EpsilonGreedyAgent:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.V = []

    def policy(self):
        """epsilon greedy算法实现
        以epsilon为概率随机选择硬币（探索），其他情况下按照各枚硬币的期望值来选择（利用）
        """
        coins = range(len(self.V))
        if random.random() < self.epsilon:  # 探索
            return random.choice(coins)
        else:
            return np.argmax(self.V)

    def play(self, env):
        # 初始化估计值
        N = [0] * len(env)
        self.V = [0] * len(env)

        env.reset()
        done = False
        rewards = []
        while not done:
            selected_coin = self.policy()
            reward, done = env.step(selected_coin)
            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)
            N[selected_coin] += 1
            self.V[selected_coin] = new_average

        return rewards


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt


    def main():
        env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
        epsilons = [0.0, 0.1, 0.2, 0.5, 0.8]
        game_steps = list(range(10, 310, 10))
        result = {}
        for e in epsilons:
            agent = EpsilonGreedyAgent(epsilon=e)
            means = []
            for s in game_steps:
                env.max_episode_steps = s
                rewards = agent.play(env)
                means.append(np.mean(rewards))
            result["epsilon={}".format(e)] = means
        result["coin toss count"] = game_steps
        result = pd.DataFrame(result)
        result.set_index("coin toss count", drop=True, inplace=True)
        result.plot.line(figsize=(10, 5))
        plt.show()


    main()