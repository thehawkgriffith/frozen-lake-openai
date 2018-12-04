import gym
import numpy as np
from time import sleep

class Agent():

	def __init__(self, env):
		self.actions = []
		for i in range(env.action_space.n):
			self.actions.append(i)
		states = set()
		for i in range(16):
			states.add(i)
		self.Qtable = {}
		for state in states:
			self.Qtable[state] = {}
			for action in self.actions:
				self.Qtable[state][action] = 0

	def move(self, state, epsilon = 0.1):
		if np.random.random() < epsilon:
			return np.random.choice(self.actions)
		else:
			action_value = 0
			best_action = 0
			for action in self.Qtable[state]:
				if self.Qtable[state][action] >= action_value:
					action_value = self.Qtable[state][action]
					best_action = action
			return best_action


def play_one(agent, env, alpha, gamma):
    done = False
    s = env.reset()
    total_reward = 0
    while not done:
        #env.close()
        a = agent.move(s)
        s_prime, r, done, _ = env.step(a)
        if done:
            r = -2
        #env.render()
        #sleep(0.2)
        agent.Qtable[s][a] = alpha * (r + gamma * (np.max(list(agent.Qtable[s_prime].values())) - agent.Qtable[s][a]))
        s = s_prime
        total_reward += r
    return total_reward

env = gym.make('FrozenLake-v0')
agent = Agent(env)
total_rewards = []
for i in range(1000):
    #print("Episode ", i+1)
    #sleep(0.3)
    total_reward = play_one(agent, env, 1, 0.9)
    total_rewards.append(total_reward)
print(np.mean(total_rewards))