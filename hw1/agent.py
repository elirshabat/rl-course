import gym
import numpy as np
import matplotlib.pyplot as plt
import os.path

MAX_STEPS = 200
SAMPLE_TIMES = 10000
NUM_TRIALS = 1000


def agent(o, w):
    if np.dot(o,w) >= 0:
        return 1
    else:
        return 0


def evaluate(enviorment, w):
    total_reward = 0
    steps = 0
    done = False
    obsrv = env.reset()
    while not done and steps < MAX_STEPS:
        # env.render()
        action = agent(obsrv, w)
        obsrv, step_reward, done, info = env.step(action)
        total_reward += step_reward
        steps += 1
    return total_reward


def random_search(enviorment):
    for i in range(SAMPLE_TIMES):
        w = np.random.uniform(-1,1,4)
        reward = evaluate(env, w)
        if reward == 200.0:
            return i + 1
    return SAMPLE_TIMES+1


def int_histogram(arr, max_value):
    res = [0]*(max_value+1)
    for value in arr:
        res[value] += 1
    return res


env_d = 'CartPole-v1'
env = gym.make(env_d)

results = [0] * NUM_TRIALS
for i in range(NUM_TRIALS):
    results[i] = random_search(env)

average = sum(results) / len(results)
print ("average: " + str(average))
bins = range(max(results)+1)
plt.hist(results, bins)

if not os.path.isdir("out/figures"):
    os.makedirs("out/figures")
plt.savefig("out/figures/agent_hits")
