import gym
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def epsilonGreedy(actions, epsilon):
    if (np.random.uniform() < epsilon):
        return np.random.randint(env.action_space.n)
    else: #greedy choice
        return np.argmax(actions)


# Load environment
env = gym.make('FrozenLake-v0')

# Implement Q-Table learning algorithm
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
e = 0.1
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0 # Total reward during current episode
    done = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        # TODO: Implement Q-Learning
        # 1. Choose an action by greedily (with noise) picking from Q table
        actions = Q[s]
        a = epsilonGreedy(actions, e)

        # 2. Get new state and reward from environment
        new_s, r, done, info = env.step(a)

        #print(new_s, r, done, info)

        # 3. Update Q-Table with new knowledge
        Q[s][a] = Q[s][a] + lr*(r + y*np.max(Q[new_s]) - Q[s][a])
        s = new_s

        # 4. Update total reward
        rAll = rAll + r

        # 5. Update episode if we reached the Goal State

        if done == True:
            e = 1./((i/50) + 1)
            break

    
    rList.append(rAll)

# Reports
print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)

# print()

# Vs = np.amax(Q,1)
# print(Vs)
