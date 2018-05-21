import gym
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable 

# Load environment
env = gym.make('FrozenLake-v0')

# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss
# TODO: define network, loss and optimiser(use learning rate of 0.1).

# device = torch.device('cpu')

dtype = torch.FloatTensor
D_in, D_out = 16, 4
learning_rate = 0.1

# w = torch.randn(D_in, D_out, device=device, requires_grad=True)
w = torch.randn(D_in, D_out, requires_grad=True)


# model = torch.nn.Sequential(torch.nn.Linear(D_in, D_out))
# loss_fn = torch.nn.MSELoss(size_average=False)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




# Implement Q-Network learning algorithm

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # 1. Choose an action greedily from the Q-network
        #    (run the network for current state and choose the action with the maxQ)
        # TODO: Implement Step 1

        Q = w[s]
        # Q = model(s)
        value, action = torch.max(Q, 0)
        action = action.item() 


        # 2. A chance of e to perform random action
        if np.random.rand(1) < e:
            action = env.action_space.sample()

        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        s1, r, d, _ = env.step(action)

        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        # TODO: Implement Step 4
        Q1 = w[s1]

        # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
        # TODO: Implement Step 5
        maxQ1 = torch.max(Q1)
        target_value = r + y*maxQ1
        loss = (target_value - value).pow(2)

        # 6. Train the network using target and predicted Q values (model.zero(), forward, backward, optim.step)
        # TODO: Implement Step 6
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            w.grad.zero_()


        rAll += r
        s = s1
        if d == True:
            #Reduce chance of random action as we train the model.
            e = 1./((i/50) + 1)
            break
    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))