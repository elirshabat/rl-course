import numpy as np

NUM_STATES = 5
NUM_ACTIONS = 2
GAMMA = 0.995
TOLERANCE = 0.01
NO_LEARNING_THRESHOLD = 20

state = 0
R = 1
new_state = 1

# Initialization
value_func = np.random.random(NUM_STATES)*0.10
transition_func = np.ones([NUM_STATES, NUM_ACTIONS, NUM_STATES])/NUM_STATES
reward_func = np.zeros(NUM_STATES)

observed_sas_num = np.zeros([NUM_STATES, NUM_ACTIONS, NUM_STATES])
observed_state_rewards_sum = np.zeros(NUM_STATES)
observed_state_count = np.zeros(NUM_STATES)

# Greedy
action = np.argmax(np.matmul(transition_func[state,:,:], value_func))

# record the number of times `state, action, new_state` occurs
# record the rewards for every `new_state`
# record the number of time `new_state` was reached
observed_sas_num[state, action, new_state] += 1
observed_state_rewards_sum[new_state] += R
observed_state_count[new_state] += 1
observed_sas_num[state, action, new_state] += 1
observed_state_rewards_sum[new_state] += R
observed_state_count[new_state] += 1
observed_sas_num[state, action, 2] += 1
observed_state_rewards_sum[2] += R
observed_state_count[2] += 1

# Update MDP model using the current accumulated statistics about the
# MDP - transitions and rewards.
# Make sure you account for the case when a state-action pair has never
# been tried before, or the state has never been visited before. In that
# case, you must not change that component (and thus keep it at the
# initialized uniform distribution).
new_reward_func = np.zeros(NUM_STATES)
observed_states_mask = observed_state_count > 0
new_reward_func[observed_states_mask] = (observed_state_rewards_sum[observed_states_mask] / 
                                         observed_state_count[observed_states_mask])

new_transition_func = np.ones([NUM_STATES, NUM_ACTIONS, NUM_STATES])/NUM_STATES
for s in range(NUM_STATES):
    for a in range(NUM_ACTIONS):
        observed_states_mask = observed_sas_num[s, a, :] > 0
        observed_states_count = np.sum(observed_states_mask)
        observed_states_weight = observed_states_count/NUM_STATES
        new_transition_func[s, a, observed_states_mask] = ((observed_sas_num[s, a, observed_states_mask] / 
                                                            np.sum(observed_sas_num[s, a, :])) * observed_states_weight)
                                                            
transition_func = new_transition_func
reward_func = new_reward_func




# Print
print(new_transition_func)

