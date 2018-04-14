import numpy as np

# Map label to index.
index = {
        'B': 0, 'K': 1,
        'O': 2, '-': 3
        }

# Probability matrix
P_mat = np.array([
        [0.1, 0.325, 0.25, 0.325], 
        [0.4, 0    , 0.4 , 0.2], 
        [0.2, 0.2  , 0.2 , 0.4], 
        [1  , 0    , 0   , 0]])


def P(l1, l2):
    """ Return value from P given labels.
    """
    return P_mat[index[l1], index[l2]]

def r1(l1, l2):
    """ Return the reward for the first time step (zero for not starting with 'B').
    """
    if l1 == 'B':
        return P(l1,l2)
    else:
        return 0

def r(l1, l2):
    """ Return the reward.
    """
    return P(l1, l2)

# Map state to index
states = {'B': 0, 'K': 1, 'O': 2}

def argmax_dict(d):
    max_val = float('-inf')
    argmax = None
    for key in d:
        if d[key] > max_val:
            max_val = d[key]
            argmax = key
    return argmax

def backward(v, traj, k):
    """ Backward one step in the value-iteration
    
        Args:
            v (dict) - value function.
            traj (dict) - best trajectory from each state.
            k (int) - remaining length.
    """
    if k <= 1:
        return v, traj
    
    # update value of previous step
    v_prev, a_prev = {}, {}
    for s in states:
        v_prev[s], a_prev[s] = float('-inf'), None
        for a in states:
            tmp_value = r(s,a)*v[a] if k > 1 else r1(s,a)*v[a]
            if tmp_value > v_prev[s]:
                v_prev[s], a_prev[s] = tmp_value, a
    
    # update trajectory
    prev_traj = {}
    for s in states:
        prev_traj[s] = [a_prev[s]] + traj[a_prev[s]]
    
    # backward
    return backward(v_prev, prev_traj, k-1)

def find_most_probable_word(k):
    """ Find the most probable word of the given length.
    
    Args:
        k (int) - length of word.
    
    Return:
        The word as a string.
    """
    # initilize
    v_prev, traj = {}, {}
    for s in states:
        v_prev[s] = r(s, '-')
        traj[s] = ['-']
    
    # backward
    v, traj = backward(v_prev, traj, k)
    
    max_value, word = float('-inf'), None
    for s in states:
        if v[s] > max_value:
            max_value, word = v[s], "".join([s] + traj[s])
    
    return word
            
print("The best probable word of length 5 is: {}".format(find_most_probable_word(5)))

