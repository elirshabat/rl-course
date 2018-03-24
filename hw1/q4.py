import numpy as np

index = {
        'B': 0, 'K': 1,
        'O': 2, '-': 3
        }

P_mat = np.array([
        [0.1, 0.325, 0.25, 0.325], 
        [0.4, 0    , 0.4 , 0.2], 
        [0.2, 0.2  , 0.2 , 0.4], 
        [1  , 0    , 0   , 0]])

def P(l1, l2):
    return P_mat[index[l1], index[l2]]

def r1(l1, l2):
    if l1 == 'B':
        return P(l1,l2)
    else:
        return 0

def r(l1, l2):
    return P(l1, l2)

def r_k(l):
    return P(l, '-')

states = {'B': 0, 'K': 1, 'O': 2}

def argmax(d):
    max_val = float('-inf')
    argmax = None
    for key in d:
        if d[key] > max_val:
            max_val = d[key]
            argmax = key
    return argmax

def find_most_probable_word(k):
    #initilize
    trajectories = {}
    v = {}
    for s in states:
        v[s] = r_k(s)
        trajectories[s] = ['-']
    
    #backward
    for t in reversed(range(k)):
        print(v)
        v_prev = {}
        best_action = {}
        for s in states:
            max_val, max_action = float('-inf'), None
            for a in states:
                value = r(s,a)*v[a] if t != 0 else r1(s,a)*v[a]
                if value > max_val:
                    max_val, argmax_val = value, a
            v_prev[s] = max_val
            trajectories[s] = [best_action[s]] + trajectories[s]
#        trajectory.append(argmax(v_next))
        v = v_next.copy()
#    trajectory.reverse()
    return trajectories
            
print(find_most_probable_word(5))
