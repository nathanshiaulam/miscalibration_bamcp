from pymc import rbeta
import numpy as np
import sys

rand = np.random.rand


class Bandits:
    """
    This class represents N Bernoulli bandits machines.

    parameters:
        p_array: a (n,) Numpy array of probabilities >0, <1.

    methods:
        pull( i ): return the results, 0 or 1, of pulling 
                   the ith bandit.
    """
    def __init__(self, p_array):
        self.p = p_array
        self.optimal = np.argmax(p_array)
        self.max_reward = 1
        self.num_states = 1
        
    def pull(self, i):
        return rand() < self.p[i] 
    
    def __len__(self):
        return len(self.p)
