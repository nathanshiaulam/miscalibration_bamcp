from pymc import rbeta
import numpy as np
import sys
import bandits
import math
import ast
import random

import fileinput, sys

rand = np.random.rand

"""FLAGS FOR MODEL"""

# Wrong Problem
SAMPLE_COST = "sample_cost"
OVER_GENERALIZE = "over_generalize"

# Wrong Inference


# Wrong Environment

def main():
    args = sys.argv

    matrix = loadGittins("gittins_indices.txt")

    arr = ast.literal_eval(args[1])
    p_arr = np.array(arr) 

    discount = float(args[2])
    epsilon = float(args[3])
    num_sims = int(args[4])

    # Probability for ea. bandit
    b = bandits.Bandits(p_arr) 

    alpha = np.array([[1,2]])
    beta = np.array([[1,1]])
    zero = 0
    one = 0
    print "Gittins zero: " + matrix[0][0]
    print "Gittins one: " + matrix[0][1]

    for i in range(0, 1):
        bamcp = BAMCP(b, alpha, beta, discount, epsilon, 1.0/math.log(2.0))
        val = bamcp.search(num_sims, 0)
        if val == 0: 
            zero += 1
        else: 
            one += 1
    print "Zero: " + str(zero)
    print "One: " + str(one)

    # comp_dict = {}
    # num_trials = 10
    # print "Testing BAMCP against GI with:"
    # print "Discount: " + str(discount) + ", Epsilon: " + str(epsilon) + ", Expl: " + str(1.0 / math.log(2.0)) + ", Num Sims: " + str(num_sims)
    # observed_indices = []
    # for i in range(0, len(matrix)):
    #     for j in range(0, len(matrix)):
    #         gittins_index = matrix[i][j]
    #         print "-------------------------------------"
    #         observed_indices.append(gittins_index)
    #         for k in range(0, len(matrix)):
    #             for l in range(0, len(matrix)):
    #                 other_index = matrix[k][l]
    #                 if other_index not in observed_indices:
    #                     print "Comparing (alpha=" + str(j + 1) + ", beta=" + str(i + 1) + ") with GI=" + str(gittins_index) + ": "
    #                     print "vs. (alpha=" + str(l + 1) + ", beta=" + str(k + 1) + ") with GI=" + str(other_index) + ": "

    #                     correct = 0
    #                     incorrect = 0

    #                     if gittins_index >= other_index: 
    #                         print "Should choose 0."
    #                     else:
    #                         print "Should choose 1."

    #                     a_b_tup = "(" + str(j + 1) + ", " + str(i + 1) + ")=" + str(gittins_index) + " vs. " + str(l + 1) + ", " + str(k + 1) + ")=" + str(other_index) 
    #                     alpha = np.array([[j + 1, l + 1]])
    #                     beta = np.array([[i + 1, k + 1]])
    #                     for m in range(0, num_trials):
    #                         bamcp = BAMCP(b, alpha, beta, discount, epsilon, 1.0 / math.log(2.0))
    #                         val = bamcp.search(num_sims, 0)
    #                         print "Bandit: " + str(val)
    #                         if gittins_index >= other_index:
    #                             if val == 0:
    #                                 correct += 1
    #                             else: 
    #                                 incorrect += 1
    #                         else:
    #                             if val == 0:
    #                                 incorrect += 1
    #                             else: 
    #                                 correct += 1
    #                     comp_dict[a_b_tup] = float(correct) / float(num_trials)
    #                     print "Correct: " + str(correct)
    #                     print "Incorrect: " + str(incorrect)
    # for k,v in comp_dict:
    #     print k + ": " + v

# def compTable():


def pull(action, dist):
    return rand() < dist[action]

def normalize(dist):
    total = 0
    for i in range(0, len(dist)):
        total += dist[i]
    for i in range(0, len(dist)):
        dist[i] /= total

def loadGittins(filename):
    gittins = []
    with open(filename) as f:
        gittins = f.readlines()
    matrix = [[0 for x in range(len(gittins))] for x in range(len(gittins))]
    for i in range(0, len(matrix)):
        vals = gittins[i].rstrip().split(" ")
        for j in range(0, len(matrix)):
            matrix[i][j] = vals[j]
    return matrix

def sample(dist):
    sample = rand()

    for i in range(1, len(dist)):
        dist[i] += dist[i - 1]

    for i in range(0, len(dist)):
        if sample < dist[i]:
            return i 

    return -1

class BAMCP:
    # no. actions = no. bandits
    # each bandit has probability p

    def __init__(self, bandits, alpha, beta, discount, epsilon, expl_const):
        self.bandits = bandits
        self.n_bandits = len(bandits)
        self.num_states = bandits.num_states

        # history
        self.wins = np.array(alpha)
        self.trials = np.array(alpha) + np.array(beta)

        self.action_counts = np.zeros(shape=(self.num_states, self.n_bandits))
        self.q_vals = np.zeros(shape=(self.num_states, self.n_bandits))
        self.total_count = np.zeros(self.num_states)

        self.discount = discount
        self.c = 1.0 / math.sqrt(2)
        self.r_max = bandits.max_reward
        self.epsilon = epsilon

    def search(self, numSimulations, state):

        # Reset count to 0 at root
        self.total_count = np.zeros(self.num_states)

        for i in range(0, numSimulations):
            prior = rbeta(1 + self.wins[state], 1 + self.trials[state] - self.wins[state])
            self.simulate(prior, 0, state)
            if (i % 1000) == 0:
                print i
            # print "Pre Qvals: " + str(self.q_vals)

        actions = np.argwhere(self.q_vals[state] == np.amax(self.q_vals[state]))
        actions = actions.flatten()
        action = random.choice(actions)

        reward = pull(action, self.bandits.p)

        self.wins[state][action] += reward
        self.trials[state][action] += 1

        return action


    def simulate(self, prior, depth, state):
        
        if (self.r_max * math.pow(self.discount, depth)) < self.epsilon:
            return 0

        if self.total_count[state] == 0:

            self.action_counts = np.empty(shape=(self.num_states, self.n_bandits))
            self.q_vals = np.empty(shape=(self.num_states, self.n_bandits))
            
            # TODO: Sample state from prior dist
            # Don't need to, because Bandits only have one state (deterministic)

            # Rollout from current state with epsilon value of .2
            action = self.rollout_policy(state, .2)
            r = pull(action, prior)

            reward = r + self.discount * self.rollout(prior, depth, state)

            self.total_count[state] += 1
            self.action_counts[state][action] += 1
            self.q_vals[state][action] += reward

            return reward

        expl_vals = list(self.q_vals[state])

        for i in range(0, self.n_bandits):

            expl_const = 0 
            if self.action_counts[state][i] == 0:
                expl_vals[i] = float("inf")
            else:
                # Constant at the end to account for unexplored VNodes
                expl_const = self.c * math.sqrt(math.log(self.total_count[state]) / (self.action_counts[state][i]))
                expl_vals[i] += expl_const

        # TODO: Sample state from prior dist
        # Don't need to, because Bandits only have one state (deterministic)

        # Select action with max Q-val, break ties randomly
        actions = np.argwhere(expl_vals == np.amax(expl_vals))
        actions = actions.flatten()
        action = random.choice(actions)

        r = pull(action, prior)

        reward = r + self.discount * self.simulate(prior, depth + 1, state)

        self.total_count[state] += 1
        self.action_counts[state][action] += 1
        self.q_vals[state][action] += (reward - self.q_vals[state][action]) / self.action_counts[state][action]

        return reward


    def rollout_policy(self, state, epsilon):
        dist = np.empty(self.n_bandits)

        if (len(set(self.q_vals[state])) == 1):

            # Create a uniform distribution for each bandit
            for i in range(0, len(dist)):
                dist[i] = self.epsilon / self.n_bandits
            normalize(dist)

        else:        

            # Select action with max Q-val, break ties randomly
            actions = np.argwhere(self.q_vals[state] == np.amax(self.q_vals[state]))
            actions = actions.flatten()
            action = random.choice(actions)

            if (rand() < (1 - epsilon)):
                return action
            # Epsilon-greedy rollout policy
            for i in range(0, len(dist)):
                if i != action:
                    dist[i] = epsilon / self.n_bandits
                else:
                    dist[i] = 1 - epsilon + (epsilon / self.n_bandits)
            normalize(dist)

        # Return sample of normalized distribution
        return sample(dist)


    def rollout(self, prior, depth, state):

        if (self.r_max * math.pow(self.discount, depth)) < self.epsilon:
            return 0

        action = self.rollout_policy(state, .2)
        r = pull(action, prior)
        return r + self.discount * self.rollout(prior, depth + 1, state)


if __name__ == "__main__":
    main()
