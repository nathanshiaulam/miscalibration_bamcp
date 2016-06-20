import numpy as np
import sys
import random
import math
import ast
import time
import fileinput, sys

import matplotlib.pyplot as plt
from matplotlib.pyplot import show

import node
import bandits
import history
from miscalibration import Miscalibration

from constants import Flags as ops
from constants import Envs as envs
from constants import Params as params

rand = np.random.rand

""" DEBUGGING PURPOSES """
start_time = time.clock()
def timeDiff(other_time):
    t = other_time - start_time
    return t

class BAMCP:
    # no. actions = no. bandits
    # each bandit has probability p

    def __init__(self, vals):
        
        """ INIT PARAMS"""
        b = bandits.Bandits(vals[params.P_ARR]) 

        self.options =  vals[params.OPTIONS]

        self.verbose = vals[params.VERBOSE]
        self.bandits = b
        self.num_actions = len(b)
        self.num_states = b.num_states
        self.learn_rate = params.LEARN_RATE
        self.discount = vals[params.DISCOUNT]
        self.c = params.EXPLORE_CONST
        self.r_max = b.max_reward
        self.epsilon = vals[params.EPSILON]
        self.start_state = 0
        self.total_reward = 0
        self.bandit_cost = vals[params.COST]
        self.trembling_hand = 0.2
        self.forget_rate = vals[params.FORGET_RATE]

        self.num_steps = vals[params.NUM_STEPS]
        self.steps_taken = 0

        """ ENVIRONMENT OPTIONS """
        self.test_gittins = False

        self._setEnv(vals[params.ENV])

        """ SET PRIOR """
        # Alpha/Beta values set via history
        self.alpha = np.array(vals[params.ALPHA])
        self.beta = np.array(vals[params.BETA])

        self.wins = np.copy(self.alpha)
        self.trials = np.copy(self.alpha + self.beta)

        # Total action and state count
        action_counts = np.zeros(self.num_actions)
        state_counts = np.zeros(self.num_states)

        # TODO: Update for multi-state MDP
        for state in range(0, self.num_states):
            state_counts[state] = np.sum(self.trials[state])
        for state in range(0, self.num_states):
            for action in range(0, self.num_actions):
                action_counts[action] += self.trials[state][action]

        self.hist = history.History(state_counts, action_counts)

        """ NODE COUNTS """
        self.qnode_count = {}
        self.qnode_val = {}
        self.vnode_count = {}

        """ MISCALIBRATION AGENT """
        self.mc = Miscalibration(self)
        self.mc.miscalibratePriors(vals[params.BAD_PRIOR])
    

    def search(self, numSimulations, state):

        # Clear node counts for initial run
        self.vnode_count = {}
        self.qnode_count = {}
        self.qnode_val = {}

        qval_0 = []
        qval_1 = []
        num_sim = []
        for i in range(0, numSimulations):
            # Sample distribution from prior for action transitions
            prior = np.random.beta(1 + self.wins[state], 1 + self.trials[state] - self.wins[state])
            # Create defensive copy of current history 
            start_hist = history.History(self.hist.getStateCounts(), self.hist.getActionCounts())
            self.simulate(prior, 0, state, start_hist)

        # Select action with highest Q-Value
        action = -1
        max_val = float("-inf")
        for i in range(0, self.num_actions):
            qnode = node.QNode(self.hist, state, i)
            val = self.qnode_val[qnode]
            if val > max_val:
                max_val = val
                action = i

        reward = self._pull(action, self.bandits.p)

        # Update history and prior based on observation
        win = False
        if action == 0:
            win = (reward != 0 and not self.mc.sample_cost) or (reward != -self.bandit_cost and self.mc.sample_cost)
        else:
            win = True

        self.mc.miscalibrateAction(state, action, win)

        if win:
            self.alpha[state][action] += 1
        else:
            self.beta[state][action] += 1 

        self.hist.updateHist(state, action)
        self.steps_taken += 1

        """ DEBUGGING PURPOSES """
        if self.verbose:
            other_time = time.clock()
            time_diff = timeDiff(other_time)

            print "Acton: %d | Reward: %f" % (action, reward)
            print "Steps Taken: %d | Time Diff: %s" % (self.steps_taken, str(time_diff))

        sys.stdout.flush()
        
        self.wins = self.alpha
        self.trials = self.alpha + self.beta

        self.total_reward += reward

        return (action, self.steps_taken)


    def simulate(self, prior, depth, state, hist):

        # Stop growing tree and back up values
        if (self.r_max * math.pow(self.discount, depth)) < self.epsilon:
            return 0

        # EXPANSION: Generate state node for <s, h>
        vnode = node.VNode(hist, state)
        
        # ROLLOUT: @ leaf state, begin rollout
        if vnode not in self.vnode_count or self.vnode_count[vnode] == 0:

            self.vnode_count[vnode] = 0

            # Init action nodes for all possible actions from 
            for a in range(0, self.num_actions):
                qnode = node.QNode(hist, state, a)
                self.qnode_count[qnode] = 0
                self.qnode_val[qnode] = 0

            # TODO: Sample state from prior dist
            # Don't need to, because Bandits only have one state (deterministic)

            # Select initial action for rollout 
            action = self.rollout_policy(state, .2, hist, prior)
            r = self._pull(action, prior)

            # Initialize new history <has'>
            new_hist = history.History(hist.getStateCounts(), hist.getActionCounts())
            new_hist.updateHist(state, action)

            qnode = node.QNode(hist, state, action)

            # Rollout with histor <has'>
            reward = r + self.discount * self.rollout(prior, depth, state, new_hist)

            # Update values for rollout QNode(<s, h>, a)
            self.vnode_count[vnode] += 1
            self.qnode_count[qnode] += 1
            self.qnode_val[qnode] += reward

            return reward

        # Generate Q-Val + Exploration Bonus for (<s, h>, a)
        expl_vals = np.zeros(self.num_actions)
        for a in range(0, self.num_actions):

            expl_const = 0 
            qnode = node.QNode(hist, state, a)

            expl_vals[a] = self.qnode_val[qnode]
            if self.qnode_count[qnode] == 0:
                expl_vals[a] = float("inf")
            else:
                expl_const = self.c * math.sqrt(math.log(self.vnode_count[vnode]) / (self.qnode_count[qnode]))
                expl_vals[a] += expl_const

        # TODO: Sample state from prior dist
        # Don't need to, because Bandits only have one state (deterministic)

        # Select action with max Q-val with exploration bonus, break ties randomly
        actions = np.argwhere(expl_vals == np.amax(expl_vals))
        actions = actions.flatten()
        action = random.choice(actions)

        r = self._pull(action, prior)

        qnode = node.QNode(hist, state, action) # Current action node

        new_hist = history.History(hist.getStateCounts(), hist.getActionCounts()) # Initialize new history <has'>
        new_hist.updateHist(state, action)

        # SIMULATION: Move down tree with history <has'>
        reward = r + self.discount * self.simulate(prior, depth + 1, state, new_hist)

        # BACKUP PHASE: Update values for all nodes
        self.vnode_count[vnode] += 1
        self.qnode_count[qnode] += 1
        self.qnode_val[qnode] += (reward - self.qnode_val[qnode]) / (self.qnode_count[qnode])

        return reward


    def rollout_policy(self, state, epsilon, hist, prior):
        dist = np.empty(self.num_actions)

        if (len(set(prior)) == 1):

            # Create a uniform distribution for each bandit
            return random.randint(0, self.num_actions - 1)
        else:        

            # Select action with max Q-val, break ties randomly
            actions = np.argwhere(prior == np.amax(prior))
            actions = actions.flatten()
            action = random.choice(actions)

            if (rand() < (1 - epsilon)):
                return action

            """ SIMULATION OPTIMISATION """
            return random.randint(0, self.num_actions - 1)


    def rollout(self, prior, depth, state, hist):

        if (self.r_max * math.pow(self.discount, depth)) < self.epsilon:
            return 0

        action = self.rollout_policy(state, .2, hist, prior)
        r = self._pull(action, prior)

        # Initialize new history <has'>
        new_hist = history.History(hist.getStateCounts(), hist.getActionCounts())
        new_hist.updateHist(state, action)

        return r + self.discount * self.rollout(prior, depth + 1, state, new_hist)

    def _pull(self, action, dist):

        if self.test_gittins:
            if self.bandits.p[action] == 1:
                return .5
            else:
                return rand() <= dist[action]
        else:
            if ops.NONE not in self.options:
                return self.mc.miscalibratePull(action, dist, self.bandit_cost)
        return rand() <= dist[action]

   
    def _normalize(self, dist):
        total = 0
        for i in range(0, len(dist)):
            total += dist[i]
        for i in range(0, len(dist)):
            dist[i] /= total

    def _sample(self, dist):
        sample = rand()

        for i in range(1, len(dist)):
            dist[i] += dist[i - 1]

        for i in range(0, len(dist)):
            if sample < dist[i]:
                return i 

        return -1

    def _setEnv(self, env):

        if env == envs.TEST_GITTINS:
            self.test_gittins = True

if __name__ == "__main__":
    main()
