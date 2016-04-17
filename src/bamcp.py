from pymc import rbeta
import numpy as np
import sys
import bandits
import math
import ast
import random
import node
import history
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
from constants import Flags as ops
from constants import Envs as envs

import fileinput, sys

rand = np.random.rand

class BAMCP:
    # no. actions = no. bandits
    # each bandit has probability p

    def __init__(self, env, bandits, alpha, beta, discount, epsilon, options):
        
        """ INIT PARAMS"""
        self.bandits = bandits
        self.num_actions = len(bandits)
        self.num_states = bandits.num_states
        self.learn_rate = .2
        self.discount = discount
        self.c = 20
        self.r_max = bandits.max_reward
        self.epsilon = epsilon
        
        if !self.deterministic:
            if self.unfavorable_prior:
                best_bandit = np.argmax(self.bandits)
                beta[best_bandit] += alpha * ops.UNFAVORABLE_BETA_CONST

        """ SET PRIOR """
        # Alpha/Beta values set via history
        self.wins = np.array(alpha)
        self.trials = np.array(alpha) + np.array(beta)



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

        """ MISCALIBRATION OPTIONS """
        self.overgeneralize = False 
        self.sample_cost = False
        self.unfavorable_prior = False

        self._setFlags(options)

        """ ENVIRONMENT OPTIONS """
        self.deterministic = False

        self._setEnv(env)

    def search(self, numSimulations, state):

        # Clear node counts for initial run
        self.vnode_count = {}
        self.qnode_count = {}
        self.qnode_val = {}

        for i in range(0, numSimulations):
            # Sample distribution from prior for action transitions
            prior = rbeta(1 + self.wins[state], 1 + self.trials[state] - self.wins[state])

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
        self.wins[state][action] += reward
        self.trials[state][action] += 1

        """ 
        MISCALIBRATION: Generalizes success/failure
        of bandit to all other bandits
        """
        if self.overgeneralize:
            for i in range(self.num_actions):
                if i != action:
                    self.wins[state][i] += reward
                    self.trials[state][i] += 1

        self.hist.updateHist(state, action)

        return action


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

            # Epsilon-greedy rollout policy
            for i in range(0, len(dist)):
                if i != action:
                    dist[i] = epsilon / self.num_actions
                else:
                    dist[i] = 1 - epsilon + (epsilon / self.num_actions)
            self._normalize(dist)

            # Return sample of normalized distribution
            return self._sample(dist)

        


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
        if self.deterministic:
            if action == 0:
                return .5
            else:
                return rand() <= dist[action]
                
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

        if env == envs.DETERMINISTIC:
            self.deterministic = True

    def _setFlags(self, options):

        if ops.OVER_GENERALIZE in options:
            print "Miscallibration: Overgeneralize"
            self.overgeneralize = True
        
        if ops.SAMPLE_COST in options:
            self.sample_cost = True

        if ops.UNFAVORABLE_PRIOR in options:
            self.unfavorable_prior = True


if __name__ == "__main__":
    main()
