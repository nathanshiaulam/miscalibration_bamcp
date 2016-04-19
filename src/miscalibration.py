import numpy as np
import random
import math
import ast

from constants import Flags as ops
import bamcp
import history

rand = np.random.rand

class Miscalibration:

    def __init__(self, bamcp):

        self.bamcp = bamcp

        self.overgeneralize = False 
        self.sample_cost = False
        self.unfavorable_prior = False
        self.forget_rate = False
        self.unfavorable_env = False
        self.anhedonia = False

        self._setFlags(bamcp.options)

    def miscalibratePriors(self):
        if not self.bamcp.test_gittins:
            if self.unfavorable_prior:
                worst_bandit = np.argmin(self.bamcp.bandits.p)
                for a in range(self.bamcp.num_actions):
                    if a != worst_bandit:
                        alpha_upper_bound = self.bamcp.alpha[self.bamcp.start_state][a] + self.bamcp.num_steps
                        self.bamcp.beta[self.bamcp.start_state][a] += alpha_upper_bound * ops.UNFAVORABLE_PRIOR_CONST
                print "Miscalibrated Priors (Alpha, Beta): (%s, %s)" % (str(self.bamcp.alpha), str(self.bamcp.beta))

        self.bamcp.wins = self.bamcp.alpha
        self.bamcp.trials = self.bamcp.alpha + self.bamcp.beta

        # Total action and state count
        action_counts = np.zeros(self.bamcp.num_actions)
        state_counts = np.zeros(self.bamcp.num_states)

        # TODO: Update for multi-state MDP
        for state in range(0, self.bamcp.num_states):
            state_counts[state] = np.sum(self.bamcp.trials[state])
        for state in range(0, self.bamcp.num_states):
            for action in range(0, self.bamcp.num_actions):
                action_counts[action] += self.bamcp.trials[state][action]

        self.bamcp.hist = history.History(state_counts, action_counts)

    def miscalibrateAction(self):

        """ 
        MISCALIBRATION: Implements a forget rate for 
        (alpha, beta) values by implementing an epsilon
        discount 
        """
        if self.forget_rate:
            self.bamcp.alpha = (1 - ops.FORGET_RATE_EPSILON) * self.bamcp.alpha 
            self.bamcp.beta = (1 - ops.FORGET_RATE_EPSILON) * self.bamcp.beta

        """ 
        MISCALIBRATION: Leaves agent in unfavorable environment
        for num_steps / UNFAVORABLE_ENV_TIME steps, with a penalty 
        for all arms other than the worse one by UNFAVORABLE_ENV_FACTOR
        for each step taken in the unfavorable environment.

        Force upper bound of mean for each beta dist to be 1 / 100.

        """
        if self.unfavorable_env:
            unfavorable_env_time = float(self.bamcp.num_steps) / float(ops.UNFAVORABLE_ENV_TIME) 
            if (self.bamcp.steps_taken < unfavorable_env_time):
                worst_bandit = np.argmin(self.bamcp.bandits.p)
                for a in range(self.bamcp.num_actions):
                    if a != worst_bandit:
                        self.bamcp.beta[state][a] += ops.UNFAVORABLE_ENV_FACTOR

        """ 
        MISCALIBRATION: Generalizes success/failure
        of bandit to all other bandits
        """
        if self.overgeneralize:
            for a in range(self.bamcp.num_actions):
                if a != action:
                    win = (reward != 0 and not self.sample_cost) or (reward != -ops.SAMPLE_COST_CONST and self.sample_cost)
                    if win:
                        self.bamcp.alpha[state][i] += 1
                    else:
                        self.bamcp.beta[state][i] += 1

        """ 
        MISCALIBRATION: Generalizes success/failure
        of bandit to all other bandits
        """
        if self.anhedonia:
            self.bamcp.alpha *= math.pow(1 - ops.ANHEDONIA_EPSILON, self.steps_taken)
            self.bamcp.beta *= math.pow(1 - ops.ANHEDONIA_EPSILON, self.steps_taken)

    def miscalibratePull(self, action, dist):
        if self.unfavorable_env:
            unfavorable_env_time = float(self.bamcp.num_steps) / float(ops.UNFAVORABLE_ENV_TIME) 
            if self.bamcp.steps_taken < unfavorable_env_time:
                return 0

        if self.sample_cost:
            min_action = np.argmin(self.bamcp.bandits.p)
            if action == min_action:
                return 0
            else:
                if rand() <= dist[action]:
                    return 1 - float(ops.SAMPLE_COST_CONST)
                else:
                    return -1 * ops.SAMPLE_COST_CONST

        return rand() <= dist[action]

    def _setFlags(self, options):

        if ops.OVER_GENERALIZE in options:
            self.overgeneralize = True
        
        if ops.SAMPLE_COST in options:
            self.sample_cost = True

        if ops.UNFAVORABLE_PRIOR in options:
            self.unfavorable_prior = True

        if ops.UNFAVORABLE_ENV in options:
            self.unfavorable_env = True

        if ops.FORGET_RATE in options:
            self.forget_rate = True

        if ops.ANHEDONIA in options:
            self.anhedonia = True
