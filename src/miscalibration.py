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
        self.trembling_hand = False
        self.do_nothing = False

        self._setFlags(bamcp.options)

    def miscalibratePriors(self, bad_priors):
        if not self.bamcp.test_gittins:

            """ MISCALIBRATE PRIOR FROM CONFIG """
            if self.unfavorable_prior:
                self.bamcp.alpha = np.copy(bad_priors[0])
                self.bamcp.beta = np.copy(bad_priors[1])

            # if self.do_nothing:
            #     zero_bandit = self.bamcp.bandits.p[len(self.bamcp.bandits) - 1]
            #     for a in range(self.bamcp.num_actions):
            #         if a != zero_bandit:
            #             self.bamcp.beta[self.bamcp.start_state][a] += beta_val
            print "Miscalibrated Priors: (Alpha, Beta): (%s, %s)" % (str(self.bamcp.alpha), str(self.bamcp.beta))
            print "Miscalibrated Bandit Cost: %f" % self.bamcp.bandit_cost

        self.bamcp.wins = np.copy(self.bamcp.alpha)
        self.bamcp.trials = np.copy(self.bamcp.alpha + self.bamcp.beta)

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

    def miscalibrateAction(self, state, action, win):

        """ 
        MISCALIBRATION: Implements a forget rate for 
        (alpha, beta) values by implementing an epsilon
        discount 
        """
        if self.forget_rate:
            self.bamcp.alpha = (1 - self.bamcp.forget_rate) * self.bamcp.alpha + self.bamcp.forget_rate * ops.FORGET_RATE_ALPHA 
            self.bamcp.beta = (1 - self.bamcp.forget_rate) * self.bamcp.beta + self.bamcp.forget_rate * ops.FORGET_RATE_BETA

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
                zero_bandit = self.bamcp.bandits.p[len(self.bamcp.bandits) - 1]
                for a in range(self.bamcp.num_actions):
                    if a != zero_bandit:
                        self.bamcp.beta[state][a] += ops.UNFAVORABLE_ENV_FACTOR

        """ 
        MISCALIBRATION: Generalizes success/failure
        of bandit to all other bandits
        """
        if self.overgeneralize:
            for a in range(self.bamcp.num_actions):
                if a != action:
                    if win:
                        self.bamcp.alpha[state][a] += 1
                    else:
                        self.bamcp.beta[state][a] += 1

      
    def miscalibratePull(self, action, dist, cost):
        if self.trembling_hand:
            if rand() < (1 - ops.TREMBLING_HAND_CONST):
                action = np.random.choice(self.bamcp.bandits.p)
        if self.sample_cost:
            zero_bandit = self.bamcp.bandits.p[len(self.bamcp.bandits) - 1]
            if action == zero_bandit:
                return 0
            else:
                if rand() <= dist[action]:
                    return 1 - cost
                else:
                    return -1 * cost
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

        if ops.DO_NOTHING in options:
            self.do_nothing = True