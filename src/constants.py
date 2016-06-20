import numpy as np
class Params:

	OPTIONS = 'options'
	P_ARR = 'p_arr'
	DISCOUNT = 'discount'
	EPSILON = 'epsilon'
	NUM_SIMS = 'num_sims'
	NUM_STEPS = 'num_steps'
	NUM_TRIALS = 'num_trials'
	ENV = 'env'
	ALPHA = 'alpha'
	BETA = 'beta'
	VERBOSE = 'verbose'
	LEARN_RATE = .2
	EXPLORE_CONST = 20
	COST = 'cost'
	BAD_PRIOR = 'bad_prior'
	FORGET_RATE = 'forget_rate'

class Envs:

	# Bernoulli Bandit variations
	TEST_GITTINS = 'g'
	TEST_GITTINS_DESC = "N-armbed Bernoulli Bandits testing against Gittins values"
	
	NORM_BANDITS = 'b'
	NORM_BANDITS_DESC = 'N-armed Bernoulli Bandits'

	# MDP Variations

class Flags:

	"""FLAGS FOR MODEL"""
	NONE = 'n' 

	# Wrong Problem
	SAMPLE_COST = 'c'
	SAMPLE_COST_DESC = 'Sets a cost to sampling a bandit, with known bandit having a value of 0.'
	SAMPLE_COST_CONST = 0.0
	SAMPLE_COST_FACTOR = 0.05

	OVER_GENERALIZE = 'g'
	OVER_GENERALIZE_DESC = 'Over generalize action outcome'
	OVER_GENERALIZE_PROB = .1
	OVER_GENERALIZE_ALPHA = 100

	UNFAVORABLE_PRIOR = 'p'
	UNFAVORABLE_PRIOR_DESC = 'Unfavorable prior for non-zero bandit(s)'
	UNFAVORABLE_PRIOR_MAX = 10 	# Preferred: np.argmax(alpha) + num_steps
	UNFAVORABLE_PRIOR_FACTOR = 2
	UNFAVORABLE_PRIOR_ALPHA = np.array([[0, 0]])
	UNFAVORABLE_PRIOR_BETA = np.array([[0, 0]])

	# Wrong Inference
	FORGET_RATE = 'f'
	FORGET_RATE_DESC = 'Increased forgetting-rate for a bandit'
	FORGET_RATE_EPSILON = .01
	FORGET_RATE_ALPHA = np.array([[0, 0]])
	FORGET_RATE_BETA = np.array([[60, 0]])

	# Wrong Environment
	UNFAVORABLE_ENV = 'e'
	UNFAVORABLE_ENV_DESC = 'Start agent off in unfavorable environment, then switch to normal one.'
	UNFAVORABLE_ENV_TIME = 10
	UNFAVORABLE_ENV_FACTOR = 100

	TREMBLING_HAND = 't'
	TREMBLING_HAND_DESC = 'Induce trembling hand- agent selects action at random with probability (1 - p)'
	TREMBLING_HAND_CONST = .9

	DO_NOTHING = 'd'
	DO_NOTHING_DESC = 'Include a do-nothing bandit that returns 0 reward with probability 1'
	DO_NOTHING_BETA = 0
