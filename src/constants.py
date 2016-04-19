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
	SAMPLE_COST_DESC = 'Sets a cost to sampling a bandit, with known bandit having a known value of 0.'
	SAMPLE_COST_CONST = .9

	OVER_GENERALIZE = 'g'
	OVER_GENERALIZE_DESC = 'Over generalize action outcome'

	UNFAVORABLE_PRIOR = 'p'
	UNFAVORABLE_PRIOR_DESC = 'Unfavorable prior for strongest bandit'
	UNFAVORABLE_PRIOR_CONST = 100

	ANHEDONIA = 'a'
	ANHEDONIA_EPSILON = .4

	# Wrong Inference
	FORGET_RATE = 'f'
	FORGET_RATE_DESC = 'Increased forgetting-rate for a bandit'
	FORGET_RATE_EPSILON = .5

	# Wrong Environment
	UNFAVORABLE_ENV = 'e'
	UNFAVORABLE_ENV_DESC = 'Start agent off in unfavorable environment, then switch to normal one.'
	UNFAVORABLE_ENV_TIME = 10
	UNFAVORABLE_ENV_FACTOR = 100
