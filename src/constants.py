class Params:

	OPTIONS = 'options'
	P_ARR = 'p_arr'
	DISCOUNT = 'discount'
	EPSILON = 'epsilon'
	NUM_SIMS = 'num_sims'
	NUM_STEPS = 'num_steps'
	NUM_TRIALS = 'num_trials'
	ENV = 'env'

class Envs:

	DETERMINISTIC = 'd'
	NORM_BANDITS = 'b'

class Flags:
	"""FLAGS FOR MODEL"""
	NONE = 'n' 

	# Wrong Problem
	SAMPLE_COST = 'c'
	OVER_GENERALIZE = 'g'
	UNFAVORABLE_PRIOR = 'p'
	UNFAVORABLE_BETA_CONST = 100

	# Wrong Inference


	# Wrong Environment