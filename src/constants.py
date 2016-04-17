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

	# Bernoulli Bandit variations
	DETERMINISTIC = 'd'
	DETERMINSITC_DESC = 'N-armed Bernoulli Bandits with a deterministic arm'

	NORM_BANDITS = 'b'
	NORM_BANDITS_DESC = 'N-armed Bernoulli Bandits'

	# MDP Variations

class Flags:
	"""FLAGS FOR MODEL"""
	NONE = 'n' 

	# Wrong Problem
	SAMPLE_COST = 'c'
	SAMPLE_COST_DESC = 'Sets a cost to sampling a bandit, with known bandit having a known value of 0.'
	SAMPLE_COST_CONST = 100

	OVER_GENERALIZE = 'g'
	OVER_GENERALIZE_DESC = 'Over generalize action outcome'

	UNFAVORABLE_PRIOR = 'p'
	UNFAVORABLE_PRIOR_DESC = 'Unfavorable prior for strongest bandit'
	UNFAVORABLE_BETA_CONST = 1000

	# Wrong Inference




	# Wrong Environment