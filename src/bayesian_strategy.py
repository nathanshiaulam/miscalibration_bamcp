from pymc import rbeta
import numpy as np
import sys
import bandits

rand = np.random.rand

def main():
	args = sys.argv
	num_sims = int(args[1])

	# Probability for ea. bandit
	p_arr = np.array([.5,.5, .4]) 
	b = bandits.Bandits(p_arr) 

	strat = BayesianStrategy(b)
	strat.sample_bandits(num_sims)


class BayesianStrategy:
	"""
	Implements a online, learning strategy to solve
	the Multi-Armed Bandit problem.
	
	parameters:
		bandits: a Bandit class with .pull method
	
	methods:
		sample_bandits(n): sample and train on n pulls.

	attributes:
		N: the cumulative number of samples
		choices: the historical choices as a (N,) array
		bb_score: the historical score as a (N,) array
	"""
	
	def __init__(self, bandits):
		
		self.bandits = bandits
		n_bandits = len(self.bandits)
		self.wins = np.zeros(n_bandits)
		self.trials = np.zeros(n_bandits)
		self.N = 0

	def sample_bandits(self, n):
		
		bb_score = np.zeros(n)
		choices = np.zeros(n)
		
		for k in range(n):
			#sample from the bandits's priors, and select the largest sample
			print "wins: " + str(self.wins)
			print "trials: " + str(self.trials)
			dist = rbeta(1 + self.wins, 1 + self.trials - self.wins)
			choice = np.argmax(dist)
			print "dist: " + str(dist)
			print "choice: " + str(choice)
			
			#sample the chosen bandit
			result = self.bandits.pull(choice)
			
			#update priors and score
			self.wins[choice] += result
			self.trials[choice] += 1

			bb_score[k] = result 
			self.N += 1
			choices[k] = choice
		
		return 

if __name__ == "__main__":
	main()
