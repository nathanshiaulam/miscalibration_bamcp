import numpy as np
import math
import ast
import random
import fileinput, sys

import matplotlib.pyplot as plt
from matplotlib.pyplot import show

import bandits
import node
import history
from bamcp import BAMCP

from constants import Flags as ops
from constants import Params as params
from constants import Envs as envs

rand = np.random.rand
def main():
    args = sys.argv

    options = list(args[1])
    environment = args[2]

    arr = ast.literal_eval(args[3])
    p_arr = np.array(arr) 

    discount = float(args[4])
    epsilon = float(args[5])
    num_sims = int(args[6])
    num_steps = int(args[7])
    num_trials = int(args[8])

    """ OPTIONAL ARGS """
    alpha = np.zeros(shape=(1, len(p_arr)))
    beta = np.zeros(shape=(1, len(p_arr)))
    verbose = 0

    if len(args) > 9:
        alpha = np.array(ast.literal_eval(args[9]))
    if len(args) > 10:
        beta = np.array(ast.literal_eval(args[10]))
    if len(args) > 11:
        verbose = int(args[11])


    options_desc = fetchOpsDesc(options)
    env_desc = fetchEnvDesc(environment, p_arr)

    print "SIMULATE WITH PROPERTIES:"
    print "---------------------------"
    print "BAMCP with ENV: %s" % env_desc
    print "Miscalibration with Options: %s" % str(options_desc)
    print "BAMCP Settings: Discount: %f, Epsilon: %f, Num_Sims: %d, Num_Steps: %d, Num_Trials: %d" % (discount, epsilon, num_sims, num_steps, num_trials)
    print "BAMCP Priors: (Alpha=%s, Beta=%s)" % (str(alpha), str(beta))
    print "True bandit probabilities: %s" % str(p_arr)
    print "---------------------------"

    vals = {
        params.ENV : environment,
        params.P_ARR : p_arr,
        params.DISCOUNT : discount,
        params.EPSILON : epsilon,
        params.OPTIONS : options,
        params.NUM_SIMS : num_sims,
        params.NUM_STEPS : num_steps,
        params.NUM_TRIALS : num_trials,
        params.ALPHA : alpha,
        params.BETA : beta, 
        params.VERBOSE : verbose,
    }

    if envs.TEST_GITTINS == environment:
        vals[params.NUM_STEPS] = 1
        testGittins(vals)

    if envs.NORM_BANDITS == environment:
        bernoulliBandits(vals)

def fetchEnvDesc(env, p_arr):
    if env == envs.TEST_GITTINS:
        assert 1 in p_arr, "At least one arm must be deterministic"
        return envs.TEST_GITTINS_DESC

    if env == envs.NORM_BANDITS:
        return envs.NORM_BANDITS_DESC


def fetchOpsDesc(options):
    options_desc = []
    for op in options:
        if ops.SAMPLE_COST == op:
            options_desc.append(ops.SAMPLE_COST_DESC)
        if ops.OVER_GENERALIZE == op:
            options_desc.append(ops.OVER_GENERALIZE_DESC)
        if ops.UNFAVORABLE_PRIOR == op:
            options_desc.append(ops.UNFAVORABLE_PRIOR_DESC)
        if ops.FORGET_RATE == op:
            options_desc.append(ops.FORGET_RATE_DESC)
        if ops.UNFAVORABLE_ENV == op:
            options_desc.append(ops.UNFAVORABLE_ENV_DESC)
        if ops.NONE == op:
            options_desc.append("None")
    return options_desc

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

def testGittins(vals):
    
    matrix = loadGittins("../data/gittins_indices.txt")

    for i in range(0, len(matrix)):
        for j in range(0, len(matrix)):
            
            gittins_index = float(matrix[i][j])
            alpha = np.array([[0, j + 1]])
            beta = np.array([[0, i + 1]])
            vals[params.ALPHA] = alpha
            vals[params.BETA] = beta
            print "Gittins Index: " + str(gittins_index)
            print "(Alpha, Beta): " + str((alpha[0][1], beta[0][1]))
 
            correct = 0
            incorrect = 0
            sys.stdout.flush()
            for k in range(0, vals[params.NUM_TRIALS]):
                trial_correct = 0
                trial_incorrect = 0
                bamcp = BAMCP(vals)
                for step in range(vals[params.NUM_STEPS]):
                    val = bamcp.search(vals[params.NUM_SIMS], 0)
                    if gittins_index <= .5:
                        if val == 0:
                            trial_correct += 1
                        else: 
                            trial_incorrect += 1
                    else:
                        if val == 0:
                            trial_incorrect += 1
                        else: 
                            trial_correct += 1
                    sys.stdout.flush()
                correct += float(trial_correct) / int(vals[params.NUM_STEPS])
                incorrect += float(trial_incorrect) / float(vals[params.NUM_STEPS])
            print "Percent Correct: " + str(float(correct) / float(vals[params.NUM_TRIALS]))
            sys.stdout.flush()

def bernoulliBandits(vals):

    correct = 0
    incorrect = 0

    wins = np.zeros(shape=(1, len(vals[params.P_ARR])))
    trials = np.zeros(shape=(1, len(vals[params.P_ARR])))

    max_bandit = np.argmax(vals[params.P_ARR])
    max_bandit_p = vals[params.P_ARR][max_bandit]
    for k in range(0, vals[params.NUM_TRIALS]):
        trial_correct = 0
        trial_incorrect = 0
        bamcp = BAMCP(vals)
        for step in range(vals[params.NUM_STEPS]):
            val = bamcp.search(vals[params.NUM_SIMS], 0)
            val_p = vals[params.P_ARR][val]
            if val_p == max_bandit_p:
                trial_correct += 1
            else: 
                trial_incorrect += 1
            sys.stdout.flush()
        wins += bamcp.wins
        trials += bamcp.trials
        correct += float(trial_correct) / int(vals[params.NUM_STEPS])
        incorrect += float(trial_incorrect) / float(vals[params.NUM_STEPS])

    if vals[params.ENV] != envs.DETERMINISTIC:
        print "Percent Correct: " + str(float(correct) / float(vals[params.NUM_TRIALS]))
    
    print "Wins: %s" % str(wins)
    print "Trials: %s"  % str(trials)

    sys.stdout.flush()

def analyze_choice_data(filename, options):
    lines = []
    with open(filename) as f:
        lines = f.readlines()

    stats = {}
    accuracy = 0
    num = 0
    tup = ()
    gittins = 0

    gittins_val = {}
    scores = {}

    all = string.maketrans('','')
    nodigs = all.translate(all, string.digits)

    for line in lines:
        tokens = line.split(" ")
        if tokens[0] == "Gittins":

            gittins = float(tokens[2])

        if tokens[0] == "(Alpha,":

            alpha = tokens[2].translate(all, nodigs)
            beta = tokens[3].translate(all, nodigs)
            tup = (int(alpha), int(beta))

            gittins_val[tup] = gittins

        if tokens[0] == "Percent":
            val = float(tokens[2])

            scores[tup] = val


    scores = collections.OrderedDict(sorted(scores.items()))
    alpha = []
    beta = []
    vals = []

    for k, v in scores.iteritems():
        alpha.append(k[0])
        beta.append(k[1])
        vals.append(v)

    f = plt.figure(0)

    plt.title("BAMCP - Number of Simulations: 5000")
    plt.scatter(alpha, beta, c=vals, s=2000, marker='s', vmin=0, vmax=1)
    plt.gray()
    plt.colorbar()
    plt.xlabel("Alpha")
    plt.ylabel("Beta")

    f.savefig('../figures/gittins_choice.png')

    raw_input()

if __name__ == "__main__":
    main()