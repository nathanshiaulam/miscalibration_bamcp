import numpy as np
import math
import ast
import random
import fileinput, sys
from collections import defaultdict

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

""" TO DO: 

    - SET OVER_GENERALIZE STRATEGY: 
    -- INCLUDE WEAK ARM WITH BOOSTED ALPHA
    -- MISCALIBRATE PRIORS ACCORDINGLY WITH BETA VALUES 
    -- OVERGENEREALIZATION PREVENTS MODEL FROM DISCOVERING

""" 

def main():

    args = sys.argv
    vals = fetchValues(args)

    options_desc = fetchOpsDesc(vals[params.OPTIONS])
    env_desc = fetchEnvDesc(vals[params.ENV], vals[params.P_ARR])

    printHeader(env_desc, options_desc, vals)

    bad_prior_alpha = np.copy(ops.UNFAVORABLE_PRIOR_ALPHA)
    bad_prior_beta = np.copy(ops.UNFAVORABLE_PRIOR_BETA)
    bad_prior_max = np.copy(np.argmax(vals[params.ALPHA]) + vals[params.NUM_STEPS])

    """ EXTRACT VALUES FROM VALS """
    cost = vals[params.COST]
    forget_rate = vals[params.FORGET_RATE]
    environment = vals[params.ENV]
    options = vals[params.OPTIONS]

    """ COMPARES CHOICE AGAINST GITTINS """
    if envs.TEST_GITTINS == environment:
        vals[params.NUM_STEPS] = 1
        testGittins(vals)

    if envs.NORM_BANDITS == environment:
        
        if ops.OVER_GENERALIZE in options:
            simulateOvergeneralization(vals)

        elif ops.FORGET_RATE in options:
            cost = .3
            vals[params.COST] = cost 

            beta_reward = defaultdict(list)
            beta_accuracy = defaultdict(list)

            vals[params.FORGET_RATE] = forget_rate
            vals[params.BAD_PRIOR] = (bad_prior_alpha, bad_prior_beta)
            ans = bernoulliBandits(vals)
            
            print "(cost,beta,forget_rate): (%s, %s, %s)" % (str(cost), str(bad_prior), str(forget_rate))
            print "(reward,accuracy): %s" % str(ans)

            sys.stdout.flush()

            beta_reward[cost].append((forget_rate, ans[0]))
            beta_accuracy[cost].append((forget_rate, ans[1]))

            generateFigs(beta_reward, beta_accuracy, options)

        elif ops.DO_NOTHING in options:
            simulateDoNothing(vals)


def simulateOvergeneralization(vals):

    start_state = 0

    """ Initial miscalibration for each state-action pair """
    bad_prior_alpha = np.copy(vals[params.BAD_PRIOR][0])
    bad_prior_beta = np.copy(vals[params.BAD_PRIOR][1])

    bad_prior_val = np.argmax(bad_prior_alpha[start_state])
    bad_prior_max = ops.UNFAVORABLE_PRIOR_ALPHA_MAX

    alpha_reward = {}
    alpha_accuracy = {}

    alpha_reward_trial = defaultdict(list)
    alpha_accuracy_trial = defaultdict(list)

    cost = vals[params.COST]

    while bad_prior_val <= bad_prior_max:

        ans = bernoulliBandits(vals)

        print "(cost): (%s)" % str(vals[params.COST])
        print "Alpha: (%s) | Beta: (%s)" % (str(bad_prior_alpha), str(bad_prior_beta))
        print "(reward,accuracy): %s" % str(ans)
        sys.stdout.flush()

        """ ALPHA -> REWARD """
        alpha_reward[bad_prior_val] = ans[0]

        """ ALPHA -> ACCURACY"""
        alpha_accuracy[bad_prior_val] = ans[1]

        """ ALPHA -> LIST OF ACCURACY PER TRIAL """
        alpha_accuracy_trial[bad_prior_val] = ans[2]

        """ ALPHA -> LIST OF REWARD PER TRIAL """
        alpha_reward_trial[bad_prior_val] = ans[3]

        """ Increment alpha only for unfavorable bandit """
        bad_prior_val += ops.UNFAVORABLE_PRIOR_ALPHA_FACTOR
        bad_prior_alpha[0][len(bad_prior_alpha[0]) - 1] = bad_prior_val

        vals[params.BAD_PRIOR] = (bad_prior_alpha, bad_prior_beta)

    """ CALCULATE STD DEV FOR EACH BETA """ 
    stddev_acc = {}
    stddev_reward = {}

    for k, v in alpha_accuracy_trial.iteritems():
        acc_list = np.array(v)
        stddev_acc[k] = np.std(acc_list)
    for k, v in alpha_reward_trial.iteritems():
        reward_list = np.array(v)
        stddev_reward[k] = np.std(reward_list)

    plotBadPriors(alpha_reward, alpha_accuracy, stddev_acc.values(), stddev_reward.values(), vals)

def simulateDoNothing(vals):
    last_two = [1.0, 1.0]

    start_state = 0

    """ Initial miscalibration for each state-action pair """
    bad_prior_alpha = np.copy(vals[params.BAD_PRIOR][0])
    bad_prior_beta = np.copy(vals[params.BAD_PRIOR][1])

    bad_prior_val = np.argmax(bad_prior_beta[start_state])
    bad_prior_max = ops.UNFAVORABLE_PRIOR_BETA_MAX

    beta_reward = {}
    beta_accuracy = {}

    beta_reward_trial = defaultdict(list)
    beta_accuracy_trial = defaultdict(list)

    while bad_prior_val <= bad_prior_max:

        ans = bernoulliBandits(vals)

        print "(cost): (%s)" % str(vals[params.COST])
        print "Alpha: (%s) | Beta: (%s)" % (str(bad_prior_alpha), str(bad_prior_beta))
        print "(reward,accuracy): %s" % str(ans)
        sys.stdout.flush()

        """ OPTIMIZATION: Stops searching when you have consecutively
            low reward values of 0 """
        last_two[1] = last_two[0]
        last_two[0] = ans[1]

        """ BETA -> REWARD """
        beta_reward[bad_prior_val] = ans[0]

        """ BETA -> ACCURACY"""
        beta_accuracy[bad_prior_val] = ans[1]

        """ BETA -> LIST OF ACCURACY PER TRIAL """
        beta_accuracy_trial[bad_prior_val] = ans[2]

        """ BETA -> LIST OF REWARD PER TRIAL """
        beta_reward_trial[bad_prior_val] = ans[3]

        if last_two[0] == 0 and last_two[1] == 0:
            num_left = (bad_prior_max - bad_prior_val) / ops.UNFAVORABLE_PRIOR_BETA_FACTOR
            for i in range(num_left):
                beta_reward[bad_prior_val] = 0.0
                beta_accuracy[bad_prior_val] = 0.0
                bad_prior_val += ops.UNFAVORABLE_PRIOR_BETA_FACTOR

        """ Increment beta for all bandits except for do-nothing """
        bad_prior_val += ops.UNFAVORABLE_PRIOR_BETA_FACTOR
        for i in range(len(bad_prior_beta[0]) - 1):
            bad_prior_beta[0][i] = bad_prior_val

        vals[params.BAD_PRIOR] = (np.array([[0,0]]), bad_prior_beta)

    """ CALCULATE STD DEV FOR EACH BETA """ 
    stddev_acc = {}
    stddev_reward = {}

    for k, v in beta_accuracy_trial.iteritems():
        acc_list = np.array(v)
        stddev_acc[k] = np.std(acc_list)
    for k, v in beta_reward_trial.iteritems():
        reward_list = np.array(v)
        stddev_reward[k] = np.std(reward_list)

    plotBadPriors(beta_reward, beta_accuracy, stddev_acc.values(), stddev_reward.values(), vals)

def plotBadPriors(beta_reward, beta_acc, stddev_acc, stddev_reward, vals):

    """ EXTRACT VALUES FOR SIMULATION """ 
    cost = vals[params.COST]
    discount = vals[params.DISCOUNT]
    epsilon = vals[params.EPSILON]
    num_sims = vals[params.NUM_SIMS]
    num_steps = vals[params.NUM_STEPS]

    """ SET TITLE LABELS """ 
    title_reward = r'$\beta$' + " vs. Reward | Miscalibrated Prior | Cost=%f \n $\gamma$=%f | $\epsilon$=%f | Simulations=%d" % (cost, discount, epsilon, num_sims)
    title_acc = r'$\beta$' + " vs. Accuracy | Miscalibrated Prior | Cost=%f \n $\gamma$=%f | $\epsilon$=%f | Simulations=%d" % (cost, discount, epsilon, num_sims)

    """ SET AXIS LABELS """
    y_axis_reward = "Reward"
    y_axis_acc = "P(Choose Stochastic Arm)"
    x_axis = r'$ \beta $'

    """ EXTRACT VALUES TO PLOT """
    x_list = beta_reward.keys()
    y_list_reward = beta_reward.values()
    y_list_acc = beta_acc.values()

    """ SET AXES FOR PLOTTING """
    axis_reward = [0, num_steps, -5, 30]
    axis_acc = [0, num_steps, 0, 1.1]

    """ SET FILENAMES OF SAVED FIGURES """
    save_string_acc = 'cost%f_reward.png' % (cost)
    save_string_reward = 'cost%f_acc.png' % (cost)

    """ PLOT BETA VS. ACCURACY WITH ERROR """
    fg_acc = plt.figure()
    plt.title(title_acc)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis_acc)
    plt.axis(axis_acc)
    errorfill(x_list, y_list_acc, stddev_acc)

    """ PLOT BETA VS. REWARD WITH ERROR """
    fg_reward = plt.figure()
    plt.title(title_reward)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis_reward)
    plt.axis(axis_reward)
    errorfill(x_list, y_list_reward, stddev_reward)

    fg_acc.show()
    fg_reward.show()

def fetchValues(args):

    options = list(args[1])
    environment = args[2]

    arr = ast.literal_eval(args[3])
    if ops.DO_NOTHING in options:
        arr.append(1) # Append known do-nothing bandit
    if ops.OVER_GENERALIZE in options:
        arr.append(ops.OVER_GENERALIZE_PROB) # Append weak bandit with high alpha

    """ TRUE PROBABILITY OF SUCCESS FOR EACH ACTION """
    p_arr = np.array(arr) 

    """ PARAMS FOR BAMCP """
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
        params.COST : ops.SAMPLE_COST_CONST,
        params.BAD_PRIOR : (ops.UNFAVORABLE_PRIOR_BETA, ops.UNFAVORABLE_PRIOR_BETA),
        params.FORGET_RATE : ops.FORGET_RATE_EPSILON,
    }

    return vals

def printHeader(env_desc, options_desc, vals):
    print "SIMULATE WITH PROPERTIES:"
    print "---------------------------"
    print "BAMCP with ENV: %s" % env_desc
    print "Miscalibration with Options: %s" % str(options_desc)
    print "BAMCP Settings: Discount: %f, Epsilon: %f, Num_Sims: %d, Num_Steps: %d, Num_Trials: %d" % (vals[params.DISCOUNT], vals[params.EPSILON], vals[params.NUM_SIMS], vals[params.NUM_STEPS], vals[params.NUM_TRIALS])
    print "BAMCP Priors: (Alpha=%s, Beta=%s)" % (str(vals[params.ALPHA]), str(vals[params.BETA]))
    print "True bandit probabilities: %s" % str(vals[params.P_ARR])
    print "---------------------------"

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
        if ops.DO_NOTHING == op:
            options_desc.append(ops.DO_NOTHING_DESC)
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
    
    """ LOAD IN GITTINS FILE """
    matrix = loadGittins("../data/gittins_indices.txt")

    alpha_vals = []
    beta_vals = []
    accuracy_vals = []
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix)):
            
            gittins_index = float(matrix[i][j])
            alpha = np.array([[0, j + 1]])
            beta = np.array([[0, i + 1]])
            vals[params.ALPHA] = alpha
            vals[params.BETA] = beta

            print "Gittins Index: " + str(gittins_index)
            print "(Alpha, Beta): " + str((alpha[0][1], beta[0][1]))
            
            alpha_vals.append(alpha[0][1])
            beta_vals.append(beta[0][1])

            correct = 0
            incorrect = 0

            sys.stdout.flush()
            for k in range(0, vals[params.NUM_TRIALS]):
                trial_correct = 0
                trial_incorrect = 0
                bamcp = BAMCP(vals)
                for step in range(vals[params.NUM_STEPS]):
                    val = bamcp.search(vals[params.NUM_SIMS], 0)[0]
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

            accuracy = float(correct) / float(vals[params.NUM_TRIALS])
            accuracy_vals.append(accuracy)
            print "Percent Correct: " + str(accuracy)
            sys.stdout.flush()

    f = plt.figure(0)
 
    plt.title("BAMCP - Alpha/Beta vs. Accuracy \n $\gamma$=%f | $\epsilon$=%f | Simulations=%d" % (vals[params.DISCOUNT], vals[params.EPSILON], vals[params.NUM_SIMS]))
    plt.scatter(alpha, beta, c=vals, s=2000, marker='s', vmin=0, vmax=1)
    plt.gray()
    plt.colorbar(label="Probability of correct decision")
    plt.xlabel(r'$ \alpha $')
    plt.ylabel(r'$ \beta $')

    f.savefig('../figures/gittins_choice.png')


def bernoulliBandits(vals):

    correct = 0
    incorrect = 0

    wins = np.zeros(shape=(1, len(vals[params.P_ARR])))
    trials = np.zeros(shape=(1, len(vals[params.P_ARR])))
    beta = np.zeros(shape=(1, len(vals[params.P_ARR])))

    tot_reward = 0

    """ List of accuracy for each trial """
    trial_acc_list = []

    """ List of accuracy for each trial """
    trial_reward_list = []

    zero_bandit = np.argmax(vals[params.P_ARR])

    for k in range(0, vals[params.NUM_TRIALS]):
        trial_correct = 0
        trial_incorrect = 0
        bamcp = BAMCP(vals)
        for step in range(vals[params.NUM_STEPS]):
            val = bamcp.search(vals[params.NUM_SIMS], 0)[0]
            val_p = vals[params.P_ARR][val]
            if val_p != 1:
                trial_correct += 1
            else: 
                trial_incorrect += 1
            sys.stdout.flush()

        """ RECORD ALPHA/BETA """
        wins += bamcp.wins
        beta += bamcp.beta

        trials += bamcp.trials

        """ RECORD TOTAL_REWARD """
        tot_reward += bamcp.total_reward

        avg_correct = float(trial_correct) / float(vals[params.NUM_STEPS])
        avg_incorrect = float(trial_incorrect) / float(vals[params.NUM_STEPS])

        correct += avg_correct
        incorrect += avg_incorrect

        """ RECORD ACCURACY AND REWARD PER TRIAL """
        trial_acc_list.append(avg_correct)
        trial_reward_list.append(bamcp.total_reward)

    avg_reward = float(tot_reward) / float(vals[params.NUM_TRIALS])
    percent_correct = float(correct) / float(vals[params.NUM_TRIALS])

    print "RESULTS ------------->"
    print "Alpha: %s"  % str(wins / vals[params.NUM_TRIALS]) 
    print "Beta: %s"  % str(beta / vals[params.NUM_TRIALS]) 
    print "Trials: %s" % str(trials / vals[params.NUM_TRIALS])

    return (avg_reward, percent_correct, trial_acc_list, trial_reward_list)
    sys.stdout.flush()


def errorfill(x, y, yerr, color="blue", alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = np.array(y) - np.array(yerr)
        ymax = np.array(y) + np.array(yerr)
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
if __name__ == "__main__":
    main()