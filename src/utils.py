import numpy as np
import math
import ast
import random
import fileinput, sys
import collections
import matplotlib.mlab as mlab
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.pyplot import show
from matplotlib.pyplot import figtext
import string
import bandits
import node
import history
from bamcp import BAMCP

from constants import Flags as ops
from constants import Params as params
from constants import Envs as envs

def main():
    filename = sys.argv[1]
    # analyze_choice_data(filename)
    # overgeneralize_fig(filename)
    # analyze_forget_rate(filename)
    all = string.maketrans('','')
    nodigs = all.translate(all, string.digits)

    prior_filestem = '../data/bad_priors/results_'
    cost = 0.0
    zero_beta_reward = {}
    zero_beta_acc = {}

    """ FOR EACH COST """
    while cost < .95:
        cost_str = str(cost).translate(all, nodigs)
        cost_str = cost_str[1:3]
        if len(cost_str) == 1:
            cost_str += '0'
        tot_beta_reward = {}
        tot_beta_acc = {}

        tot_acc_error = defaultdict(list)
        tot_reward_error = defaultdict(list)
        for i in range(5):
            if i == 0:
                filename = prior_filestem + cost_str + '.txt'
                params = parse_parameters(filename)

                trials = params[4]
                vals = parse_priors_file(filename, cost, trials)

                beta_reward = vals[0]
                beta_acc = vals[1]

                for j in range(26):
                    beta = j * 2
                    if beta not in tot_beta_reward:
                        tot_beta_reward[beta] = 0
                        tot_beta_acc[beta] = 0
                    tot_beta_reward[beta] += beta_reward[beta]
                    tot_beta_acc[beta] += beta_acc[beta]
            elif cost != 0:
                filename = prior_filestem + cost_str + '_0' + str(i) + '.txt'
                params = parse_parameters(filename)

                trials = params[4]
                vals = parse_priors_file(filename, cost, trials)

                beta_reward = vals[0]
                beta_acc = vals[1]

                acc_error = vals[2]
                reward_error = vals[3]

                for j in range(26):
                    beta = j * 2
                    if beta not in tot_beta_reward:
                        tot_beta_reward[beta] = 0
                        tot_beta_acc[beta] = 0
                    tot_beta_reward[beta] += beta_reward[beta]
                    tot_beta_acc[beta] += beta_acc[beta]
                    
                    curr_reward_list = tot_reward_error[beta]
                    curr_acc_list = tot_acc_error[beta]
                    new_reward_list = reward_error[beta]
                    new_acc_list = acc_error[beta]



                    tot_reward_list = curr_reward_list + new_reward_list
                    tot_acc_list = curr_acc_list + new_acc_list

                    tot_reward_error[beta] = tot_reward_list
                    tot_acc_error[beta] = tot_acc_list


        for i in range(len(tot_beta_reward)):
            beta = i * 2
            if cost == 0:
                tot_beta_reward[beta] /= 5
                tot_beta_acc[beta] /= 5
                zero_beta_acc = tot_beta_acc
                zero_beta_reward = tot_beta_reward
            else:
                tot_beta_acc[beta] /= 45
                tot_beta_reward[beta] /= 45

        stderr_reward = {}
        stderr_acc = {}
        print "Total Acc Error: " + str(tot_acc_error)
        print "Total Reward Error: " + str(tot_reward_error)

        for k, v in tot_acc_error.iteritems():
            acc_list = np.array(v)
            stderr_acc[k] = np.std(acc_list)
        for k, v in tot_reward_error.iteritems():
            reward_list = np.array(v)
            stderr_reward[k] = np.std(reward_list)

        print "STDERR Reward: " + str(stderr_reward)
        print "STDERR Accuracy: " + str(stderr_acc)
        title = r'$\beta$' + " vs. Reward | Miscalibrated Prior | Cost=%f \n $\gamma$=%f | $\epsilon$=%f | Simulations=%d" % (cost, params[0], params[1], params[2])
        x_axis = r'$ \beta $'
        y_axis = "Reward"
        ax = [0, 50, -5, 30]
        x_list = tot_beta_reward.keys()
        y_list = tot_beta_reward.values()
        save_string = 'cost%f_reward.png' % (cost)
        fg = plt.figure()
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.axis(ax)
        if cost != 0:
            errorfill(x_list, y_list, stderr_reward.values())
        else:
            plt.plot(x_list, y_list)
        # fg.savefig(save_string)

        title = r'$\beta$' + " vs. Accuracy | Miscalibrated Prior | Cost=%f \n $\gamma$=%f | $\epsilon$=%f | Simulations=%d" % (cost, params[0], params[1], params[2])
        x_axis = r'$ \beta $'
        y_axis = "P(Choose Stochastic Arm)"
        ax = [0, 50, 0, 1.1]
        x_list = tot_beta_acc.keys()
        y_list = tot_beta_acc.values()
        save_string = 'cost%f_acc.png' % (cost)
        fg2 = plt.figure()
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.axis(ax)

        if cost != 0:
            errorfill(x_list, y_list, stderr_acc.values())
        else:
            plt.plot(x_list, y_list)       

        # fg2.savefig(save_string)
        tot_beta_reward = {}
        tot_beta_acc = {}
        cost += 0.05


def parse_parameters(filename):
    settings = 'BAMCP Settings:'

    all = string.maketrans('','')
    nodigs = all.translate(all, string.digits)

    discount = 0
    epsilon = 0
    num_sim = 0
    num_steps = 0
    num_trials = 0

    lines = []
    with open(filename) as f:
        lines = f.readlines()

    for line in lines:
        tokens = line.split(' ')
        if settings in line:
            discount = float(tokens[3][:len(tokens[3]) - 1])
            epsilon = float(tokens[5][:len(tokens[5]) - 1]) * .1
            num_sim = float(tokens[7][:len(tokens[7]) - 1]) 
            num_steps = float(tokens[9][:len(tokens[9]) - 1])
            num_trials = float(tokens[11][:len(tokens[11]) - 1])

    return (discount, epsilon, num_sim, num_steps, num_trials)

def parse_priors_file(filename, curr_cost, trials):
    curr_step = 'Acton:'
    cost = '(cost,beta):'
    vals = '(reward,accuracy):'

    all = string.maketrans('','')
    nodigs = all.translate(all, string.digits)

    trial_reward = 0
    trial_acc = 0.0
    step = 1
    
    curr_beta = 0
    acc_freq = {}
    
    acc_error = defaultdict(list)
    reward_error = defaultdict(list)
    beta_reward = {}
    beta_acc = {}

    lines = []
    with open(filename) as f:
        lines = f.readlines()

    for line in lines:
        tokens = line.split(' ')
        if curr_step in line:
            k = (curr_cost, curr_beta, step)

            choice = int(tokens[1])
            reward = float(tokens[4])

            if k not in acc_freq:
                acc_freq[k] = 0
            acc_freq[k] += choice 
            trial_reward += reward
            if choice == 0:
                trial_acc += 1
            step += 1

            if step == 50:
                trial_acc = float(trial_acc) / 50.0
                acc_error[curr_beta].append(trial_acc)
                reward_error[curr_beta].append(trial_reward)
                trial_acc =0
                trial_reward = 0
                step = 0

        if vals in line:
            beta_reward[curr_beta] = float(tokens[1][1:len(tokens[1]) - 2]) * float(trials)
            beta_acc[curr_beta] = float(tokens[2][:len(tokens[2]) - 2]) * float(trials)
            if trials == 5:
                for i in range(int(trials)):
                    acc_error[curr_beta].append(0)
                    reward_error[curr_beta].append(0)
            curr_beta += 2
            

    for i in range(len(beta_reward), 26):
        beta = i * 2
        beta_reward[beta] = 0
        beta_acc[beta] = 0
        for j in range(int(trials)):
            acc_error[beta].append(0)
            reward_error[beta].append(0)
   
    beta_reward = collections.OrderedDict(sorted(beta_reward.items()))
    beta_acc = collections.OrderedDict(sorted(beta_acc.items()))

    return (beta_reward, beta_acc, acc_error, reward_error)

def errorfill(x, y, yerr, color="blue", alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = np.array(y) - np.array(yerr)
        ymax = np.array(y) + np.array(yerr)
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def analyze_forget_rate(filename):
    settings = 'BAMCP Settings:'
    curr_step = 'Acton:'
    reset = 'RESULTS ------------->'
    cost = '(cost,beta):'
    vals = '(reward,accuracy):'

    lines = []
    with open(filename) as f:
        lines = f.readlines()

    choice_step = {}
    reward_step = {}

    step = 1

    for line in lines:
        tokens = line.split(' ')
        if curr_step in line:
            choice = int(tokens[1])
            reward = float(tokens[4])
            if step not in choice_step:
                choice_step[step] = 0
                reward_step[step] = 0
            choice_step[step] += choice
            reward_step[step] += reward 

            step += 1
            if step == 50:
                step = 0

    choice_list = []
    reward_list = []
    step_list = []
    for i in range(1, 50):
        choice_list.append(choice_step[i])
        reward_list.append(reward_step[i])
        step_list.append(i)

    choice_list = np.array(choice_list) / 5.0
    reward_list = np.array(reward_list) / 5.0

    plt.plot(step_list, choice_list)
    plt.axis([1,50,0,1])
    plt.show()
    raw_input()
    print choice_list
    print reward_list





def overgeneralize_fig(filename):

    settings = 'BAMCP Settings:'
    curr_step = 'Acton:'
    reset = 'RESULTS ------------->'
    cost = '(cost,beta):'
    vals = '(reward,accuracy):'

    all = string.maketrans('','')
    nodigs = all.translate(all, string.digits)

    discount = 0
    epsilon = 0
    num_sim = 0
    num_steps = 0
    num_trials = 0

    step = 1

    step_freq = {}
    reward_freq = {}
    cost_reward = {}
    cost_acc = {}


    curr_cost = 0.05
    lines = []
    with open(filename) as f:
        lines = f.readlines()

    trial_reward = 0
    trial_acc = 0
    acc_error = defaultdict(list)
    reward_error = defaultdict(list)
    for line in lines:
        tokens = line.split(' ')
        if settings in line:
            discount = float(tokens[3][:len(tokens[3]) - 1])
            epsilon = float(tokens[5][:len(tokens[5]) - 1]) * .1
            num_sim = float(tokens[7][:len(tokens[7]) - 1]) 
            num_steps = float(tokens[9][:len(tokens[9]) - 1])
            num_trials = float(tokens[11][:len(tokens[11]) - 1])
        if curr_step in line:
            k = (curr_cost, step)
            choice = int(tokens[1])
            reward = float(tokens[4])
            if k not in step_freq:
                step_freq[k] = 0
                reward_freq[k] = 0
            step_freq[k] += choice
            reward_freq[k] += reward 

            trial_reward += reward
            if choice == 0:
                trial_acc += 1 
            step += 1
            if step == 50:
                trial_acc = float(trial_acc) / 50.0
                acc_error[curr_cost].append(trial_acc)
                reward_error[curr_cost].append(trial_reward)
                trial_acc =0
                trial_reward = 0
                step = 0

        if vals in line:
            cost_reward[curr_cost] = float(tokens[1][1:len(tokens[1]) - 2])
            cost_acc[curr_cost] = float(tokens[2][:len(tokens[2]) - 2])
            curr_cost += 0.05

    cost_reward = collections.OrderedDict(sorted(cost_reward.items()))
    cost_acc = collections.OrderedDict(sorted(cost_acc.items()))

    stderr_reward = {}
    stderr_acc = {}

    for k, v in acc_error.iteritems():
        trials = np.array(v)
        stderr_acc[k] = np.std(trials)

    for k, v in reward_error.iteritems():
        trials = np.array(v)
        stderr_reward[k] = np.std(trials)

    stderr_acc = collections.OrderedDict(sorted(stderr_acc.items()))
    stderr_reward = collections.OrderedDict(sorted(stderr_reward.items()))

    title = "Cost vs. Reward | Overgeneralize | Simulations=%d \n $\gamma$=%f | $\epsilon$=%f | Steps=%d" % (num_sim, discount, epsilon, num_steps)
    x_axis = "Cost"
    y_axis = "Reward"
    ax = [.05, .5, -10, 50]
    x_list = cost_reward.keys()
    y_list = cost_reward.values()
    save_string = '../figures/overgeneralize/cost_reward_%f_%f_%d.png' % (discount, epsilon, num_trials)
    fg = plt.figure()
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.errorbar(x_list, y_list, stderr_reward.values())
    plt.axis(ax)
    fg.show()
    fg.savefig(save_string)

    title = "Cost vs. Accuracy | Overgeneralize | Simulations=%d \n $\gamma$=%f | $\epsilon$=%f | Steps=%d" % (num_sim, discount, epsilon, num_steps)
    x_axis = "Cost"
    y_axis = "P(Choose Stochastic Arm)"
    ax = [0.05, .5, 0, 1]
    x_list = cost_acc.keys()
    y_list = cost_acc.values()
    save_string = '../figures/overgeneralize/cost_acc_%f_%f_%d.png' % (discount, epsilon, num_trials)
    fg2 = plt.figure()
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.errorbar(x_list, y_list, stderr_acc.values())
    plt.axis(ax)
    fg2.show()
    fg2.savefig(save_string)

    raw_input()

def generateFigs(filename):

    miscal = []
    forget_rate_tag = 'Forget Rate'
    over_tag = 'Overgeneralize'
    env_tag = 'Unfavorable Environment'
    sub_dir = 'bad_prior'
    tags = ''

    for op in options:
        if op == ops.OVER_GENERALIZE:
            sub_dir = 'overgeneralize'
            miscal.append(over_tag)
        if op == ops.FORGET_RATE:
            sub_dir = 'forget_rate'
            miscal.append(forget_rate_tag)
        if op == ops.UNFAVORABLE_ENV:
            sub_dir = 'unfavorable_env'
            miscal.append(env_tag)

    for item in miscal:
        tags += item + ' '

    print sub_dir
    for k, v in beta_reward.iteritems():
        title = "Miscalibrated Prior vs. Reward | Cost: %f | %s" % (float(k), tags)
        x_axis = "Beta"
        ax = [0, (len(v) - 1) * 2, -10, 50]
        if over_tag in miscal:
            x_axis = "Cost"
            ax = [0, .75, -10, 50]
        if forget_rate_tag in miscal:
            x_axis = forget_rate_tag
            ax = [0, 1, -10, 50]
        y_axis = "Reward"
        save_string = "../figures/%s/%s_vs_reward_%f.png" % (sub_dir, sub_dir, float(k))
        x_list = []
        y_list = []

        for i in range(len(v)):
            x_list.append(v[i][0])
            y_list.append(v[i][1])

        fg = plt.figure()
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.plot(x_list, y_list)
        plt.axis(ax)
        fg.savefig(save_string)

    for k, v in beta_accuracy.iteritems():
        title = "Miscalibrated Prior vs. Accuracy | Cost: %f | %s" % (float(k), tags)
        x_axis = "Beta"
        ax = [0, (len(v) - 1) * 2, 0, 1]
        if forget_rate_tag in miscal:
            x_axis = forget_rate_tag
            ax = [0, 1, 0, 1]
        if over_tag in miscal:
            x_axis = "Cost"
            ax = [0, .75, 0, 1]
        y_axis = "Accuracy"
        save_string = "../figures/%s/%s_vs_accuracy_%f.png" % (sub_dir ,sub_dir, float(k))
        x_list = []
        y_list = []

        for i in range(len(v)):
            x_list.append(v[i][0])
            y_list.append(v[i][1])


        fg = plt.figure()
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.plot(x_list, y_list)
        plt.axis(ax)
        fg.savefig(save_string)

def analyze_choice_data(filename):
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
    gittins_score = {}
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

            gittins_score[gittins] = val
            scores[tup] = val


    scores = collections.OrderedDict(sorted(scores.items()))
    gittins_score = collections.OrderedDict(sorted(gittins_score.items()))

    alpha = []
    beta = []
    vals = []

    git = []
    othervals = []
    for k, v in gittins_score.iteritems():
        git.append(k)
        othervals.append(v)

    for k, v in scores.iteritems():
        alpha.append(k[0])
        beta.append(k[1])
        vals.append(v)

    f = plt.figure(0)
 
    plt.title("BAMCP - Alpha/Beta vs. Accuracy \n $\gamma$=.95 | $\epsilon$=.001 | Simulations=5000")
    plt.scatter(alpha, beta, c=vals, s=2000, marker='s', vmin=0, vmax=1)
    plt.gray()
    plt.colorbar(label="Probability of correct decision")
    plt.xlabel(r'$ \alpha $')
    plt.ylabel(r'$ \beta $')

    f2 = plt.figure(1)
    plt.plot(git, othervals)
    plt.axis([.2,1,0,1.1])
    plt.xlabel('Gittins Value')
    plt.ylabel('Accuracy')
    plt.title("BAMCP - Gittins Index vs. Accuracy \n $\gamma$=.95 | $\epsilon$=.001 | Simulations=5000")
    f.savefig('../figures/gittins_choice.png')
    f2.savefig('../figures/gittins_acc.png')
if __name__ == "__main__":
    main()