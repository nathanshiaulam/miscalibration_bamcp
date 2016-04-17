import fileinput, sys
import string
import collections
import matplotlib.pyplot as plt
from collections import defaultdict

def main():

    args = sys.argv

    lines = []
    with open(args[1]) as f:
        lines = f.readlines()

    tup = ()
    gittins = 0

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

        if tokens[0] == "Percent":
            val = float(tokens[2])

            scores[tup] = val


    scores = collections.OrderedDict(sorted(scores.items()))
    alpha = []
    beta = []
    vals = []
    print scores
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

    f.savefig('gittins_choice.png')

    raw_input()


if __name__ == "__main__":
    main()