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

    stats = {}
    key = ""
    accuracy = 0
    num = 0
    tup1 = ()
    tup2 = ()

    gittins_val = {}
    tups = {}

    all = string.maketrans('','')
    nodigs = all.translate(all, string.digits)

    for line in lines:
        tokens = line.split(" ")
        if tokens[0] == "Gittins":

            gittins = float(tokens[2])

        if tokens[0] == "(Alpha,":

            key += line.rstrip()
            alpha = tokens[2].translate(all, nodigs)
            beta = tokens[3].translate(all, nodigs)

            tup = (int(alpha), int(beta))
            gittins_val[tup] = gittins
        if tokens[0] == "Percent:

            val = float(tokens[3])
            correct = val / 10

            tups[tup] = correct
            stats[key] = correct

            key = ""
            num += 1
            accuracy += correct

    tups = collections.OrderedDict(sorted(tups.items()))
    alpha = defaultdict(list)
    beta = defaultdict(list)
    for k,v in tups.iteritems():
        tup2 = k[1]
        alpha[k[0]].append(tup2[0])
        beta[k[0]].append(tup2[1])


    i = 1
    for k, v in alpha.iteritems():
        f = plt.figure(i)
        plt.title("Comparing against " + str(k) + " with Gittins Index: " + str(float(gittins_val[k]) * .0001))
        x = alpha[k]
        y = beta[k]
        vals = []
        for i in range(0, len(x)):
            vals.append(tups[(k, (x[i], y[i]))])
        plt.scatter(x, y, c=vals, s=2000, marker="s", vmin=0, vmax=1)
        plt.gray()
        plt.colorbar()
        plt.xlabel("Alpha")
        plt.ylabel("Beta")
        i += 1
        f.show()
    raw_input()


    ave_acc = accuracy / num
    print num
    print "Average Accuracy: " + str(ave_acc)




if __name__ == "__main__":
    main()