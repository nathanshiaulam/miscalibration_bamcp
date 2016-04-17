class QNode:

    def __init__(self, hist, state, action):
        self.hist = hist.toTup()
        self.state = state
        self.action = action


    def __hash__(self):
        return hash((self.hist, self.state, self.action))

    def __eq__(self, other):
        return hash(self) == other

    def __ne__(self, other):
        return not(self == other)

    def __str__(self):
        return "QNODE: (Hist: " + str(self.hist) + ", Action: " + str(self.action) + ")"

    # def updateCount(self):
    #     self.count += 1

    # def updateValue(value):
    #     self.value = value

class VNode:

    def __init__(self, hist, state):
        self.hist = hist.toTup()
        self.state = state

    def __hash__(self):
        return hash((self.hist, self.state))

    def __eq__(self, other):
        return hash(self) == other

    def __ne__(self, other):
        return not(self == other)

    def __str__(self):
        return "VNODE: (Hist: " + str(self.hist) + ")"


    # def updateCount(self):
    #     self.count += 1
