class History:

    def __init__(self, state_counts, action_counts):

        self.state_counts = state_counts
        self.action_counts = action_counts

    def updateHist(self, state, action):
        self.state_counts[state] += 1
        self.action_counts[int(action)] += 1

    def getActionCounts(self):
        a_count = list(self.action_counts)
        return a_count

    def getStateCounts(self):
        s_count = list(self.state_counts)
        return s_count

    def toTup(self):
        return (tuple(self.state_counts), tuple(self.action_counts))

    def __str__(self):
        return "(State: " + str(self.state_counts) + ", Action: " + str(self.action_counts) + ")"
