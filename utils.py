import numpy as np

class repstats:
    def __init__(self):
        self.sizes = []
        self.perts = []
        self.states = []
        self.actions = []
        self.exps = []

    def add_size(self, sz, loc):
        size = sum(len(lst) for lst in sz.values())
        self.sizes.append(size) if loc else self.perts.append(size)
        # if not loc:print(self.perts[-1] - self.sizes[-1])
        # print(self.actions[-1])

    def add_actions(self, act):
        self.actions.append(act)

    def add_exps(self,exp ):
        self.exps.append(exp)

    def get_acts(self):
        x = np.array(self.actions)
        unique, counts = np.unique(x[:,0], return_counts=True)
        a = np.asarray((unique, counts)).T
        unique, counts = np.unique(x[:,1], return_counts=True)
        b = np.asarray((unique, counts)).T
        unique, counts = np.unique(x[:,2], return_counts=True)
        c = np.asarray((unique, counts)).T
        return [a, b, c]
    
    def get_exps(self):
        x = np.array(self.exps)
        unique, counts = np.unique(x[:,0:10], return_counts=True)
        a = np.asarray((unique, counts)).T
        unique, counts = np.unique(x[:,-10:-1], return_counts=True)
        b = np.asarray((unique, counts)).T
        return [a, b]

    def get_stats(self):
        a = np.mean(self.sizes)
        b = np.std(self.sizes)
        c = np.mean(self.perts)
        d = np.std(self.perts)
        return [a, b, c, d]