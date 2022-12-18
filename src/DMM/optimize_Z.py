import numpy as np


class Opt_Z:
    def __init__(self, U, T):
        self.U = U
        self.I, self.J = np.shape(self.U)
        self.T = T
        return

    def Est_Diff_Rank(self, X):
        X_sum_t = np.sum(X, axis=1)
        Z = np.zeros((self.J, self.J), dtype=int)
        index = np.argsort(X_sum_t)
        for j in range(len(index)):
            Z[index[j], j] = int(1)
        return Z
