import sys
import numpy as np

sys.path.append("/Users/shukitakeuchi/irt_pro/src")

from util.log import LoggerUtil


class Opt_Init_V:
    def __init__(self, U, N, T):
        self.U = U
        self.I, self.J = np.shape(self.U)
        self.N = N
        self.T = T
        self.logger = LoggerUtil.get_logger(__name__)
        return

    def initialize_V(self, Z):
        list_index = []
        V = np.zeros((self.J, self.N))
        for j in range(self.J):
            k = np.argmax(Z[j, :])
            list_index.append(k)
        C = np.array_split(list_index, self.N)
        for n in range(self.N):
            for j in C[n]:
                V[j, n] = 1
        return V
