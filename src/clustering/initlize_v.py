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
        V = np.zeros((self.J, self.N), dtype=int)
        index = np.argsort(Z)
        C = np.array_split(index, self.N)
        for n in range(len(C)):
            for j in C[n]:
                V[j, n] = int(1)
            # self.logger.info(f"{j},{index[j]}")
        return V
