import sys
import numpy as np

sys.path.append("/Users/shukitakeuchi/irt_pro/src")
from util.log import LoggerUtil


class Opt_Z:
    def __init__(self, U, T):
        self.U = U
        self.I, self.J = np.shape(self.U)
        self.T = T
        self.logger = LoggerUtil.get_logger(__name__)
        return

    def Est_Diff_Rank(self, X):
        X_sum_t = np.sum(X, axis=1)
        # self.logger.info(f"X_sum_t{X_sum_t}")
        Z = np.zeros((self.J, self.J), dtype=int)
        index = np.argsort(X_sum_t)
        # self.logger.info(f"index{index}")
        for j in range(len(index)):
            Z[index[j], j] = int(1)
            # self.logger.info(f"{j},{index[j]}")
        return Z
