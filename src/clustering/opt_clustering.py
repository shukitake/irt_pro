import sys
import numpy as np
import pandas as pd

sys.path.append("/Users/shukitakeuchi/irt_pro/src")

from clustering.emalgorithm import EM_Algo


class Opt_clustering:
    def __init__(self, U, Y, V, N, T):
        # 初期設定
        self.U = U
        self.init_Y = Y
        self.V = V
        self.N = N
        self.T = T
        self.I, self.J = np.shape(self.U)

    def opt(self):
        em_algo = EM_Algo(self.U, self.init_Y, self.V, self.N, self.T)
        W, V = em_algo.repeat_process()
        return V
