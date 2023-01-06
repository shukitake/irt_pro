import numpy as np

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
        if self.N == self.J:
            V = np.identity(self.J)
        elif self.N == 1:
            V = np.ones((self.J, 1))
        else:
            em_algo = EM_Algo(self.U, self.init_Y, self.V, self.N, self.T)
            W, V = em_algo.repeat_process()
        return V
