import sys
import numpy as np

sys.path.append("/Users/shukitakeuchi/irt_pro/src")

from DMM.optimize_W import Opt_W
from util.log import LoggerUtil
from joblib import Parallel, delayed
from tqdm import tqdm


class DMM_EM_Algo:
    def __init__(self, U, init_Y, Z, V, N, T):
        self.U = U
        self.init_Y = init_Y
        self.Z = Z
        self.V = V
        self.N = N
        self.T = T
        self.I, self.J = np.shape(self.U)
        self.logger = LoggerUtil.get_logger(__name__)
        return

    @classmethod
    def con_prob(cls, W_kt, Z_jk, U_ij):
        return np.power(np.power(W_kt, U_ij) * np.power(1 - W_kt, 1 - U_ij), Z_jk)

    def convert_Y_calss(self, Y):
        index = np.argmax(Y, axis=1)
        Y = np.zeros((self.I, self.T), dtype=int)
        for i in range(len(index)):
            Y[i, index[i]] = 1
        return Y

    def EStep(self, pi, W, Z):
        # self.logger.info("EStep start")
        f = np.array(
            [
                [
                    np.prod(
                        [
                            DMM_EM_Algo.con_prob(W[k, t], Z[j, k], self.U[i, j])
                            for j in range(self.J)
                            for k in range(self.J)
                        ]
                    )
                    for t in range(self.T)
                ]
                for i in range(self.I)
            ]
        )
        f1 = pi * f
        f2 = np.sum(f1, 1).reshape(-1, 1)
        Y = f1 / f2
        Y_opt = DMM_EM_Algo.convert_Y_calss(self, Y)
        # self.logger.info("EStep finish")
        # self.logger.info(f"Y:{Y_opt}")
        return Y, Y_opt

    def MStep(self, Y, Z):
        # self.logger.info("MStep start")
        # piの更新
        pi = np.sum(Y, axis=0) / self.I

        # Wの更新
        opt_W = Opt_W(self.U, Y, Z, self.V, self.N, self.T)
        opt_W.modeling()
        W_opt, obj = opt_W.solve()
        W_opt = np.reshape(W_opt, [self.J, self.T])
        self.logger.info(f"W optimized ->{W_opt}")
        # self.logger.info("MStep finish")
        # self.logger.info(f"objective:{obj}")
        return pi, W_opt

    def repeat_process(self, Z):
        # emstep
        self.logger.info("emstep start")
        # 初期ステップ -> MStep
        i = 1
        # Yを初期化
        Y_opt = self.init_Y
        # self.logger.info("first step")
        pi, W = DMM_EM_Algo.MStep(self, Y_opt, Z)
        est_Y = np.empty((self.I, self.T))
        while np.any(est_Y != Y_opt):
            est_Y = Y_opt
            # 繰り返し回数
            i += 1
            self.logger.info(f"{i}th step")
            # EStep
            Y, Y_opt = DMM_EM_Algo.EStep(self, pi, W, Z)
            # MStep
            pi, W = DMM_EM_Algo.MStep(self, Y_opt, Z)
            # 収束しない時、50回で終了させる
            if i == 3:
                return W, Y_opt
        self.logger.info("emstep finish")
        return W, Y_opt
