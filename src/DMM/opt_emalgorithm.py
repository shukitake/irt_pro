import sys
import numpy as np

sys.path.append("/Users/shukitakeuchi/irt_pro/src")

from DMM.optimize_W import Opt_W
from util.log import LoggerUtil


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
        f1 = np.multiply(pi, f)
        f2 = np.sum(f1, 1).reshape(-1, 1)
        Y = np.divide(f1, f2)
        Y_opt = DMM_EM_Algo.convert_Y_calss(self, Y)
        return Y, Y_opt

    def MStep(self, Y, Z):
        # piの更新
        pi = np.sum(Y, axis=0) / self.I

        # Wの更新
        opt_W = Opt_W(self.U, Y, Z, self.V, self.N, self.T)
        opt_W.modeling()
        W_opt, obj = opt_W.solve()
        W_opt = np.reshape(W_opt, [self.J, self.T])
        return pi, W_opt

    def repeat_process(self, Z):
        # emstep
        # 初期ステップ -> MStep
        i = 1
        # Yを初期化
        Y_opt = self.init_Y
        self.logger.info("first step")
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
            pi, W = DMM_EM_Algo.MStep(self, Y, Z)
            # 収束しない時、20回で終了させる
            if i == 20:
                return W, Y_opt
        self.logger.info("emstep finish")
        return W, Y_opt
