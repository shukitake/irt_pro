import sys
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append("/Users/shukitakeuchi/irt_pro/src")
from util.log import LoggerUtil
from DMM.optimize_W import Opt_W
from DMM.optimize_y import Opt_y


class Opt_DMM_Algo:
    def __init__(self, U, init_Y, init_Z, V, N, T):
        self.U = U
        self.init_Y = init_Y
        self.init_Z = init_Z
        self.V = V
        self.I, self.J = np.shape(self.U)
        self.N = N
        self.T = T
        self.logger = LoggerUtil.get_logger(__name__)
        return

    def cl_list(self, n):
        cluster_list = []
        for j in range(self.J):
            if self.V[j, n] == 1:
                k = np.argmax(self.init_Z[j, :])
                cluster_list.append(k + 1)
        cluster_list.sort()
        return cluster_list

    def cl_list_Z(self):
        return [Opt_DMM_Algo.cl_list(n) for n in self.N]

    def Parallel_step2(self, i, W_opt, Z_opt):
        # モデルの作成
        opt_y = Opt_y(self.U, W_opt, Z_opt, self.T)
        opt_y.modeling(i=i)
        # モデルの最適化
        y_opt, obj = opt_y.solve()
        return y_opt, obj

    def process(self):
        # step1
        """DMM(Y,Z)についてWの最適解を求める"""
        self.logger.info("DMM start")

        opt_W = Opt_W(self.U, self.init_Y, self.init_Z, self.V, self.N, self.T)
        opt_W.modeling()
        W_opt, obj = opt_W.solve()
        W_opt = np.reshape(W_opt, [self.J, self.T])
        self.logger.info("DMM(Y,Z) optimized W")
        # self.logger.info(f"W optimized ->{W_opt}")

        # step2
        """DMM(W,Z)についてYの最適解を求める"""
        self.logger.info("DMM(W,Z) optimize Y")
        # 並列化
        with LoggerUtil.tqdm_joblib(self.I):
            out = Parallel(n_jobs=-1)(
                delayed(Opt_DMM_Algo.Parallel_step2)(self, i, W_opt, self.init_Z)
                for i in range(self.I)
            )
        # self.logger.info(f"{out}")
        Y_opt = np.concatenate([[sample[0]] for sample in out], axis=0)
        obj = np.concatenate([[sample[1]] for sample in out], axis=0)
        self.logger.info("DMM(W,Z) optimized Y")
        # self.logger.info(f"Y optimized ->{Y_opt}")
        return W_opt, Y_opt
