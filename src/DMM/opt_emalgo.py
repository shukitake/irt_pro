import sys
import numpy as np

sys.path.append("/Users/shukitakeuchi/irt_pro/src")

from util.log import LoggerUtil
from joblib import Parallel, delayed
from tqdm import tqdm
from DMM.optimize_Z import Opt_Z
from DMM.optimize_W import Opt_W


class DMM_EM_Algo:
    def __init__(self, U, init_Y, Z, V, N, T):
        self.U = U
        self.init_Y = init_Y
        self.I, self.J = np.shape(self.U)
        self.Z = Z
        self.T = T
        self.V = V
        self.N = N
        self.T = T
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

    def cl_list(self, n):
        cluster_list = []
        item_list = []
        for j in range(self.J):
            if self.V[j, n] == 1:
                k = np.argmax(self.Z[j, :])
                cluster_list.append(k)
                item_list.append(j)
        return cluster_list, item_list

    def EStep(self, pi, W):
        f = np.array(
            [
                [
                    np.prod(
                        [
                            DMM_EM_Algo.con_prob(W[k, t], self.Z[j, k], self.U[i, j])
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

    def parallel(self, n):
        cluster_list, item_list = DMM_EM_Algo.cl_list(self, n)
        num_item = len(cluster_list)
        tmp_Z = np.zeros((num_item, num_item))
        tmp_U = self.U[:, item_list]
        dif_list = np.argsort(cluster_list)
        for j in range(num_item):
            tmp_Z[dif_list[j], j] = 1
        opt_W = Opt_W(tmp_U, self.init_Y, tmp_Z, self.T)
        opt_W.modeling()
        W_opt, obj = opt_W.solve()
        W_opt = np.reshape(W_opt, [num_item, self.T])
        return W_opt, obj, dif_list, cluster_list

    def MStep(self, Y):
        # piの更新
        pi = np.sum(Y, axis=0) / self.I

        # Wの更新
        """tmp_W = np.zeros((self.J, self.T))
        for n in range(self.N):
            cluster_list, item_list = DMM_EM_Algo.cl_list(self, n)
            num_item = len(cluster_list)
            tmp_Z = np.zeros((num_item, num_item))
            tmp_U = self.U[:, item_list]
            dif_list = np.argsort(cluster_list)
            for j in range(num_item):
                tmp_Z[dif_list[j], j] = 1
            opt_W = Opt_W(tmp_U, self.init_Y, tmp_Z, self.T)
            opt_W.modeling()
            W_opt, obj = opt_W.solve()
            W_opt = np.reshape(W_opt, [num_item, self.T])
            m = 0
            for k in dif_list:
                tmp_W[cluster_list[k], :] = W_opt[m, :]
                m += 1"""

        tmp_W = np.zeros((self.J, self.T))
        with LoggerUtil.tqdm_joblib(self.N):
            out = Parallel(n_jobs=-1, verbose=0)(
                delayed(DMM_EM_Algo.parallel)(self, n) for n in range(self.N)
            )
        for m in range(self.N):
            W_opt = out[m][0]
            dif_list = out[m][2]
            cluster_list = out[m][3]
            j = 0
            for k in dif_list:
                tmp_W[cluster_list[k]] = W_opt[j]
                j += 1
        return pi, tmp_W

    def repeat_process(self):
        # 初期ステップ -> MStep
        i = 1
        # Yを初期化
        Y_opt = self.init_Y
        self.logger.info("first step")
        pi, W = DMM_EM_Algo.MStep(self, Y_opt)
        est_Y = np.empty((self.I, self.T))
        while np.any(est_Y != Y_opt):
            est_Y = Y_opt
            # 繰り返し回数
            i += 1
            self.logger.info(f"{i}th step")
            # EStep
            self.logger.info("Estep")
            Y, Y_opt = DMM_EM_Algo.EStep(self, pi, W)
            # MStep
            self.logger.info("Mstep")
            pi, W = DMM_EM_Algo.MStep(self, Y)
            # 収束しない時、30回で終了させる
            if i == 30:
                return W, Y_opt
        return W, Y_opt
