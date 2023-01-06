import sys
import numpy as np

sys.path.append("/Users/shukitakeuchi/irt_pro/src")

from util.log import LoggerUtil
from joblib import Parallel, delayed
from tqdm import tqdm
from MHM.optimize_x import Opt_x
from MHM.optimize_y import Opt_y


class Heu_MHM_Algo:
    def __init__(self, U, Y, T):
        self.U = U
        self.init_Y = Y
        self.I, self.J = np.shape(self.U)
        self.T = T
        self.logger = LoggerUtil.get_logger(__name__)
        return

    def Parallel_step1(self, j, Y):  # モデルの作成
        opt_x = Opt_x(self.U, Y, self.T)
        opt_x.modeling(j=j)
        # モデルの最適化
        x_opt, obj = opt_x.solve()
        return x_opt, obj

    def Parallel_step2(self, i, X):
        # モデルの作成
        opt_y = Opt_y(self.U, X, self.T)
        opt_y.modeling(i=i)
        # モデルの最適化
        y_opt, obj = opt_y.solve()
        return y_opt, obj

    def process(self, Y):
        # step1
        self.logger.info("MHM step1")
        """X_opt = np.empty((self.J, self.T))
        for j in range(self.J):
            # self.logger.info(f"{j+1}th item optimized")
            opt_x = Opt_x(self.U, Y, self.T)
            opt_x.modeling(j=j)
            x_opt, obj = opt_x.solve()
            # self.logger.info(f"x_{j+1}->{x_opt}")
            X_opt[j, :] = x_opt"""
        # 並列化
        with LoggerUtil.tqdm_joblib(self.J):
            out = Parallel(n_jobs=-1, verbose=0)(
                delayed(Heu_MHM_Algo.Parallel_step1)(self, j, Y) for j in range(self.J)
            )
        # self.logger.info(f"{out}")
        X_opt = np.concatenate([[sample[0]] for sample in out], axis=0)
        obj = np.concatenate([[sample[1]] for sample in out], axis=0)
        # self.logger.info(f"X optimized ->{X_opt}")
        # step2
        self.logger.info("step2")
        """Y_opt = np.empty((self.I, self.T))
        for i in range(self.I):
            opt_y = Opt_y(self.U, X_opt, self.T)
            opt_y.modeling(i=i)
            y_opt, obj = opt_y.solve()
            # self.logger.info(f"y_{i}->{y_opt}")
            Y_opt[i, :] = y_opt"""
        # 並列化
        with LoggerUtil.tqdm_joblib(self.I):
            out = Parallel(n_jobs=-1, verbose=0)(
                delayed(Heu_MHM_Algo.Parallel_step2)(self, i, X_opt)
                for i in range(self.I)
            )
        # self.logger.info(f"{out}")
        Y_opt = np.concatenate([[sample[0]] for sample in out], axis=0)
        obj = np.concatenate([[sample[1]] for sample in out], axis=0)
        # self.logger.info(f"Y optimized ->{Y_opt}")
        return X_opt, Y_opt

    def repeat_process(self, Y):
        # step0 init_Y
        self.logger.info("initialize")
        best_Y = self.init_Y
        X_opt, Y_opt = Heu_MHM_Algo.process(self, best_Y)
        i = 1
        while np.any(best_Y != Y_opt):
            # 繰り返し回数
            i += 1
            self.logger.info(f"{i}th step")
            X_opt, Y_opt = Heu_MHM_Algo.process(self, best_Y)
            best_Y = Y_opt
            # 収束しない時、30回で終了させる
            if i == 30:
                return X_opt, Y_opt
        return X_opt, Y_opt
