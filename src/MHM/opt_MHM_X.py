import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.append("/Users/shukitakeuchi/irt_pro/src")

from MHM.optimize_x import Opt_x
from util.log import LoggerUtil


class Opt_MHM_X:
    def __init__(self, U, Y, T):
        # 初期設定
        self.U = U
        self.init_Y = Y
        self.T = T
        self.I, self.J = np.shape(self.U)
        self.logger = LoggerUtil.get_logger(__name__)

    def Parallel_step1(self, j):
        opt_x = Opt_x(self.U, self.init_Y, self.T)
        opt_x.modeling(j=j)
        x_opt, obj = opt_x.solve()
        return x_opt, obj

    def opt(self):
        self.logger.info("MHM(X) start")
        # 並列化
        with LoggerUtil.tqdm_joblib(self.J):
            out = Parallel(n_jobs=-1)(
                delayed(Opt_MHM_X.Parallel_step1)(self, j) for j in range(self.J)
            )
        X_opt = np.concatenate([[sample[0]] for sample in out], axis=0)
        obj = np.concatenate([[sample[1]] for sample in out], axis=0)
        self.logger.info("MHM(X) finish")
        return X_opt
