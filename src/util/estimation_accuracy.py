import numpy as np


class est_accuracy:
    @classmethod
    def show_class(cls, Y):
        T = 1 + np.argmax(Y, 1)
        return T

    @classmethod
    def rmse_class(cls, T_true, T_est):
        I = len(T_true)
        rmse = np.sqrt(
            np.sum([np.square(T_true[i] - T_est[i]) for i in range(len(T_true))]) / I
        )
        return rmse

    @classmethod
    def rmse_2plicc(cls, icc_true, X):
        J, T = np.shape(X)
        icc_est = X
        rmse = np.sqrt(
            np.sum(
                np.square(icc_true[j, t] - icc_est[j, t])
                for j in range(J)
                for t in range(T)
            )
            / (J * T)
        )
        return rmse

    @classmethod
    def rmse_icc(cls, icc_true, Z, W):
        J, T = np.shape(W)
        icc_est = np.dot(Z, W)
        rmse = np.sqrt(
            np.sum(
                np.square(icc_true[j, t] - icc_est[j, t])
                for j in range(J)
                for t in range(T)
            )
            / (J * T)
        )
        return rmse
