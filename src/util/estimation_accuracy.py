import numpy as np


class est_accuracy:
    @classmethod
    def show_class(cls, Y):
        T = 1 + np.argmax(Y, 1)
        return T

    @classmethod
    def rsme_class(cls, T_true, T_est):
        I = len(T_true)
        rsme = np.sqrt(
            np.sum([np.square(T_true[i] - T_est[i]) for i in range(len(T_true))]) / I
        )
        return rsme

    @classmethod
    def show_icc(cls):
        return

    def rsme_icc(cls, true_x, est_x):
        return
