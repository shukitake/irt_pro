import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class data_visualization:
    @classmethod
    def MHM_icc_show(cls, X, J, T):
        x = np.arange(1, 11)
        for j in range(5):
            y = X[j, :]
            plt.plot(x, y, label=j + 1)

        plt.title("Monotone Homogenity model ICC")
        plt.xlabel("latent abilities")
        plt.ylabel("probarility of correct answer")
        plt.legend()
        plt.show()

    @classmethod
    def cluster_icc(cls, W, Z, V, J, N, T):
        x = np.arange(1, 11)
        for n in range(N):
            for j in range(J):
                if V[j][n] == 1:
                    y = [sum(W[k, t] * Z[j, k] for k in range(J)) for t in range(T)]
                    plt.plot(x, y, label=f"{j}th item")
                    plt.legend()
            plt.title(f"cluster_id={n} ICC")
            plt.xlabel("latent abilities")
            plt.ylabel("probarility of correct answer")
            plt.show()

    @classmethod
    def cl_icc_show(cls, W, V, J, N, T):
        x = np.arange(1, 11)
        for n in range(N):
            y = [W[n, t] for t in range(T)]
            plt.plot(x, y, label=n + 1)
        plt.title("clustering ICC")
        plt.xlabel("latent abilities")
        plt.ylabel("probarility of correct answer")
        plt.legend()
        plt.show()

    @classmethod
    def DMM_icc_show(cls, W, Z, J, T):
        x = np.arange(1, 11)
        for j in range(J):
            y = [sum(Z[j, k] * W[k, t] for k in range(J)) for t in range(T)]
            plt.plot(x, y, label=j + 1)

        plt.title("Double monotonicity model ICC")
        plt.xlabel("latent abilities")
        plt.ylabel("probarility of correct answer")
        plt.legend()
        plt.show()
