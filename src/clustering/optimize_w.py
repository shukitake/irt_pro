import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *


class Opt_W:
    def __init__(self, U, Y, V, N, T) -> None:
        # モデル
        self.solver = "ipopt"
        # 初期設定
        self.U = U
        self.Y = Y
        self.V = V
        self.I, self.J = np.shape(self.U)
        self.N = N
        self.T = T

    def modeling(self):
        # 非線形最適化モデル作成（minimize)
        self.model = pyo.ConcreteModel("Maximize Non Convex Optimization")
        # 変数のセット
        self.model.I = pyo.Set(initialize=range(1, self.I + 1))
        self.model.J = pyo.Set(initialize=range(1, self.J + 1))
        self.model.T = pyo.Set(initialize=range(1, self.T + 1))
        self.model.T1 = pyo.Set(initialize=range(1, self.T))
        self.model.N = pyo.Set(initialize=range(1, self.N + 1))
        # 決定変数
        """0.01から0.99を取るような連続変数W"""
        self.model.W = pyo.Var(
            self.model.N, self.model.T, domain=pyo.Reals, bounds=(0.01, 0.99)
        )
        # 制約
        self.model.const = pyo.ConstraintList()
        # 制約式
        # 制約1
        """単調等質性の制約"""
        for n in self.model.N:
            for t in self.model.T1:
                lhs = self.model.W[n, t + 1] - self.model.W[n, t]
                self.model.const.add(lhs >= 0)
        # 目的関数
        expr = sum(
            (
                self.Y[i - 1, t - 1]
                * self.V[j - 1, n - 1]
                * (
                    (self.U[i - 1, j - 1] * log(self.model.W[n, t]))
                    + ((1 - self.U[i - 1, j - 1]) * log(1 - self.model.W[n, t]))
                )
            )
            for i in self.model.I
            for j in self.model.J
            for n in self.model.N
            for t in self.model.T
        )
        self.model.Obj = pyo.Objective(expr=expr, sense=pyo.maximize)
        return

    def solve(self):
        opt = pyo.SolverFactory(self.solver)
        # opt.options["halt_on_ampl_error"] = "yes"
        opt.solve(self.model, tee=False)
        return pyo.value(self.model.W[:, :]), self.model.Obj()
