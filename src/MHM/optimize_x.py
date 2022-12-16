import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import *


class Opt_x:
    def __init__(self, U, Y, T) -> None:
        # モデル
        self.solver = "ipopt"
        # 初期設定
        self.U = U
        self.Y = Y
        self.T = T
        self.I, self.J = np.shape(self.U)

    def modeling(self, j):
        # 非線形最適化モデル作成（maximize)
        self.model = pyo.ConcreteModel("Maximize Non Convex Optimization")
        # 変数のセット
        self.model.I = pyo.Set(initialize=range(1, self.I + 1))
        self.model.T = pyo.Set(initialize=range(1, self.T + 1))
        self.model.T1 = pyo.Set(initialize=range(1, self.T))
        # 決定変数
        self.model.x_j = pyo.Var(self.model.T, domain=pyo.Reals, bounds=(0.01, 0.99))
        # 制約
        self.model.const = pyo.ConstraintList()
        # 制約式
        # 制約1
        for t in self.model.T1:
            lhs = self.model.x_j[t + 1] - self.model.x_j[t]
            self.model.const.add(lhs >= 0)
        # 目的関数
        expr = sum(
            (
                self.Y[i - 1, t - 1]
                * (
                    (self.U[i - 1, j] * log(self.model.x_j[t]))
                    + ((1 - self.U[i - 1, j]) * log(1 - self.model.x_j[t]))
                )
            )
            for i in self.model.I
            for t in self.model.T
        )

        self.model.obj = pyo.Objective(expr=expr, sense=pyo.maximize)
        return

    def solve(self):
        opt = pyo.SolverFactory(self.solver)
        # opt.options["halt_on_ampl_error"] = "yes"
        opt.solve(self.model, tee=False)
        return pyo.value(self.model.x_j[:]), self.model.obj()
