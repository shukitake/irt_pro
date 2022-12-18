import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *


class Opt_y:
    def __init__(self, U, X, T) -> None:
        # モデル
        self.solver = "scip"
        # 初期設定
        self.U = U
        self.X = X
        self.I, self.J = np.shape(self.U)
        self.T = T

    def modeling(self, i):
        # 非線形最適化モデル作成（minimize)
        self.model = pyo.ConcreteModel("Mixed Integer Linear Optimization")
        # 変数のセット
        self.model.J = pyo.Set(initialize=range(1, self.J + 1))
        self.model.T = pyo.Set(initialize=range(1, self.T + 1))
        # 決定変数
        self.model.y_i = pyo.Var(self.model.T, domain=pyo.Binary)
        # 制約
        self.model.const = pyo.ConstraintList()
        # 制約式
        # 制約1
        lhs = np.sum([self.model.y_i[t] for t in self.model.T])
        self.model.const.add(lhs == 1)
        # 目的関数
        expr = sum(
            (
                self.model.y_i[t]
                * (
                    (self.U[i, j - 1] * log(self.X[j - 1, t - 1]))
                    + ((1 - self.U[i, j - 1]) * log(1 - self.X[j - 1, t - 1]))
                )
            )
            for j in self.model.J
            for t in self.model.T
        )

        self.model.Obj = pyo.Objective(expr=expr, sense=pyo.maximize)
        return

    def solve(self):
        opt = pyo.SolverFactory(self.solver)
        # opt.options["halt_on_ampl_error"] = "yes"
        opt.solve(self.model, tee=False)
        return pyo.value(self.model.y_i[:]), self.model.Obj()
