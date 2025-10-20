import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import numpy as np
import random

from applications.optimization_model import OptimizationModel
from helper import roundBinaryValue, getValueDict

from .instance import Instance
from .. import SecondStageModelType

import copy
from itertools import product


def getFirstStageObjective(inst, first_stage_solution):
    r0, y0 = first_stage_solution
    return  sum(inst.f_i[i].FIRST * r0[i] for i in inst.I) + \
            sum(inst.d_i[i].FIRST * y0[i] for i in inst.I)
def getSecondStageObjective(inst, variables):
    I, J, V = inst.I, inst.J, inst.V
    r0, y0, rk, yk, xk, tk = variables
    return  sum(inst.f_i[i].SECOND * (rk[i] - r0[i]) for i in I) + \
            sum(inst.d_i[i].SECOND * (yk[i] - y0[i]) for i in I) + \
            sum(inst.c_ij[i, j] * xk[i, j] for i in V for j in V) + \
            sum(inst.gamma_ij[i, j] * tk[i, j] for i in V for j in V) + \
            sum(inst.alpha_ij[i, j] * xk[i, j] for i in V for j in V) + \
            sum(inst.F * xk[i, j] for i in I for j in J)

def createSecondStage(instance, k, model, variables):
    """
        The model needs to have the variables x, r, y, s, t for all I, J and the scenario k (and 0 for r and y)
    """
    inst = instance
    x, r, y, s, t = variables
    I, J, V = inst.I, inst.J, inst.V
    J_bar = [j for j in inst.J if instance.beta_k_j[k][j] > 0]
    J_bar_complement = [j for j in inst.J if j not in J_bar]

    model.addConstrs(quicksum(x[k][i, j] for i in V if i != j) == 1 for j in J_bar)
    model.addConstrs(quicksum(x[k][j, l] for l in V if l != j) == 1 for j in J_bar)
    model.addConstrs(quicksum(x[k][i, j] for i in V if i != j) == 0 for j in J_bar_complement)
    model.addConstrs(quicksum(x[k][j, l] for l in V if l != j) == 0 for j in J_bar_complement)

    model.addConstrs(quicksum(x[k][i, j] - x[k][j, i] for j in J) == 0 for i in I)
    model._demandConstr = model.addConstrs(quicksum(t[k][i, j] for i in V if i != j)
                                           - quicksum(t[k][j, l] for l in V if l != j) == inst.beta_k_j[k][j] for j in J)
    model.addConstrs(t[k][i, j] <= inst.Q * x[k][i, j] for i in V for j in V)
    model.addConstrs(quicksum(t[k][i, j] for j in J) <= y[k][i] for i in I)
    model.addConstrs(y[0][i] <= y[k][i] for i in I)
    model.addConstrs(y[k][i] <= inst.C_i[i] * r[k][i] for i in I)
    model.addConstrs(r[0][i] <= r[k][i] for i in I)

    model.addConstrs(s[k][i, j] - s[k][i, u] <= 1 - x[k][j, u] - x[k][u, j]
                     for i in I for j in J for u in J if j != u)
    model.addConstrs(s[k][i, j] >= x[k][i, j] for i in I for j in J)
    model.addConstrs(s[k][i, j] >= x[k][j, i] for i in I for j in J)
    model.addConstrs(quicksum(s[k][i, j] for i in I) == 1 for j in J)

    return getSecondStageObjective(inst, (r[0], y[0], r[k], y[k], x[k], t[k]))


def initSubsetEmpty(instance: Instance):
    return []

def initSubsetRandom(instance: Instance):
    return [random.choice(instance.scenarios)]

def initSubsetMaxDemand(instance: Instance):
    min_sc = min(instance.scenarios, key=lambda s: (sum(instance.beta_k_j[s][j] >= 0 for j in instance.J),
                                                    sum(instance.beta_k_j[s][j] for j in instance.J)))
    return [min_sc]

class MasterModel(OptimizationModel):

    def __init__(self, instance: Instance, scenarios, *argc,
                 **argv):

        super().__init__(*argc, **argv)

        if not all(i > 0 for i in scenarios):
            raise(Exception("scenarios must be numbered with integers greater or equal 1."))

        inst = copy.deepcopy(instance)
        I, J, V = inst.I, inst.J, inst.V
        self._instance = inst
        self._scenarios = scenarios

        x = {}
        for k in scenarios:
            x[k] = self.addVars(V, V, vtype=GRB.BINARY, name="x_{}".format(k))

        t = {}
        for k in scenarios:
            t[k] = self.addVars(V, V, name="t_{}".format(k))

        s = {}
        for k in scenarios:
            s[k] = self.addVars(I, J, vtype=GRB.BINARY, name="s_{}".format(k))

        r = {}
        r[0] = self.addVars(I, vtype=GRB.BINARY, name="r_0")
        for k in scenarios:
            r[k] = self.addVars(I, vtype=GRB.BINARY, name="r_{}".format(k))

        y = {}
        y[0] = self.addVars(I, name="y_0")
        for k in scenarios:
            y[k] = self.addVars(I, name="y_{}".format(k))

        # nonnegative variable by default
        z = self.addVar(vtype=GRB.CONTINUOUS, name="z")

        self.addConstrs(y[0][i] <= inst.C_i[i] * r[0][i] for i in I)

        for k in scenarios:
            ss_obj = createSecondStage(inst, k, self, (x, r, y, s, t))
            self.addConstr(z >= ss_obj)

        FS_obj = getFirstStageObjective(inst, (r[0], y[0]))

        self.setObjective( FS_obj + z )

    def get_first_stage_solution(self):
        r0 = {i : roundBinaryValue(self._vars["r_0"][i].X) for i in self._instance.I}
        y0 = {i : self._vars["y_0"][i].X for i in self._instance.I}
        return r0, y0

    def get_first_stage_objective(self):
        return getFirstStageObjective(self._instance, self.get_first_stage_solution())

    def get_second_stage_bound(self):
        return self._vars['z'].X

    def get_second_stage_solution_for_scenario(self, k):
        if k not in self._scenarios:
            return None
        xk = getValueDict(self._vars[f"x_{k}"], roundBinaries=True)
        rk = getValueDict(self._vars[f"r_{k}"], roundBinaries=True)
        yk = getValueDict(self._vars[f"y_{k}"])
        sk = getValueDict(self._vars[f"s_{k}"], roundBinaries=True)
        tk = getValueDict(self._vars[f"t_{k}"])
        return xk, rk, yk, sk, tk


class SecondStageModel(SecondStageModelType):

    def __init__(self, instance: Instance, k, first_stage_solution, warmstart = None, *argc, **argv):

        super().__init__(*argc, **argv)

        self._ireason  = None
        r_0, y_0 = first_stage_solution

        if not k >= 1:
            raise(Exception("scenarios must be numbered with integers greater or equal 1."))

        inst = instance
        I, J, V = inst.I, inst.J, inst.V

        self._instance = inst
        self._k = k
        self._r_0 = r_0
        self._y_0 = y_0

        x = {}
        x[k] = self.addVars(V, V, vtype=GRB.BINARY, name="x_{}".format(k))

        t = {}
        t[k] = self.addVars(V, V, name="t_{}".format(k))

        s = {}
        s[k] = self.addVars(I, J, vtype=GRB.BINARY, name="s_{}".format(k))

        r = {}
        r[0] = {i : float(r_0[i]) for i in I}
        r[k] = self.addVars(I, vtype=GRB.BINARY, name="r_{}".format(k))

        y = {}
        y[0] = {i : float(y_0[i]) for i in I} # cant use numpy types here, see https://support.gurobi.com/hc/en-us/articles/360039628832-Constraint-has-no-bool-value-are-you-trying-lb-expr-ub-
        y[k] = self.addVars(I, name="y_{}".format(k))

        self.setObjective(createSecondStage(inst, k, self, (x, r, y, s, t)))

        if warmstart:
            xk, rk, yk, sk, tk = warmstart
            for i, j in product(V, V):
                x[k][i,j].Start = xk[i,j]
                t[k][i, j].Start = tk[i, j]
            for i, j in product(I, J):
                s[k][i, j].Start = sk[i, j]
            for i in I:
                r[k][i].Start = rk[i]
                y[k][i].Start = yk[i]

    def get_second_stage_solution(self):
        k=self._k
        xk = getValueDict(self._vars[f"x_{k}"], roundBinaries=True)
        rk = getValueDict(self._vars[f"r_{k}"], roundBinaries=True)
        yk = getValueDict(self._vars[f"y_{k}"])
        sk = getValueDict(self._vars[f"s_{k}"], roundBinaries=True)
        tk = getValueDict(self._vars[f"t_{k}"])
        return xk, rk, yk, sk, tk

    @property
    def mipgap(self):
        try: return self.getAttr("MIPGap")
        except: return np.inf

    @property
    def objbound(self):
        try: return self.getAttr("objbound")
        except: return -np.inf

    @property
    def objval(self):
        try: return self.getAttr("objval")
        except: return np.inf
