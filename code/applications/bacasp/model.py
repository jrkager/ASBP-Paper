import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import numpy as np

from ..optimization_model import OptimizationModel
from helper import getValueDict

from .instance import Instance

from itertools import product

from .. import SecondStageModelType


def getFirstStageObjective(inst, first_stage_solution):
    return 0

def getSecondStageObjective(inst:Instance, variables):
    c, ii = variables
    return gp.quicksum(c[ii][k] - inst.vessel_arrival_time[ii][k] for k in inst.V)

def createSecondStage(instance, ii, model, variables):
    inst = instance
    x, y, b, pi, sigma, t, z, alpha, beta, gamma, c = variables
    V, T, B, G = inst.V, inst.T, inst.B, inst.G
    slack = inst.slack
    J, M = inst.J, inst.M
    NC = inst.NC

    for k in V:
        for l in V:
            if k != l:
                model.addConstr(t[ii][l] >= c[ii][k] + slack + (M+slack) * (x[k, l] - 1), name=f"E4_{k}_{l}_{ii}")

    for k in V:
        model.addConstr(t[ii][k] >= inst.vessel_arrival_time[ii][k], name=f"E6_{k}_{ii}")
        model.addConstr(
            gp.quicksum(z[ii][g, k, j] * inst.crane_processing_rate[g] for j in T for g in G) >= inst.vessel_load[k],
            name=f"G4_{k}_{ii}")

    for j in T:
        for g in G:
            model.addConstr(gp.quicksum(z[ii][g, k, j] for k in V) <= 1, name=f"G1_{j}_{g}_{ii}")
            for k in V:
                model.addConstr(t[ii][k] <= j * z[ii][g, k, j] + M * (1 - z[ii][g, k, j]), name=f"G2_{k}_{j}_{g}_{ii}")
                model.addConstr(c[ii][k] >= (j + 1) * z[ii][g, k, j], name=f"G3_{k}_{j}_{g}_{ii}")
                model.addConstr(b[k] + inst.vessel_length[k] <= min(inst.crane_to[g], J) * z[ii][g, k, j] + (1 - z[ii][g, k, j]) * (J+1), name=f"G5_{k}_{j}_{g}_{ii}")
                model.addConstr(b[k] >= max(inst.crane_from[g], 0) * z[ii][g, k, j], name=f"G6_{k}_{j}_{g}_{ii}")

    for j in T:
        for k in V:
            for l in V:
                if k != l:
                    # takes effect, when l berths unter k
                    model.addConstrs(
                        (z[ii][g, k, j] + z[ii][g_prime, l, j] <= 2 - y[k, l] for g in G for g_prime in G if g_prime < g),
                        name=f"G7_{k}_{l}_{j}_{ii}")

    for k in V:
        for j in T:
            model.addConstr(gp.quicksum(z[ii][g, k, j] for g in G) <= NC[k], name=f"G8_{k}_{j}_{ii}")

            for g in G:
                sumind = range(max(inst.crane_from[g], 0), min(inst.crane_to[g] - inst.vessel_length[k], J) + 1)
                model.addConstr(z[ii][g, k, j] <= gp.quicksum(pi[k, n] for n in sumind), name=f"G11_{j}_{k}_{ii}")

    for k in V:
        for j in T:
            model.addConstr(c[ii][k] >= (j + 1) * beta[ii][k, j], name=f"T4_{j}_{k}_{ii}")

    for k in V:
        for j in T:
            for g in G:
                model.addConstr(z[ii][g, k, j] <= beta[ii][k, j], name=f"T5_{j}_{g}_{k}_{ii}")

    for k in V:
        model.addConstr(t[ii][k] == gp.quicksum(j * alpha[ii][k, j] for j in T), name=f"T3_{k}_{ii}")
        model.addConstr(gp.quicksum(alpha[ii][k, j] for j in T) == 1, name=f"T6_{k}_{ii}")
        for j in T[1:]:
            model.addConstr(alpha[ii][k, j] >= beta[ii][k, j] - beta[ii][k, j - 1], name=f"T7_{k}_{j}_{ii}")
            model.addConstr(alpha[ii][k, j] <= 1 - beta[ii][k, j - 1], name=f"T10_{k}_{j}_{ii}")

        model.addConstr(alpha[ii][k, 1] >= beta[ii][k, 1], name=f"T8_{k}_{ii}")

        for j in T:
            model.addConstr(alpha[ii][k, j] <= beta[ii][k, j], name=f"T9_{k}_{j}_{ii}")

    for k, l, j, i in product(V, V, T, T):
        if k != l and i >= j - slack:
            model.addConstr(x[k, l] + beta[ii][k, i] + alpha[ii][l, j] <= 2, name=f"T11_{k}_{l}_{j}_{i}_{ii}")

    for k, j in product(V, T):
        if j < M:
            model.addConstr(gamma[ii][k, j] >= beta[ii][k, j] - beta[ii][k, j + 1], name=f"A1_{k}_{j}_{ii}")
            model.addConstr(gamma[ii][k, j] <= 1 - beta[ii][k, j + 1], name=f"A4_{k}_{j}_{ii}")
        model.addConstr(gamma[ii][k, j] <= beta[ii][k, j], name=f"A3_{k}_{j}_{ii}")
    for k in V:
        model.addConstr(gamma[ii][k, M] >= beta[ii][k, M], name=f"A2_{k}_{ii}")
        model.addConstr(quicksum(gamma[ii][k,j] for j in T) == 1, name=f"A5_{k}_{ii}")

    return getSecondStageObjective(instance, (c, ii))


def initialize_first_subset(instance: Instance):
    A = instance.vessel_arrival_time
    min_slack = min(instance.scenarios, key=lambda ii: sum(min(abs(A[ii][k] - A[ii][k-1]),
                                                               abs(A[ii][k+1] - A[ii][k]))
                                                           for k in range(1, instance.N-1)))
    return [min_slack]


class MasterModel(OptimizationModel):

    def __init__(self, instance: Instance, scenarios, *argc, **argv):
        super().__init__(*argc, **argv)

        inst = instance
        self._instance = instance
        self._scenarios = scenarios

        J = inst.J
        M = inst.M
        V = inst.V # range(inst.N): vessels. index k
        T = inst.T # range(1, M+1) # time periods 1 to M
        B = inst.B # range(J+1) # berthing position 0 to J. index n
        G = inst.G # range(7) # crane 0 to 7. index g

        # H_k is vessel_length in instance
        # Q_k is vessel_load
        # A_w_k is vessel_arrival_time in scenario w

        # first stage variables
        x = self.addVars(V, V, vtype=GRB.BINARY, name="x")
        y = self.addVars(V, V, vtype=GRB.BINARY, name="y")
        b = self.addVars(V, lb=0, vtype=GRB.INTEGER, name="b")
        pi = self.addVars(V, B, vtype=GRB.BINARY, name="pi")
        sigma = self.addVars(V, B, vtype=GRB.BINARY, name="sigma")

        # auxiliary
        if len(scenarios) == 0:
            theta = self.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
        else:
            theta = self.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name="theta")

        # first stage constraints
        for k, l in product(V, V):
            if k < l:
                self.addConstr(x[l, k] + x[k, l] + y[l, k] + y[k, l] == 1, name=f"E1_{k}_{l}")
            if k != l:
                self.addConstr(b[k] >= b[l] + inst.vessel_length[l] + (J+1) * (y[k, l] - 1), name=f"E5_{k}_{l}")
        for k in V:
            self.addConstr(b[k] <= J - inst.vessel_length[k] + 1, name="E7_{k}")
            self.addConstr(b[k] == gp.quicksum(n * pi[k, n] for n in B), name=f"C1_{k}")
            self.addConstr(gp.quicksum(sigma[k, n] for n in B) == inst.vessel_length[k], name=f"C2_{k}")
            self.addConstr(gp.quicksum(pi[k, n] for n in B) == 1, name=f"C3_{k}")
            self.addConstr(pi[k, 0] >= sigma[k, 0], name=f"C5_{k}")
        for k in V:
            for n in B:
                self.addConstr(pi[k, n] <= sigma[k, n], name=f"C6_{k}_{n}")
                if n > 0:
                    self.addConstr(pi[k, n] >= sigma[k, n] - sigma[k, n - 1], name=f"C4_{k}_{n}")
                    self.addConstr(pi[k, n] <= 1 - sigma[k, n-1], name=f"C7_{k}_{n}")
        for k, l in product(V, V):
            if k != l:
                for n in B:
                    sumind = range(max(n - inst.vessel_length[l] + 1, 0), J+1)
                    self.addConstr(y[k, l] + gp.quicksum(pi[l, m] for m in sumind) + pi[k, n] <= 2, name=f"C8_{k}_{l}_{n}")

        for k, l in product(V, V):
            self.addConstr(b[k] <= b[l] + inst.vessel_length[l] - 1 + y[k,l] * J, name=f"M1_{k}_{l}")

        # second stage variables
        t, z, alpha, beta, gamma, c = {}, {}, {}, {}, {}, {}
        for ii in scenarios:
            t[ii] = self.addVars(V, lb=0, vtype=GRB.INTEGER, name=f"t_{ii}")
            z[ii] = self.addVars(G, V, T, vtype=GRB.BINARY, name=f"z_{ii}")
            alpha[ii] = self.addVars(V, T, vtype=GRB.BINARY, name=f"alpha_{ii}")
            beta[ii] = self.addVars(V, T, vtype=GRB.BINARY, name=f"beta_{ii}")
            gamma[ii] = self.addVars(V, T, vtype=GRB.BINARY, name=f"gamma_{ii}")
            c[ii] = self.addVars(V, lb=0, vtype=GRB.INTEGER, name=f"c_{ii}")

        # second stage constraints
        for ii in scenarios:
            ss_obj = createSecondStage(inst, ii, self, (x, y, b, pi, sigma, t, z, alpha, beta, gamma, c))
            self.addConstr(theta >= ss_obj, name=f"Aux_{ii}")

        self.setObjective(theta, GRB.MINIMIZE)

    def get_first_stage_solution(self):
        x0 = getValueDict(self._vars["x"], roundBinaries=True)
        y0 = getValueDict(self._vars["y"], roundBinaries=True)
        b0 = getValueDict(self._vars["b"], roundInteger=True)
        pi0 = getValueDict(self._vars["pi"], roundBinaries=True)
        sigma0 = getValueDict(self._vars["sigma"], roundBinaries=True)
        return x0, y0, b0, pi0, sigma0

    def get_first_stage_objective(self):
        return 0

    def get_second_stage_bound(self):
        return self._vars['theta'].X

    def get_second_stage_solution_for_scenario(self, ii):
        if ii not in self._scenarios:
            return None
        tii = getValueDict(self._vars[f"t_{ii}"], roundInteger=True)
        zii = getValueDict(self._vars[f"z_{ii}"], roundBinaries=True)
        alphaii = getValueDict(self._vars[f"alpha_{ii}"], roundBinaries=True)
        betaii = getValueDict(self._vars[f"beta_{ii}"], roundBinaries=True)
        gammaii = getValueDict(self._vars[f"gamma_{ii}"], roundBinaries=True)
        cii = getValueDict(self._vars[f"c_{ii}"], roundInteger=True)
        return tii, zii, alphaii, betaii, gammaii, cii


class SecondStageModel(SecondStageModelType):

    def __init__(self, instance: Instance, k, first_stage_solution, warmstart = None, *argc, **argv):
        super().__init__(*argc, **argv)

        ii = k
        self._k = k
        self._ireason = None
        self._first_stage_solution = first_stage_solution

        inst = instance
        x, y, b, pi, sigma = first_stage_solution
        V, T, B, G = inst.V, inst.T, inst.B, inst.G

        t, z, alpha, beta, gamma, c = {}, {}, {}, {}, {}, {}
        t[ii] = self.addVars(V, lb=0, vtype=GRB.INTEGER, name=f"t_{ii}")
        z[ii] = self.addVars(G, V, T, vtype=GRB.BINARY, name=f"z_{ii}")
        alpha[ii] = self.addVars(V, T, vtype=GRB.BINARY, name=f"alpha_{ii}")
        beta[ii] = self.addVars(V, T, vtype=GRB.BINARY, name=f"beta_{ii}")
        gamma[ii] = self.addVars(V, T, vtype=GRB.BINARY, name=f"gamma_{ii}")
        c[ii] = self.addVars(V, lb=0, vtype=GRB.INTEGER, name=f"c_{ii}")

        obj = createSecondStage(instance, ii, self, (x, y, b, pi, sigma, t, z, alpha, beta, gamma, c))
        self.setObjective(obj, GRB.MINIMIZE)

        if warmstart:
            tii, zii, alphaii, betaii, gammaii, cii = warmstart
            for k in V:
                t[ii][k].Start = tii[k]
                c[ii][k].Start = cii[k]
            for g, k, j in product(G, V, T):
                z[ii][g,k,j].Start = zii[g,k,j]
            for k, j in product(V, T):
                alpha[ii][k, j].Start = alphaii[k, j]
                beta[ii][k, j].Start = betaii[k, j]
                gamma[ii][k, j].Start = gammaii[k, j]

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
