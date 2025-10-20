from applications.lrp.instances import toro_instances
from .. import InstanceStrings

import numpy as np

import random

class PenalizedCost:
    def __init__(self, firstStageCost, secondStageCost):
        self.FIRST = firstStageCost
        self.SECOND = secondStageCost

    def __str__(self):
        return f"({self.FIRST:.2f}, {self.SECOND:.2f})"

def penalize(cost, penalty=1.5):
    if isinstance(cost, list):
        keys = range(len(cost))
    else:
        keys = cost.keys()
    return {i : PenalizedCost(cost[i], penalty * cost[i]) for i in keys}

class OffsetList(list):
    def __init__(self, offset=0, iterable=[]):
        super().__init__(iterable)
        self.offset = offset

    def __getitem__(self, key):
        return self[key + self.offset]

    def __setitem__(self, index, item):
        super().__setitem__(index - self.offset, item)

    def insert(self, index, item):
        super().insert(index - self.offset, item)

    def append(self, item):
        super().append(item)

class Instance:

    def __init__(self, I, J, beta_k_j, d_i, f_i, c_ij, alpha_ij, gamma_ij, F, C_i, Q, loc, name="", sample_number=None, scenarios=None):
        """

        :param I: set of warehouse indices (from 0 to m-1)
        :param J: set of customer indices (from m to m+n-1)
        :param d_i, f_i, ...: as dict or list of PenalizedCosts, indexing from I and then .FIRST or .SECOND
        :param beta_k_j: dict of dicts. demand of customer j (J) in scenario k (>= 1). scenario 1 usually defines the nominal demand
        :param loc: np array with size m+n x 2. indices I and J work, array might have unused rows.
        :param scenarios: list of used scenarios (was previously a param to the algorithm, now part of the instance)
        """
        self._I = list(I)
        self._J = list(J)
        self.V = self._I + self._J
        self.beta_k_j = beta_k_j
        self.d_i = d_i  # of type PenalizedCost
        self.f_i = f_i  # of type PenalizedCost
        if not all(isinstance(d_i[i], PenalizedCost) and isinstance(f_i[i], PenalizedCost) for i in I):
            raise TypeError("Not given as PenalizedCost")
        self.c_ij = c_ij
        self.alpha_ij = alpha_ij
        self.gamma_ij = gamma_ij
        self.F = F
        self.C_i = C_i
        self._max_C = max(C_i)
        self.Q = Q
        self.loc = loc
        self.name = name
        self.sample_number = sample_number
        self.scenarios = scenarios or list(beta_k_j.keys())

        m, n, K = len(I), len(J), len(self.scenarios)
        self.strings = InstanceStrings()
        self.updateStrings()

    def updateStrings(self):
        m, n, K = len(self._I), len(self._J), len(self.scenarios)
        self.strings.ALG_INTRO_TEXT = f"algorithm for m = {m}, n={n}, #scenarios={K}, sample = {self.sample_number}\n"
        self.strings.UNIQUE_IDENTIFIER = f"{m}-{n}-{K}-{self.sample_number}"

    @property
    def I(self):
        return self._I

    @I.setter
    def I(self, value):
        self._I = list(value)
        self.V = self._I + self._J
        self.updateStrings()

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, value):
        self._J = list(value)
        self.V = self._I + self._J
        self.updateStrings()

    @staticmethod
    def generateScenario(b, Q, C_total):
        """
          create scenario where each customer has a chance of 50% to be 20% above nominal and a 3% chance to be closed
          :param b: dict of (nominal) demands
          :param Q: vehicle capacity. no single demand will be bigger than Q
          :param C_total: total warehouse capacity, sum of demands will not be bigger than this.
          :return: scenario as dict with keys like b
          """
        # nominal beta (here from toro30)
        # b = {5: 5, 6: 3, 7: 5, 8: 5, 9: 4, 10: 3, 11: 5, 12: 4, 13: 3, 14: 4, 15: 5, 16: 3, 17: 5, 18: 4, 19: 3, 20: 5, 21: 3,
        #     22: 5, 23: 5, 24: 3, 25: 5, 26: 5, 27: 3, 28: 4, 29: 5, 30: 3, 31: 5, 32: 3, 33: 5, 34: 5}
        deviation = 1.2
        prob_dev = 0.5
        prob_closed = 0.03
        sc = {}
        choices_dev = random.choices([1, deviation], weights=[1 - prob_dev, prob_dev], k=len(b))
        choices_dev = iter(choices_dev)
        choices_closed = random.choices([1, 0], weights=[1 - prob_closed, prob_closed], k=len(b))
        choices_closed = iter(choices_closed)
        for j in b:
            sc[j] = min(b[j] * next(choices_dev) * next(choices_closed), Q)
        if sum(sc.values()) > C_total:
            if sum(b.values()) > C_total:
                raise Exception("Nominal demands more than total warehouse capacity")
            return Instance.generateScenario(b, Q, C_total)
        return sc

    @staticmethod
    def createInstance(no_customers=None, no_warehouses=None, no_scenarios=50, sample_number = None, cost_factor=1/40, instance = None, with_nominal=False):
        """
        creates a RecoverableInstance with no_scenarios random scenarios. The scenarios are reproducible for a given no_scenarios and sample number.
        A bigger no_scenarios with same sample number will always be a superset of scenarios of the smaller no_scenarios with this sample number.
        The number of customers and facilities is given by the max number of the instance data (10 WH, 200 customers) if not given otherwise.
        :param no_customers: number of customers that should be added
        :param no_warehouses: number of warehouses that should be added
        :param no_scenarios: number of scenarios that should be added
        :param cost_factor: multiply the warehouse fixed and variable costs by this factor
        :param sample_number: number of sample. can be any number, but usually 1, 2, 3, 4 and so on. None, if system random should be used
        :return: instance with beta_k_j indexed from 1 on to all other scenarios. beta_k_j will have a length of no_scenarios. Scenario 0 is nominal if with_nominal=True, otherwise no key 0 exists.
        """
        if instance is None:
            ti = toro_instances.TORO200
        else:
            ti = instance

        # important:
        random.seed(sample_number)

        beta_k_j = {}
        for i in range(1,no_scenarios+1):
            beta_k_j[i] = Instance.generateScenario(ti.beta_j, ti.Q, sum(ti.C_i.values()))
        if with_nominal:
            #nominal scenario at position 0 (not used usually, but we make it available for the tests)
            beta_k_j[0] = ti.beta_j

        loc = np.full(fill_value=np.nan, shape=(max(ti.loc.keys())+1,2))
        for k in ti.loc.keys():
            loc[k,:] = ti.loc[k]
        c_ij = np.hypot(loc[:, 0, None] - loc[:, 0, None].T, loc[:, 1, None] - loc[:, 1, None].T)

        ret = Instance(range(ti.m),
                       range(ti.m, ti.m + ti.n),
                       beta_k_j,
                       [PenalizedCost(cost_factor * ti.d_i[i], 1.5 * cost_factor * ti.d_i[i]) for i in range(ti.m)],
                       [PenalizedCost(cost_factor * ti.f_i[i], 1.5 * cost_factor * ti.f_i[i]) for i in range(ti.m)],
                       c_ij,
                       ti.alpha * ti.emissions_per_fuel_unit * c_ij,
                       ti.gamma * ti.emissions_per_fuel_unit * c_ij,
                       ti.F,
                       list(ti.C_i.values()),
                       ti.Q,
                       loc,
                       name=f"TORO200 (CF=1/{1/cost_factor:.0f}, s={sample_number})",
                       sample_number=sample_number)

        if no_customers is not None:
            ret.J = ret.J[:no_customers]
        if no_warehouses is not None:
            ret.I = ret.I[:no_warehouses]

        return ret