from .. import InstanceStrings

import numpy as np

from itertools import product, chain
import os
import random

from helper import vprint


class Instance:
    def __init__(self, N, i, homogeneous=True, max_number_scenarios=None):
        """
        scenarios are numbered from 1 upwards
        vessels are numbered from 0 upwards (python style)
        order will always be the same, no matter max_number_scenarios
        :param N: number of vessels
        :param i: instance number (from 1 to 10)
        :param max_number_scenarios: can sepcify the number of scenarios to generate, otherwise generates the max number of scenarios possible with the described budgeted method.
        self.scenarios will be a list of indices, and self.vessel_arrival_time will be indexable with the scenario number first and then the vessel number
        """
        if not max_number_scenarios:
            max_number_scenarios = np.inf

        filename =  os.path.join(os.path.dirname(os.path.abspath(__file__)), f'instances/R_{N}_{i}.dat')
        data_read = self.read_data(filename)
        if not data_read:
            raise Exception(f"Error when reading data file {filename} not found!")

        self.N = N
        self.instance_number = i
        self.J = 34
        self._M = 60
        self.V = range(N)
        self.T = range(1, self.M + 1)
        self.B = range(self.J + 1)
        self.G = range(7)
        self.slack = 1 # time safety space between two vessels
        self.NC = {k: 4 for k in self.V}  # maximum number of cranes that can simultaneously work on a vessel k

        self.scenarios = []
        self.vessel_arrival_time = {}
        self.generate_scenarios(max_number_scenarios)
        vprint(2, "generated", len(self.scenarios), "scenarios for N=", self.N)

        self.name = f"BACASP {N} {i} {homogeneous}"

        p = self.crane_processing_rate
        if homogeneous:
            p[2] = p[0]
            p[3] = p[0]
        else:
            p[2] = 319001
            p[3] = 319001

        K = len(self.scenarios)

        self.strings = InstanceStrings()
        self.strings.ALG_INTRO_TEXT = f"algorithm for N = {N}, i = {i}, #scenarios={K}\n"
        self.strings.UNIQUE_IDENTIFIER = f"{N}-{i}-{K}"
        if not homogeneous:
            self.strings.ALG_INTRO_TEXT = f"algorithm for N = {N}, i = {i} (heterog), #scenarios={K}\n"
            self.strings.UNIQUE_IDENTIFIER = f"{N}-{i}-{K}-het"

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        self._M = value
        self.T = range(1, self._M + 1)

    def read_data(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

            # Reading the lines according to the provided structure
            self.vessel_arrival_time_deterministic = self.parse_line(lines[0])
            self.vessel_load = self.parse_line(lines[2])
            self.crane_processing_rate = self.parse_line(lines[4])
            self.crane_from = self.parse_line(lines[6])
            self.crane_to = self.parse_line(lines[8])
            self.vessel_length = self.parse_line(lines[10])
            return True
        return False

    def generate_scenarios(self, max_number_scenarios):
        """
        :param max_number_scenarios:
        :return:
        """

        GAMMA = 2
        temp_sc_list = []

        bound1 = round(self.N / 3)
        bound2 = 2 * round(self.N / 3)
        for ii1, r1 in chain(product(range(0, bound1), (0.5, 1)), ((0, 0),)):
            for ii2, r2 in chain(product(range(bound1, bound2), (0.5, 1)), ((bound1, 0),)):
                for ii3, r3 in chain(product(range(bound2, self.N), (0.5, 1)), ((bound2, 0),)):
                    temp_sc_list.append(self.vessel_arrival_time_deterministic.copy())
                    temp_sc_list[-1][ii1] = temp_sc_list[-1][ii1] + round(r1 * GAMMA)
                    temp_sc_list[-1][ii2] = temp_sc_list[-1][ii2] + round(r2 * GAMMA)
                    temp_sc_list[-1][ii3] = temp_sc_list[-1][ii3] + round(r3 * GAMMA)


        random.seed(1)
        random.shuffle(temp_sc_list)
        self.scenarios = list(range(1, min(len(temp_sc_list),max_number_scenarios)+1))
        self.vessel_arrival_time = {k+1:v for k,v in enumerate(temp_sc_list[:len(self.scenarios)])}


    @staticmethod
    def parse_line(line):
        return [int(x) for x in line.split(':')[1].strip()[1:-1].split()]

    @staticmethod
    def createInstance(N, instance=1, K=None, homogeneous=True):
        scenarios = None
        if K:
            scenarios = list(range(1, K+1))
        return Instance(N, instance, homogeneous=homogeneous, max_number_scenarios=K)