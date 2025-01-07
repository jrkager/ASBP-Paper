
import helper
helper.VerbosityManager.global_verbosity = 1

from algorithm_types import AlgorithmType
from classes import AlgorithmParams, TimeoutException
from algorithm import ourAlgorithm, toenAlgorithm, rodrAlgorithm

from applications.bacasp.instance import Instance
import applications.bacasp.model as model
from applications.bacasp.heuristic import bacasp_heuristic
from applications import Application

ships = 6
scenarios = 16
sample = 1
gap = 0.05
timelimit = 30 * 60
alg = "our" # or "toen" or "rodr"

type = AlgorithmType() # use standard values
inst = Instance.createInstance(N=ships, K=scenarios, instance=sample)
appl = Application(inst=inst, MasterModel=model.MasterModel,
                   SecondStageModel=model.SecondStageModel)
params = AlgorithmParams(app=appl,
                         start_sc=model.initialize_first_subset(inst),
                         desired_gap=gap,
                         heuristic=bacasp_heuristic,
                         total_timelimit=timelimit,
                         n_threads=1)

try:
    if alg == "our":
        s = ourAlgorithm(params=params, type=type)
    elif alg == "toen":
        s = toenAlgorithm(params=params)
    elif alg == "rodr":
        s = rodrAlgorithm(params=params)
    else:
        raise Exception("Wrong alg name given.")
    timeout_reached = False
except TimeoutException as tex:
    s = tex.stats
    s.TIME_TOT = timelimit
    s.reached_gap = tex.reached_gap
    timeout_reached = True


print(f"Time total: {s.TIME_TOT_PROC:.3f}, Time Master: {s.TIME_MASTER_PROC:.3f}, Time Second Stage: {s.TIME_SS_PROC:.3f}, "
      f"Reached gap: {s.reached_gap:.5f}, Iterations: {s.ITERATIONS}, Timeout reached: {timeout_reached}\n")