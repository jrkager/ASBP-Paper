
from gurobipy import GRB
from dataclasses import dataclass
import time, datetime
import os.path
import numpy as np

ALG_VERSION = "v2"

from heuristics import timelimit_heuristic

import helper
getValueArray = helper.getValueArray
roundBinaryValue = helper.roundBinaryValue
from helper import vprint, fseconds

from algorithm_types import AlgorithmType
from algorithm_types import ChooseNextTypes as ctypes
from algorithm_types import AlgorithmTypes as types
from algorithm_types import InterruptConsequence
from algorithm_types import AlgorithmOptions

from applications import SecondStageModelType
from classes import SecondStagePlaceholder, AlgorithmStats, AlgorithmParams, TimeoutException


class StopAlgException(Exception):
    def __init__(self, reached_gap, text):
        super().__init__(text)
        self.reached_gap = reached_gap

def log(s, params):
    if params.logfile:
        params.logfile.write(s)
        params.logfile.flush()
    else:
        print(s, end="")

@dataclass
class CBTerminationData:
    type: AlgorithmType
    remaining_scenarios: set
    D: set
    S: set
    p: float
    params: AlgorithmParams
    EPS: float
    master_time: float = None
    add_to_D: list[int] = None
    # Delta: float = None
    # Q: float = None
    message: str = None
    ssmodels: dict[int, SecondStageModelType] = None
    stats: AlgorithmStats = None
    bound: float = None
    starttime: float = None


def filter_and_terminate(model, k, UB, LB, log, cbt : CBTerminationData = None):
    """
    Calling this after .optimize is finished. If only the dual solution changes, the callback is not called.
    That's why we (almost) duplicated the function and call it again manually.
    """

    rs = cbt.remaining_scenarios
    EPS = cbt.EPS

    # if model was kicked out
    if k not in rs:
        return

    def interrupt(message, reason):
        if message: log(message)
        model._ireason = reason

    def check_break_iter():
        nonlocal cbt, rs, UB, LB

        # ONLY_ONE
        if len(rs) == 1:
            rse = next(iter(rs)) # get the only remaining scenario
            LBrse = LB[rse] # otherwise alg stops when one sc is left but it could not be the worst(others kicked by z bound)
            if LBrse >= cbt.bound + EPS and rse not in cbt.D:
                cbt.add_to_D = [rse]
                interrupt(None, types.ONLY_ONE)
                return True

        # WORST_TIMELIMITS_REACHED
        # this is handled in iteration method right at the beginning of the iteration

    def check_terminate():
        nonlocal cbt, rs, UB, LB
        if len(rs) == 0:
            interrupt("   No scenario left.\n", types.ONLY_D)
            return True

        # ONLY_D
        if cbt.type.is_set(types.NO_Z_BOUND) and rs <= cbt.D:
            interrupt("   Only scenarios in D left.\n", types.ONLY_D)
            return True

        # D_OPT_IS_WORST
        if cbt.type.is_set(types.NO_Z_BOUND):
            maxub = UB[max(rs, key=UB.get)] - cbt.EPS
            for j in rs:
                if j in cbt.D and UB[j] >= maxub and LB[j] >= UB[j] - cbt.EPS:
                    interrupt(None, types.D_OPT_IS_WORST)
                    return True


    # kick out others
    for j in rs - {k}:
        if UB[j] <= LB[k] - EPS: # must lie considerably under the LB. in the other way round it's useful if it's still here
            log(f"   Removing scenario {j} (LB[k] bigger)\n")
            rs.remove(j)
    # kick out myself
    # only if I am not infeasible (ie infeasible and LB=UB=inf)
    # in filter_and_terminate_cb this has not to be tested since then it is no MIPSOL
    if UB[k] != np.inf:
        if UB[k] <= cbt.bound + EPS: #using + EPS bc for bacasp we have int values
            rs.remove(k)
            if check_terminate():
                return
            if check_break_iter():
                return
            interrupt(f"   Removing scenario {k} (current sc) (z bound)\n", types.OTHER_CHOICE)
            return
        if UB[k] <= max(LB.values()) - EPS: # with - EPS I dont have to exclude k from LB. (otherwise problem when gap 0 on k)
            rs.remove(k)
            if check_terminate():
                return
            if check_break_iter():
                return
            interrupt(f"   Removing scenario {k} (current sc) (UB smaller)\n", types.OTHER_CHOICE)
            return

    if check_terminate():
        return
    if check_break_iter():
        return

    # check remaining interruption criterions
    max_val_UB = max([UB[j] for j in rs])
    candidates_UB = {j for j in rs if UB[j] >= max_val_UB - EPS}
    if k not in candidates_UB:
        interrupt(None, types.OTHER_CHOICE)
        return True

def filter_and_terminate_cb(model, where, *, k, UB, LB, log, cbt : CBTerminationData = None, use_lower_bound = True):
    # gets run only in our algorithm in thre callback
    if where == GRB.Callback.MIPSOL:
        rs = cbt.remaining_scenarios
        EPS = cbt.EPS
        UB[k] = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        if use_lower_bound:
            LB[k] = model.cbGet(GRB.Callback.MIPSOL_OBJBND)

        # gurobi model doesn't terminate immediately. To prevent additional checks, we return right here
        if model._alreadyterminated and model._ireason.consequence() == InterruptConsequence.TERMINATE:
            # there are break iteration reasons that can't become terminate alg. reasons even with more time (for example ONLY_ONE).
            # also in these cases we can return here
            return

        # if model was kicked out
        if k not in rs:
            return

        def interrupt(message, reason):
            # we should set the reason again, maybe the current scenario got better even after
            # terminating and led to a stronger interruption rule
            if message: log(message)
            model._ireason = reason
            model._alreadyterminated = True
            model.terminate()

        def check_break_iter():
            nonlocal cbt, rs, UB, LB

            # ONLY_ONE is default inttype
            if len(rs) == 1:
                rse = next(iter(rs)) # get the only remaining scenario
                LBrse = LB[rse] # otherwise alg stops when one sc is left but it could be that this can still be kicked by z bound
                if LBrse >= cbt.bound + EPS and rse not in cbt.D:
                    cbt.add_to_D = [rse]
                    interrupt(None, types.ONLY_ONE)
                    return True
                # the case where rse in cbt.D is covered by check_terminate

            # WORST_TIMELIMITS_REACHED
            # this is handled in iteration method right at the beginning of the iteration

        def check_terminate():
            nonlocal cbt, rs, UB, LB

            if len(rs) == 0:
                interrupt("   No scenario left.\n", types.ONLY_D)
                return True

            # ONLY_D
            if cbt.type.is_set(types.NO_Z_BOUND) and rs <= cbt.D:
                interrupt("   Only scenarios in D left.\n", types.ONLY_D)
                return True

            # D_OPT_IS_WORST
            if cbt.type.is_set(types.NO_Z_BOUND):
                maxub = UB[max(rs, key=UB.get)] - cbt.EPS
                for j in rs:
                    if j in cbt.D and UB[j] >= maxub and LB[j] >= UB[j] - cbt.EPS:
                        interrupt(None, types.D_OPT_IS_WORST)
                        return True


        rem_sc_changed = False
        # kick out others
        for j in rs - {k}:
            if UB[j] <= LB[k] - EPS:
                log(f"   Removing scenario {j} (LB[k] bigger)\n")
                rs.remove(j)
                rem_sc_changed = True
        # kick out myself
        if UB[k] <= cbt.bound + EPS:
            rs.remove(k)
            if check_terminate():
                return
            if check_break_iter():
                return
            interrupt(f"   Removing scenario {k} (current sc) (z bound)\n", types.OTHER_CHOICE)
            return
        if UB[k] <= max(LB.values()) - EPS: # with - EPS I dont have to exclude k from LB. (otherwise problem when gap 0 on k)
            rs.remove(k)
            if check_terminate():
                return
            if check_break_iter():
                return
            interrupt(f"   Removing scenario {k} (current sc) (UB smaller)\n", types.OTHER_CHOICE)
            return

        if rem_sc_changed:
            if check_terminate():
                return
            if check_break_iter():
                return

        # check if another scenario is bigger now
        max_val_UB = max([UB[j] for j in rs])
        candidates_UB = {j for j in rs if UB[j] >= max_val_UB - EPS}
        if k not in candidates_UB:
            interrupt(None, types.OTHER_CHOICE)
            return True



def opt_and_update_bnd(ssmodels: list[SecondStageModelType], k, UB, LB, cbt, stats: AlgorithmStats, log, callback=None):
    stats.JUMPS += 1

    ssmodels[k]._alreadyterminated = False
    ssmodels[k]._ireason = None

    if cbt.type.is_set(types.VAR_TIMELIMIT):
        ssmodels[k].params.TimeLimit = max(0, cbt.type.options.VAR_TIMELIMIT_MINIMUM - ssmodels[k]._accRuntime, cbt.type.options.VAR_TIMELIMIT_FACTOR * cbt.master_time - ssmodels[k]._accRuntime)
    if cbt.type.is_set(types.FIX_TIMELIMIT):
        ssmodels[k].params.TimeLimit = cbt.type.options.FIX_TIMELIMIT
    if cbt.params.total_timelimit < np.inf:
        ssmodels[k].params.TimeLimit = max(0, min(ssmodels[k].params.TimeLimit, cbt.params.total_timelimit - (time.process_time() - cbt.starttime)))
    log(f"   Setting time limit to {fseconds(ssmodels[k].params.TimeLimit)}.\n")

    ssmodels[k].optimize(callback=callback)
    if cbt.params.total_timelimit - (time.process_time() - cbt.starttime) <= 1:
        # log("\nTIMEOUTEXCEPTION! Stopping...\n\n")
        # gap will be set in iteration function
        ssmodels[k]._ireason = types.TIMEOUT_EXC
        return


    if ssmodels[k].status == GRB.INFEASIBLE or ssmodels[k].status == GRB.INF_OR_UNBD:
        UB[k] = LB[k] = np.inf
    else:
        UB[k] = ssmodels[k].objval
        if cbt.type.is_set(types.TOENISSEN):
            if ssmodels[k].status == GRB.OPTIMAL or ssmodels[k].mipgap <= 0.001:
                LB[k] = UB[k] # sometimes objval and objbound differ quite a bit by the solver, we need them equal.
        else: # ours
            if not cbt.type.is_set(types.NO_LOWER_BOUND) or ssmodels[k].status == GRB.OPTIMAL:
                LB[k] = ssmodels[k].objbound # we could also use the same as above (with optimal check), but it is covered in iteration

    if not cbt.type.is_set(types.TOENISSEN):
        filter_and_terminate(ssmodels[k], k, UB, LB, log, cbt)

    # there is no callback in the TOENISSEN version. We check some conditions here
    if cbt.type.is_set(types.TOENISSEN):
        if max(UB.values()) == np.inf:
            ssmodels[k]._ireason = types.ONLY_ONE
            cbt.add_to_D = [max(UB, key=UB.get)]
            cbt.message = "Worst scenario is unbounded, adding it..."
        # stop when biggest scenario is optimal and bigger than any other UB (considering float inaccuracies)
        elif max(LB.values()) + cbt.EPS >= max(UB.values()):
            worst_sc = max(UB, key=UB.get)
            if worst_sc in cbt.D:
                ssmodels[k]._ireason = types.ONLY_D
                cbt.message = "Worst scenario found and is in D."
            else:
                ssmodels[k]._ireason = types.ONLY_ONE
                cbt.add_to_D = [worst_sc]
                cbt.message = "Worst scenario found and is not in D, adding it..."

    # test the timelimit only here
    if cbt.type.is_set(types.VAR_TIMELIMIT):
        if ssmodels[k].status == GRB.TIME_LIMIT or \
                ssmodels[k]._accRuntime >= max(cbt.type.options.VAR_TIMELIMIT_MINIMUM, cbt.type.options.VAR_TIMELIMIT_FACTOR * cbt.master_time):
            log(f"   Time of {k} ran out ({fseconds(ssmodels[k]._accRuntime)}).\n")
            ssmodels[k]._time_over = True
            if not (ssmodels[k]._ireason and ssmodels[k]._ireason.consequence() in [InterruptConsequence.TERMINATE, InterruptConsequence.BREAK_ITER]):
                ssmodels[k]._ireason = types.VAR_TIMELIMIT
    if cbt.type.is_set(types.FIX_TIMELIMIT):
        if ssmodels[k].status == GRB.TIME_LIMIT or \
                ssmodels[k]._accRuntime >= cbt.type.options.FIX_TIMELIMIT:
            log(f"   Time of {k} ran out ({fseconds(ssmodels[k]._accRuntime)}).\n")
            ssmodels[k]._time_over = True
            if not (ssmodels[k]._ireason and ssmodels[k]._ireason.consequence() in [InterruptConsequence.TERMINATE, InterruptConsequence.BREAK_ITER]):
                ssmodels[k]._ireason = types.FIX_TIMELIMIT

    if ssmodels[k]._ireason is None and ssmodels[k].status == GRB.OPTIMAL:
        ssmodels[k]._ireason = types.OPTIMAL # ireason optimal is stronger than ireason timelimit
        LB[k] = UB[k]

    # this occurs when no second stage timelimit is used (for example in Toenissen or with the corresponding flags in our algorithm disabled) and the
    # global timelimit is reached in this scenario. ssmodels[k].status will be TIME_LIMIT!
    # Using None as ireason forwards the handling of the global timelimit interrupt to other functions
    if ssmodels[k]._ireason is None:
        log(f"status ireason None: {ssmodels[k].status}\n")
        ssmodels[k]._ireason = types.NONE

def init_and_heur(ssmodels, k, dm, UB, LB, D, params: AlgorithmParams, s: AlgorithmStats, log, l, cbt):

    second_stage_warmstart = dm.get_second_stage_solution_for_scenario(k) # is None when k not in D
    first_stage = dm.get_first_stage_solution()
    bound = cbt.bound

    if params.heuristic:
        th, ubound, lbound = params.heuristic(params.app, k, ssmodels, first_stage, second_stage_warmstart, bound, params, log, s)
        th_proc = th
        if ssmodels.get(k, None) is None:
            # time of heuristic does not count towards the timelimit for each scenario
            ssmodels[k] = SecondStagePlaceholder(time_over=False, accTime=0, accProctime=0)
    elif params.HEURTIMELIMIT > 0:
        # this sets ssmodels[k] to a real secondstage problem
        th, th_proc, ubound, lbound = timelimit_heuristic(params.app, k, ssmodels, first_stage, second_stage_warmstart, bound, params, log, s)
    else:
        # no heuristic is used
        ssmodels[k] = SecondStagePlaceholder(time_over=False, accTime=0, accProctime=0)
        th, th_proc, ubound, lbound = 0, 0, np.inf, -np.inf

    s.TIME_HEUR += th
    s.TIME_HEUR_PROC += th_proc

    UB[k] = ubound
    if cbt.type.is_set(types.TOENISSEN) or cbt.type.is_set(types.RODRIGUES) or cbt.type.is_set(types.NO_LOWER_BOUND):
        # lower bound from heuristic (i.e. when gap is not inf) should not be used to kick out anything. Later when a scenario is solved to optimality we use it
        LB[k] = -np.inf
    else:
        LB[k] = lbound

    if cbt.type.is_set(types.DISCARD_HEUR_MODEL):
        ssmodels[k] = SecondStagePlaceholder(time_over=False, accTime=0, accProctime=0)

def iteration(EPS, LB, UB, cbt : CBTerminationData, dm, log, params, remaining_scenarios, type):
    # we land here only in toenissen and our alg
    # - terminate: throws exc (set reached gap in the stopalg exception!)
    # - break iteration: returns
    # - choose other scenario: happens within method
    # returns what should be added to D (as list) and a message

    D = cbt.D
    S = cbt.S
    s : AlgorithmStats = cbt.stats
    ssmodels : list[SecondStageModelType] = cbt.ssmodels


    while True:
        if len(remaining_scenarios) == 0:
            worse_sc = max(S, key=UB.get)
            if worse_sc in D:
                raise StopAlgException(s.reached_gap, "Worst scenario is in D. STOP.\n")
            if UB[worse_sc] <= cbt.bound:
                raise StopAlgException(s.reached_gap, "Desired gap reached. STOP.\n")
            else:
                return [worse_sc], "No remaining scenario\n"

        # assuming we always use HIGHEST UB
        max_val_UB = max([UB[k] for k in remaining_scenarios])
        candidates_UB = {k for k in remaining_scenarios if UB[k] >= max_val_UB - EPS}

        # OPTIMAL
        if not type.is_set(types.TOENISSEN):
            responsible_k = next((k for k in candidates_UB if ssmodels[k].status == GRB.OPTIMAL), None)
            if responsible_k is not None:
                if responsible_k in D:
                    raise StopAlgException(s.reached_gap, "A highest UB scenario is optimal and in D. STOP.\n")
                else:
                    return [responsible_k], types.OPTIMAL.str_reason()

        # WORST_TIMELIMITS_REACHED
        if type.is_set(types.WORST_TIMELIMITS_REACHED):
            # there is a k not in D with timeover under the worst UB scenarios
            responsible_k = next((k for k in candidates_UB if k not in D and ssmodels[k]._time_over), None)
            if responsible_k is not None:
                return [responsible_k], types.WORST_TIMELIMITS_REACHED.str_reason()

            # if any candidates are in D and have timeover, but we have a sc not in D remaining, we return that one.
            if any(k in D and ssmodels[k]._time_over for k in candidates_UB):
                if len(remaining_scenarios - D) > 0:
                    return [max(remaining_scenarios - D, key=UB.get)], "Worst with timeover is in D.\n"

        next_sc = candidates_UB.pop()

        # model was already created
        if not isinstance(ssmodels[next_sc], SecondStagePlaceholder):
            if type.is_set(types.TOENISSEN):
                log(f" - continue with {next_sc} (gap {100 * ssmodels[next_sc].mipgap:.1f}%), "
                    f"left: {len([k for k in S if ssmodels[k].mipgap > 1E-3])} "
                    f"[{LB[next_sc]:.2f},{UB[next_sc]:.2f}]\n")
            else:
                log(f" - continue with {next_sc} (gap {100 * ssmodels[next_sc].mipgap:.1f}%), "
                    f"left: {len(remaining_scenarios)} "
                    f"[{LB[next_sc]:.2f},{UB[next_sc]:.2f}]\n")
        else:
            # creating mip model
            log(f" - starting with {next_sc}, "
                f"left: {len(remaining_scenarios) if not type.is_set(types.TOENISSEN) else len([k for k in S if ssmodels[k].mipgap > 1E-3])} "
                f"[{LB[next_sc]:.2f},{UB[next_sc]:.2f}]\n")
            second_stage_warmstart = dm.get_second_stage_solution_for_scenario(next_sc)  # is None when k not in D
            first_stage = dm.get_first_stage_solution()
            ssm = params.app.SecondStageModel(instance=params.app.inst, k=next_sc,
                                              first_stage_solution=first_stage,
                                              warmstart=second_stage_warmstart)
            ssm._time_over = False
            ssm._ireason = None
            ssm.Params.Threads = params.n_threads
            ssmodels[next_sc] = ssm

        # start solving
        ssmodels[next_sc].resetParamsButLogAndThreads()
        if type.is_set(types.TOENISSEN):
            # toenissen algorithm (theirs) solves the scenario with biggest UB to optimality before continuing
            callback = None
        else:
            callback = lambda model, where: filter_and_terminate_cb(model, where, k=next_sc, UB=UB, LB=LB, log=log, cbt=cbt, use_lower_bound=not cbt.type.is_set(types.NO_LOWER_BOUND))

        opt_and_update_bnd(ssmodels, next_sc, UB, LB, cbt, s, log=log, callback=callback)

        ireason = ssmodels[next_sc]._ireason
        if ireason.consequence() == InterruptConsequence.TERMINATE or ireason.consequence() == InterruptConsequence.BREAK_ITER:
            # erreichten gap berechnen
            worst_ub = max(UB.values())
            LB_master = dm.objbound if dm else -np.inf
            f_x_tilde = dm.get_first_stage_objective()
            try:
                calc_gap = abs( (f_x_tilde + worst_ub - LB_master) / (f_x_tilde + worst_ub) )
            except ZeroDivisionError:
                calc_gap = np.inf
            if np.isnan(cbt.stats.reached_gap) or calc_gap < cbt.stats.reached_gap:
                cbt.stats.reached_gap = calc_gap

        if ireason == ireason.TIMEOUT_EXC:
            raise TimeoutException(reached_gap=cbt.stats.reached_gap)

        log(f"   Solved {next_sc} to gap {100 * ssmodels[next_sc].mipgap:.2f}% in "
            f"{ssmodels[next_sc].runtime:.2f}s [{LB[next_sc]:.2f},{UB[next_sc]:.2f}]\n")

        log(ireason.str_reason())

        if ireason.consequence() == InterruptConsequence.CHOOSE_AGAIN:
            continue # go to start of method
        if ireason.consequence() == InterruptConsequence.BREAK_ITER:
            return cbt.add_to_D, cbt.message
        if ireason.consequence() == InterruptConsequence.TERMINATE:
            if ireason == ireason.GLOBAL_GAP:
                raise StopAlgException(s.reached_gap, "Desired gap reached. STOP.\n")
            if ireason == ireason.ONLY_D:
                raise StopAlgException(s.reached_gap, "Worst scenario is in D. STOP.\n")
            if ireason == ireason.D_OPT_IS_WORST:
                raise StopAlgException(s.reached_gap, "One of worst scenarios is in D. STOP.\n")
        log(f"Consequence invalid! {ireason.consequence()}\nContinuing with next scenario.\n")
        continue



def algorithm(params: AlgorithmParams, type: AlgorithmType = None, log_header = True):
    """
    implementation of general algorithm (our, toen and rodr can be chosen via type flags. Some code is shared).
    Some options:
    - turn var timelimit on by passing the appropriate type
    - turn z bound off by passing the NO_Z_BOUND type
    - turn the pi-trick on by setting GLOBAL_GAP as type
    - use toenissen or rodrigues version by passing the appropriate type

    We made the code more clear:
    - we removed the option for a starting first stage


    :return: returns filled AlgorithmStats or terminates with a custom TimeoutException
    """

    starttime = time.process_time()
    starttime_wall = time.perf_counter()
    # using my own log function here because the global one includes a if check, which could be less performing
    if params.logfile is None:
        def log(s, printtime=True):
            if printtime:
                print(str(datetime.timedelta(seconds=int(time.process_time()-starttime))) + "   ", end="")
            if s is None:
                log("NoneType\n")
            print(s, end="")
    else:
        def log(s):
            if s is None:
                s = "NoneType\n"
            if '\n' in s:
                params.logfile.write(str(datetime.timedelta(seconds=int(time.process_time()-starttime))) + "   ")
            params.logfile.write(s)
            params.logfile.flush()

    if type is None:
        print("running our algorithm")
        return ourAlgorithm(params, type)

    if type.is_set(types.VAR_TIMELIMIT) and type.options.VAR_TIMELIMIT_FACTOR is None:
        raise Exception("Specify VAR_TIMELIMIT_FACTOR in options")

    # we add this as default
    if types.ONLY_D not in type.alg_types:
        type.alg_types.append(types.ONLY_D)

    if params.mastermodel_logfolder and not os.path.exists(params.mastermodel_logfolder):
        os.mkdir(params.mastermodel_logfolder)
    if params.ssmodel_logfolder and not os.path.exists(params.ssmodel_logfolder):
        os.mkdir(params.ssmodel_logfolder)

    K = len(params.app.inst.scenarios)
    if log_header:
        log(f"({ALG_VERSION}) General Algorithm (choose={type.choose_next}, int={type.alg_types}, options={type.options}) on {params.n_threads} threads\n")
    log(params.app.inst.strings.ALG_INTRO_TEXT)

    type.choose_next = tuple(type.choose_next)
    type.alg_types = set(type.alg_types)

    s = AlgorithmStats()

    EPS = 1e-4

    if params.MASTER_P is not None and params.desired_gap < params.MASTER_P:
        params.MASTER_P = params.desired_gap
        log(f"MASTER_P was adjusted to desired gap {params.desired_gap}.\n")
    if params.MASTER_P is None:
        params.MASTER_P = params.desired_gap
    log(f"MASTER_P = {params.MASTER_P}, desired gap = {params.desired_gap}. HEURLIMIT = {params.HEURTIMELIMIT}s\n")
    log(f"Total timelimit = {params.total_timelimit}s\n")

    S = set(params.app.inst.scenarios)
    if params.start_sc is None:
        raise Exception("Provide a start_sc list (or empty list)")
    D = set(params.start_sc)
    ssmodels = {}

    iteration_counter = 0
    try: # until StopAlgException is thrown
        while True: # this does the iterations
            iteration_counter += 1
            log(f"\n--- ITERATION {iteration_counter} ---\n")
            log(f"D: {D}\n")


            # solve MASTER model with gap p
            if params.mastermodel_logfolder:
                dlfile = os.path.join(params.mastermodel_logfolder, f"master-{'theirs' if type.is_set(types.TOENISSEN) else 'ours'}-{params.app.inst.strings.UNIQUE_IDENTIFIER}-{iteration_counter}.txt")
            else:
                dlfile = ""

            if type.is_set(types.TOENISSEN) or type.is_set(types.NO_Z_BOUND):
                remaining_scenarios = set(S)
            else:
                remaining_scenarios = set(S) - set(D)
            UB = {k: np.inf for k in S}
            LB = {k: -np.inf for k in S}
            rodr_stopping = False

            dm = params.app.MasterModel(params.app.inst, scenarios=D, LogFile=dlfile)
            dm.Params.Threads = params.n_threads
            dm.Params.MIPGap = params.MASTER_P
            if params.total_timelimit < np.inf:
                dm.Params.TimeLimit = max(0, params.total_timelimit - (time.process_time() - starttime))

            dm.optimize(params.master_callback)

            s.TIME_MASTER += dm._accRuntime
            s.TIME_MASTER_PROC += dm._accProctime
            achieved_p = dm.MIPGap  # actual master gap
            cbt = CBTerminationData(type=type, remaining_scenarios=remaining_scenarios,
                                    D=D, S=S, p=achieved_p, EPS=EPS,
                                    params=params,
                                    master_time=dm._accRuntime,
                                    ssmodels=None,
                                    stats=s,
                                    starttime=starttime)

            if dm.status == GRB.TIME_LIMIT and dm.mipgap > params.MASTER_P:
                raise TimeoutException(reached_gap=s.reached_gap)
            z_tilde = dm.get_second_stage_bound()
            first_stage = dm.get_first_stage_solution()
            f_x_tilde = dm.get_first_stage_objective()
            LB_master = dm.objbound
            if cbt.type.is_set(types.TOENISSEN) or cbt.type.is_set(types.NO_Z_BOUND):
                cbt.bound = -np.inf
            else:
                if cbt.type.is_set(types.GLOBAL_GAP): # use pi trick
                    cbt.bound = (1 - cbt.p) / (1 - params.desired_gap) * z_tilde + (params.desired_gap - cbt.p) / (
                            1 - params.desired_gap) * f_x_tilde
                else:
                    cbt.bound = z_tilde
                for k in D:
                    UB[k] = z_tilde

            log(f"Master model run in {dm._accRuntime:3.2f}s to gap {dm.mipgap * 100:.1f}%\n")
            log(f" Obj. from master model = {f_x_tilde + z_tilde:.1f}\n")
            log(f" z bound from master model = {z_tilde:.1f}\n")
            log(f" used bound = {cbt.bound:.1f}\n")
            # print first stage infos
            log_first_stage_results(dm.get_first_stage_solution(), log, params)

            # build second stage models or solve heuristics
            if "ssmodels" in locals():
                # accumulate time from last iteration (if available)
                s.TIME_SS += sum(ssm._accRuntime for ssm in ssmodels.values())
                s.TIME_SS_PROC += sum(ssm._accProctime for ssm in ssmodels.values())
            ssmodels = {}
            cbt.ssmodels = ssmodels

            for k in remaining_scenarios:
                init_and_heur(ssmodels, k, dm, UB, LB, D, params, s, log, iteration_counter, cbt)
                if time.process_time() - starttime >= params.total_timelimit:
                    raise TimeoutException(reached_gap=s.reached_gap)
                if type.is_set(types.RODRIGUES):
                    rodr_stopping = rodr_alg_part(D, UB, cbt, dm, k, log, params)
                    if rodr_stopping:
                        break
            if type.is_set(types.RODRIGUES) and rodr_stopping:
                continue # with master

            # update reached gap after heuristic phase
            try:
                calc_gap = abs( (f_x_tilde + max(UB.values()) - LB_master) / (f_x_tilde + max(UB.values())) )
            except ZeroDivisionError:
                calc_gap = np.inf
            if np.isnan(s.reached_gap) or calc_gap < s.reached_gap:
                s.reached_gap = calc_gap


            # prefiltering
            if not type.is_set(types.TOENISSEN):
                for k in remaining_scenarios.copy():
                    if UB[k] <= cbt.bound + EPS:
                        remaining_scenarios.remove(k)
                        log(f"Removing scenario {k} (z bound)\n")
                        continue
                    if not type.is_set(types.RODRIGUES) and len(remaining_scenarios) > 1: # darf nicht von eigenem LB gekickt werden
                        if UB[k] <= max(LB[i] for i in remaining_scenarios - {k}) + EPS:
                            remaining_scenarios.remove(k)
                            log(f"Removing scenario {k} (UB smaller)\n")
                if remaining_scenarios <= D:
                    # when z bound or RODR is used, remaining_scenarios it is empty because all D were removed before
                    log(f"Master gap = {achieved_p * 100:.2f}%, Reached gap = {s.reached_gap * 100:.2f}%\n")
                    raise StopAlgException(s.reached_gap, "Stopping because all remaining scenarios are in D already or below zbound (in prefiltering).\n")


            # ITERATION
            # - terminate: throws exc
            # - break iteration: returns
            # - choose other scenario: happens within method
            add_to_D, message = iteration(EPS, LB, UB, cbt, dm, log, params, remaining_scenarios, type)
            log(f"{message}\n")

            D.update(add_to_D)
            outstring = ", ".join(f"{sc} [{LB[sc]:.2f}, {UB[sc]:.2f}]" for sc in add_to_D)
            log(f"+ Adding scenarios {{{outstring}}} to D.\n")
            log(f"Reached gap = {s.reached_gap*100:.2f}%.\n")

            continue # with master

    except StopAlgException as lpe:
        s.TIME_TOT = time.perf_counter() - starttime_wall
        s.TIME_TOT_PROC = time.process_time() - starttime
        s.TIME_SS += sum(ssm._accRuntime for ssm in ssmodels.values())
        s.TIME_SS_PROC += sum(ssm._accProctime for ssm in ssmodels.values())
        log(str(lpe))
        log(f"Finished algorithm with gap {cbt.stats.reached_gap * 100:.2f}%\n")
    except TimeoutException as timeout_exc:
        s.TIME_TOT = time.perf_counter() - starttime_wall
        s.TIME_TOT_PROC = time.process_time() - starttime
        s.TIME_SS += sum(ssm._accRuntime for ssm in ssmodels.values())
        s.TIME_SS_PROC += sum(ssm._accProctime for ssm in ssmodels.values())
        timeout_exc.stats = s
        log(f"# Stopping because of Timeout ({params.total_timelimit}s) with gap {s.reached_gap*100:.2f}% #\n\n")
        raise timeout_exc


    log(f"Scenarios in final selection: {D}\n")
    log(f"Iterations needed: {iteration_counter}\n")
    log(f"Final cost = {dm.objval:.3f}\n")

    s.first_stage = first_stage
    s.OPENED = int(sum(first_stage[0].values()))
    s.COST = dm.objval
    s.ITERATIONS = iteration_counter
    log(f"TIME_MASTER = {s.TIME_MASTER:.3f}, TIME_SS = {s.TIME_SS:.3f}, TIME_TOT = {s.TIME_TOT:.3f}\n")
    log(f"TIME_MASTER = {s.TIME_MASTER_PROC:.3f}, TIME_SS = {s.TIME_SS_PROC:.3f}, TIME_TOT = {s.TIME_TOT_PROC:.3f} (proc times)\n")
    log(f"JUMPS = {s.JUMPS}\n\n\n\n")

    return s


def rodr_alg_part(D, UB, cbt, dm, k, log, params):
    if UB[k] <= cbt.bound + cbt.EPS:
        log(f" - RODR heur kicks sc {k}\n")
    else:
        # immediately solve the scenario to optimality if it was not kicked by HEUR + zbound
        log(f" - RODR heur {UB[k]:.3f} > zbound {cbt.bound:.3f}\n")

        if isinstance(cbt.ssmodels[k], SecondStagePlaceholder):
            second_stage_warmstart = dm.get_second_stage_solution_for_scenario(k)  # is None when k not in D
            first_stage = dm.get_first_stage_solution()
            ssm = params.app.SecondStageModel(instance=params.app.inst, k=k,
                                              first_stage_solution=first_stage,
                                              warmstart=second_stage_warmstart)
            ssm.Params.Threads = params.n_threads
            cbt.ssmodels[k] = ssm
        else:
            log(f"  - re-use model from timeheuristic\n")
            ssm = cbt.ssmodels[k]
            ssm.resetParamsButLogAndThreads()
        ssm.params.TimeLimit = max(0, params.total_timelimit - (time.process_time() - cbt.starttime))
        ssm.optimize()
        UB[k] = ssm.objval
        log(f" - solving {k} to opt, "
            f"objval {ssm.objval:.3f}\n")
        if time.process_time() - cbt.starttime >= params.total_timelimit:
            raise TimeoutException(reached_gap=cbt.stats.reached_gap)
        if ssm.objval > cbt.bound:
            D.update([k])
            log(f"+ Adding scenario {k} with obj {ssm.objval:.2f} to D\n")
            return True
    return False


def log_first_stage_results(first_stage, log, params):
    if params.app.inst.name.startswith("TORO"):
        sizes_string = ",".join(f"{v:.2f}" for v in first_stage[1].values())
        log(f" Opened {int(sum(first_stage[0].values()))} locations (sizes: [{sizes_string}])\n")
    if params.app.inst.name.startswith("BACASP"):
        vprint(2, "b =", first_stage[2])
        for l in params.app.inst.V:
            vprint(2, f"ship {l} berths AFTER {[v for v in params.app.inst.V if first_stage[0][v, l]]} departed")
        for l in params.app.inst.V:
            vprint(2, f"ship {l} berths BELOW {[v for v in params.app.inst.V if first_stage[1][v, l]]}")
        vprint(2, f"x = 1 for {sum(first_stage[0][k, l] for k in params.app.inst.V for l in params.app.inst.V)}")
        vprint(2, f"y = 1 for {sum(first_stage[1][k, l] for k in params.app.inst.V for l in params.app.inst.V)}")


def toenAlgorithm(params: AlgorithmParams, type: AlgorithmType = None):
    """
    calls toenissen ISAM algorithm with appropriate type specs
    """
    log(f"({ALG_VERSION}) Improved Toenissen on {params.n_threads} threads\n", params)

    alg_types = [types.TOENISSEN]
    if type is not None:
        if types.DISCARD_HEUR_MODEL in type.alg_types:
            alg_types = [types.TOENISSEN, types.DISCARD_HEUR_MODEL]
    type = AlgorithmType(choose_next=[ctypes.HIGHEST_UB],
                         alg_types=alg_types)

    log(f"with additional types: {type}\n", params)

    if not params.start_sc:
        log("Filling start scenario with an empty list.\n", params)
        params.start_sc = []

    return algorithm(params, type, log_header=False)

def rodrAlgorithm(params: AlgorithmParams, type: AlgorithmType = None):
    """
    calls rodrigues SRP algorithm with appropriate type specs
    """
    log(f"({ALG_VERSION}) Rodrigues on {params.n_threads} threads\n", params)

    alg_types = [types.RODRIGUES]
    if type is not None:
        if types.DISCARD_HEUR_MODEL in type.alg_types:
            alg_types = [types.RODRIGUES, types.DISCARD_HEUR_MODEL]
    type = AlgorithmType(choose_next=[ctypes.HIGHEST_UB],
                         alg_types=alg_types)
    log(f"with additional types: {type}\n", params)

    if not params.start_sc:
        log("Filling start scenario with an empty list.\n", params)
        params.start_sc = []

    return algorithm(params, type, log_header=False)

def ourAlgorithm(params: AlgorithmParams, type: AlgorithmType = None):
    """
    if no type is given use the standard version
    calls our algorithm with appropriate type specs
    """
    log(f"({ALG_VERSION}) Our Algorithm on {params.n_threads} threads\n", params)

    if type is None:
        options = AlgorithmOptions(VAR_TIMELIMIT_FACTOR=2, VAR_TIMELIMIT_MINIMUM=1)
        type = AlgorithmType(alg_types=[types.VAR_TIMELIMIT, types.GLOBAL_GAP], options=options)
    if type.options is None:
        type.options = AlgorithmOptions(VAR_TIMELIMIT_FACTOR=2, VAR_TIMELIMIT_MINIMUM=1)
    if type.options.VAR_TIMELIMIT_FACTOR is None:
        type.options.VAR_TIMELIMIT_FACTOR = 2
    if type.options.VAR_TIMELIMIT_MINIMUM is None:
        type.options.VAR_TIMELIMIT_MINIMUM = 1
    if type.alg_types is None:
        type.alg_types = [types.VAR_TIMELIMIT, types.GLOBAL_GAP]

    log(f"with additional types: {type}\n", params)
    # set default values for our alg
    if ctypes.HIGHEST_UB not in type.choose_next:
        type.choose_next.insert(0, ctypes.HIGHEST_UB)
    # types.OPTIMAL, types.OTHER_CHOICE are not appended because those are used in the algorithm anyways!
    for it in [types.ONLY_D, types.D_OPT_IS_WORST, types.ONLY_ONE]:
        if it not in type.alg_types:
            type.alg_types.append(it)

    if types.VAR_TIMELIMIT in type.alg_types or types.FIX_TIMELIMIT in type.alg_types:
        if types.WORST_TIMELIMITS_REACHED not in type.alg_types:
            type.alg_types.append(types.WORST_TIMELIMITS_REACHED)

    return algorithm(params, type, log_header=False)
