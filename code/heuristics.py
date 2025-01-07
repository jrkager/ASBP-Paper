
def timelimit_heuristic(app, k, ssmodels, first_stage, warmstart, bound, params, log, stats):
    ssm = params.app.SecondStageModel(instance=params.app.inst, k=k,
                                      first_stage_solution=first_stage,
                                      warmstart=warmstart)
    ssm._time_over = False
    ssm._ireason = None
    ssm.Params.Threads = params.n_threads
    ssmodels[k] = ssm

    ssm.params.timelimit = params.HEURTIMELIMIT
    ssm.params.BestObjStop = bound - 1E-3
    ssm.optimize()
    stats.JUMPS += 1
    th = ssm.runtime
    th_proc = ssm._accProctime

    # doesnt matter if we dont find a solution here

    ssm._accRuntime = 0 # accRuntime should not include the runtime of the ssmodel that was used for heuristic part
    ssm._accProctime = 0

    log(f"init sc {ssm._k:3d} in {th:5.2f}s to gap {100 * ssm.mipgap:4.1f}% ([{ssm.objbound:.2f},{ssm.objval:.2f}])\n")
    return th, th_proc, ssm.objval, ssm.objbound
