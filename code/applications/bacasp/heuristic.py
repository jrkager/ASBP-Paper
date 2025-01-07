from .. import Application
from .instance import Instance
from itertools import product
import time
import numpy as np
from helper import vprint

def set(var, val):
    var.LB=var.UB=val

def bacasp_heuristic(app:Application, ii, ssmodels, first_stage, warmstart, bound, params, log, stats):
    inst = app.inst
    first_stage_solution = first_stage

    vprint(2, "scenario", ii)
    vprint(3, "arrival =", {v: round(inst.vessel_arrival_time[ii][v]) for v in inst.V})

    def run_heur_version(op1, op2):
        ret = bacasp_inner_heuristic(op1, op2, inst, first_stage_solution, ii, log)
        return ret

    def run_versions():
        best, best_option = np.inf, ""

        curr = run_heur_version(1, 2)
        if curr < best:
            best, best_option = curr, "(1, 2)"
            if best <= bound:
                return best, best_option
        curr = run_heur_version(1, 0)
        if curr < best:
            best, best_option = curr, "(1, 0)"
            if best <= bound:
                return best, best_option
        curr = run_heur_version(2, 2)
        if curr < best:
            best, best_option = curr, "(2, 2)"
            if best <= bound:
                return best, best_option
        curr = run_heur_version(2, 0)
        if curr < best:
            best, best_option = curr, "(2, 0)"
            if best <= bound:
                return best, best_option
        for _ in range(100):
            curr = run_heur_version(1, 1)
            if curr < best:
                best, best_option = curr, f"(1, 1)"
                if best <= bound:
                    return best, best_option
            curr = run_heur_version(2, 1)
            if curr < best:
                best, best_option = curr, f"(2, 1)"
                if best <= bound:
                    return best, best_option
        return best, best_option

    starttime = time.process_time()

    best, best_option = run_versions()

    th = time.process_time() - starttime

    stats.JUMPS += 1
    log(f"init sc {ii:3d} with {best_option} in {th:5.3f}s to [NA,{best:.2f}]\n")

    return th, best, -np.inf


def is_there_intersection(k, l, b, vessel_length):
    # Define this function based on the logic for intersection
    if (b[l] <= b[k] < b[l] + vessel_length[l]) or \
        (b[k] <= b[l] < b[k] + vessel_length[k]):
        return True
    return False


def bacasp_inner_heuristic(op1, op2, inst: Instance, first_stage_solution, ii: int, log):
    """

    :param op1:
    :param op2: 0 based on arrival, 2 based on carga, 1 random swap
    :param inst:
    :param first_stage_solution:
    :param ii:
    :return:
    """
    x, y, b, pi, sigma = first_stage_solution

    V, G, Q, i, f = inst.V, inst.G, inst.vessel_load, inst.crane_from, inst.crane_to
    T = list(inst.T) #+ list(range(inst.T.stop, inst.T.stop + 100)) # the extension is needed if x+y=1 for a pair of ships
    h, p = inst.vessel_length, inst.crane_processing_rate
    cH = {k: 0 for k in V} # it was emptied in the timestep before cH[k]
    arr_time = {k: round(inst.vessel_arrival_time[ii][k]) for k in V}
    Carga = {k : inst.vessel_load[k] for k in V}
    GN = {(k, g): 1 if b[k] >= i[g] and b[k] + h[k] <= f[g] else 0 for k in V for g in G}
    zHeur = {(g, k): 0 for g in G for k in V}
    Ant_Navio = {(k, j): 0 for k in V for j in T}
    GT = {(g, j): 1 for g in G for j in T}

    custo_atual = 0

    # Greedy approach
    for j in T:
        if j >= 21:
            pass
        if all(Carga[k] == 0 for k in V): # all ships done
            break

        n1 = -1
        n2 = -1
        Navio = {k: 0 for k in V}

        for g, k in product(G, V):
            zHeur[g, k] = 0
        for k in V:
            if arr_time[k] <= j and Carga[k] > 0:
                Navio[k] = 1
        custo_atual += sum(Navio[k] for k in V) # in each iteration add 1 for every ship that is currently waiting. Will then be the sum of waiting times

        for k, l in product(V, V):
            if k != l and x[k,l]: # taking value one if ship l berths after ship k had departed
                if Carga[k] > 0: # ship k has not departed yet
                    Navio[l] = 0 # ship l cannot berth
                    vprint(3, j, f"disabling ship {l}")


        # Prevents ships going to the port from docking if it is occupied
        for k, l in product(V, V):
            if l != k and Navio[k] == 1 and Navio[l] == 1:
                if is_there_intersection(k, l, b, h):
                    if Carga[l] < Q[l]: # if l already started
                        Navio[k] = 0
                    elif Carga[l] == Q[l] and Carga[k] == Q[k]: # prioritize the one with less carga left
                        if Q[l] > Q[k]:
                            Navio[l] = 0
                        else:
                            Navio[k] = 0

        for k, l in product(V, V): # slack (F, inst.slack)
            if l != k and cH[k] == j - inst.slack + 1:
                if j > 1 and Navio[l] == 1:
                    # if is_there_intersection(k, l, b, h): # according to paper the safety slack F is always applied
                    Navio[l] = 0

        navios_total = sum(Navio[k] for k in V)

        # Find most important 2 ships
        if op1 == 1: # smaller cargo
            while navios_total > 2:
                maximo = -1
                ind = -1
                for k in V:
                    if Navio[k] == 1 and Carga[k] > maximo:
                        maximo = Carga[k]
                        ind = k
                navios_total -= 1
                Navio[ind] = 0
        elif op1 == 2: # earlier arrival
            while navios_total > 2:
                maximo = -1
                ind = -1
                for k in V:
                    if Navio[k] == 1 and arr_time[k] > maximo:
                        maximo = arr_time[k]
                        ind = k
                navios_total -= 1
                Navio[ind] = 0

        # find the two ships k that are available
        if n1 >= 0 and n2 >= 0: # n1 and n2 already set
            if Navio[n1] == 0:
                n1 = n2
                n2 = -1
            elif Navio[n2] == 0:
                n2 = -1
        else:
            for k in V:
                if Navio[k] == 1:
                    if n1 == -1:
                        n1 = k
                    else:
                        n2 = k
                        break

        # If there is only one ship
        if n2 == -1 and n1 != -1:
            ng = 0
            for g in G:
                if GN[n1, g] == 1: # crane g is available for full length of ship n1
                    if ng < 4 and Carga[n1] > 0: # max 4 cranes
                        GT[g, j] = 0  # Crane no longer available
                        Carga[n1] = max(0, Carga[n1] - p[g])  # Load decreases
                        if Carga[n1] == 0: # ship finished, set cH to next time
                            cH[n1] = j + 1
                        ng += 1 # number cranes used increments
                        vprint(3,j, "  G ", g, " no ship ", n1, "-> carga ", Carga[n1], "  (ng ", ng, " )")

        # If there are (at least) two ships
        if n2 != -1:
            # Check the minimum time they take
            if b[n1] <= 13 and b[n1] + h[n1] <= 26:
                tmin_n1 = Carga[n1] / (4 * p[0])
            elif b[n1] + h[n1] <= 26:
                tmin_n1 = Carga[n1] / (3 * p[0] + p[2]) # p[2] is the non-homogeneous ship
            else:
                tmin_n1 = Carga[n1] / (2 * p[0])

            if b[n2] <= 13 and b[n2] + h[n2] <= 26:
                tmin_n2 = Carga[n2] / (4 * p[0])
            elif b[n2] + h[n2] <= 26:
                tmin_n2 = Carga[n2] / (3 * p[0] + p[2])
            else:
                tmin_n2 = Carga[n2] / (2 * p[0])

            # wie lange die schiffe, die noch nicht fertig sind und nicht unter den aktuellen 2 sind, warten müssten, wenn sie ankommen
            # bevor schiff n1 bzw n2 fertig ist
            int_n1 = sum(j + np.ceil(tmin_n1) + 1 - arr_time[k] for k in V if k != n1 and k != n2 and cH[k] == 0 and j + np.ceil(tmin_n1) >= arr_time[k])
            int_n2 = sum(j + np.ceil(tmin_n2) + 1 - arr_time[k] for k in V if k != n1 and k != n2 and cH[k] == 0 and j + np.ceil(tmin_n2) >= arr_time[k])
            # wie lange die schiffe, die noch nicht fertig sind und mit n1 intersecten und nicht unter den aktuellen 2 sind, warten müssten, wenn sie ankommen
            # bevor schiff n1 bzw n2 fertig ist. Im paper "slack" genannt und mit min definiert (nicht sum)
            inf_n1 = min([T[-1]] + [arr_time[k] - (j + np.ceil(tmin_n1) + 1) for k in V if k != n1 and k != n2 and cH[k] == 0 and j + np.ceil(tmin_n1) >= arr_time[k]
                         and is_there_intersection(n1, k, b, h)])
            inf_n2 = min([T[-1]] + [arr_time[k] - (j + np.ceil(tmin_n2) + 1) for k in V if k != n1 and k != n2 and cH[k] == 0 and j + np.ceil(tmin_n2) >= arr_time[k]
                         and is_there_intersection(n2, k, b, h)])

            # It is assumed that ship 1 always has priority
            if int_n1 == 0 and int_n2 > 0: # If there is none in front of n1, the priority changes
                n1, n2 = n2, n1
                tmin_n1, tmin_n2 = tmin_n2, tmin_n1

            # If both have ships ahead, serve the one with less load
            if int_n2 > 0 and int_n1 > 0:
                if op2 == 2 and inf_n2 > inf_n1:
                    n1, n2 = n2, n1
                    tmin_n1, tmin_n2 = tmin_n2, tmin_n1

                if op2 == 0 and Carga[n2] < Carga[n1]:
                    n1, n2 = n2, n1
                    tmin_n1, tmin_n2 = tmin_n2, tmin_n1

            if op2 == 1 and np.random.rand() > 0.5: # random swap
                n1, n2 = n2, n1
                tmin_n1, tmin_n2 = tmin_n2, tmin_n1

            if op2 == 3: # swap
                n1, n2 = n2, n1
                tmin_n1, tmin_n2 = tmin_n2, tmin_n1


            # check how many periods will be left
            capacidade = Carga[n1] - np.floor(tmin_n1) * Carga[n1] / tmin_n1
            ng_final = np.ceil(capacidade / p[0])
            gruas_outro = 4 - ng_final
            vprint(3,j, n1, " -> ", capacidade, "   ", ng_final)

            # start placing cranes on ship n1
            ng = 0
            for g in G:
                if GN[n1, g] == 1:
                    if ng < 4 and Carga[n1] > 0:
                        GT[g, j] = 0   # crane is no longer available
                        zHeur[g, n1] = 1
                        Carga[n1] = max(0, Carga[n1] - p[g])  # the load decreases
                        if Carga[n1] == 0:
                            cH[n1] = j + 1
                        ng += 1  # One crane stops the ship
                        vprint(3,j, "  G ", g, " on ship ", n1, "-> carga ", Carga[n1], "  (ng ", ng, " )")

            # Place cranes on ship n2
            ng = 0
            for g in G:
                if GN[n2, g] == 1 and GT[g, j] == 1:
                    if ng < 4 and Carga[n2] > 0:
                        GT[g, j] = 0   # crane is no longer available
                        zHeur[g, n2] = 1
                        Carga[n2] = max(0, Carga[n2] - p[g])  # the load decreases
                        if Carga[n2] == 0:
                            cH[n2] = j + 1
                        ng += 1  # One crane stops the ship
                        vprint(3,j, "  G ",g," on ship ",n2, "-> carga ",Carga[n2],"  (ng ",ng," )")

            # Crane readjustment
            if gruas_outro > 0:
                for g in G:
                    if zHeur[g, n1] == 1 and GN[n2, g] == 1:
                        if sum(zHeur[gg, n2] for gg in G) < 4 and gruas_outro > 0:
                            zHeur[g, n1] = 0
                            zHeur[g, n2] = 1
                            Carga[n1] += p[g]
                            Carga[n2] -= p[g]
                            vprint(3,j, " AG ",g," em ",n1," (",Carga[n1],") Stops ",n2," (",Carga[n2],")")
                            gruas_outro -= 1

        vprint(3,j, "Tempo ", j, "  -> ", custo_atual)
        for k in V:
            Ant_Navio[k, j] = Navio[k]

    if sum(Carga.values()) > 0:
        if max(inst.T) < 100:
            log(f"  Not all ships done, expanding time {max(inst.T)+1}.\n")
            T=inst.T
            inst.T = range(min(inst.T), max(inst.T) + 2)
            custo_atual = bacasp_inner_heuristic(op1, op2, inst, first_stage_solution, ii, log)
            inst.T = T
            return custo_atual
        custo_atual = np.inf
    for k in V:
        vprint(3, k, " atraca em ", b[k], "  entre ", arr_time[k], " e ", cH[k])
    return custo_atual
