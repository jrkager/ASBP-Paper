import numpy as np
import gurobipy as gp


def fseconds(seconds: float, precision = 2):
  return f"{seconds:.{precision}f}s"

class VerbosityManager:
    """
    utility class that stores the global verbosity values so that they can be used by every other file
    """
    # from verbosity 0 (don't print anything but input-dialogs) to verbosity 4 (max level of verbosity)
    global_verbosity = -1

def vprint(min_verbosity=None, *args, **kwargs):
    """
    utility function that works like print() but takes an additional first parameter which indicates the minimum
    level of verbosity for printing this. if verbosity is less than min_verbosity nothing is printed.
    :param min_verbosity:
    :param args: just like standard function print
    :param kwargs: just like print
    :return:
    """
    v = VerbosityManager.global_verbosity
    if min_verbosity is None:
        print()
    if not isinstance(min_verbosity, int):
        print(min_verbosity, *args, **kwargs)
    if v >= min_verbosity:
        print(*args, **kwargs)
    if min_verbosity == 99: # 99 is for testing purposes, if you just want to print one specific thing
        print("99: ", *args, **kwargs)

def vvprint(min_verbosity=None, max_verbosity=None, *args, **kwargs):
    """
    allows to specify a max_verbosity also (excluding).
    :param min_verbosity:
    :param max_verbosity:
    :param args:
    :param kwargs:
    :return:
    """
    v = VerbosityManager.global_verbosity
    if max_verbosity is None:
        if min_verbosity is None:
            print()
        else:
            vprint(min_verbosity)
    if not isinstance(max_verbosity, int):
        if not isinstance(min_verbosity, int):
            print(min_verbosity, max_verbosity, *args, **kwargs)
        else:
            vprint(min_verbosity, max_verbosity, *args, **kwargs)
    if v >= min_verbosity and v < max_verbosity:
        print(*args, **kwargs)


def roundBinaryValue(var):
    if type(var) == np.ndarray:
        ret = var.astype(int)
        ret[var < 0.5] = 0
        ret[var >= 0.5] = 1
        return ret
    return 0 if var < 0.5 else 1


def roundIntegerValue(var):
  if type(var) == np.ndarray:
    ret = var.round().astype(int)
    return ret
  return round(var) # is an integer

def roundBinaryVariable(var):
    if type(var) == gp.Var:
      return roundBinaryValue(var.X)
    return roundBinaryValue(var)

def getValueArray(gurobi_vars, roundBinaries = False):
    if not type(gurobi_vars) is gp.tupledict:
        if type(gurobi_vars) is gp.Var:
            if roundBinaries and gurobi_vars.vtype == 'B':
                return roundBinaryValue(gurobi_vars.X)
            return gurobi_vars.X
        raise Exception("gurobi_vars is neither gp.tupledict nor gp.Var")

    # array will be indexable with the indices from gurobi_vars. might have unused positions
    if type(gurobi_vars.keys()[0]) != tuple:
        array_size = max(gurobi_vars) + 1
    else:
        array_size = tuple(d+1 for d in max(gurobi_vars.keys()))
    if roundBinaries and gurobi_vars.values()[0].vtype == 'B':
        result_array = np.full(array_size, np.nan, dtype = int)
    else:
        roundBinaries = False
        result_array = np.zeros(array_size, dtype=float)
    for key, value in gurobi_vars.items():
        if roundBinaries:
            result_array[key] = roundBinaryValue(value.X)
        else:
            result_array[key] = value.X
    return result_array


def getValueDict(gurobi_vars, roundBinaries=False, roundInteger=False):
  if not type(gurobi_vars) is gp.tupledict:
    if type(gurobi_vars) is gp.Var:
      if roundBinaries and gurobi_vars.vtype == 'B':
        return roundBinaryValue(gurobi_vars.X)
      elif roundInteger and gurobi_vars.vtype == 'I':
        return roundIntegerValue(gurobi_vars.X)
      return gurobi_vars.X
    raise Exception("gurobi_vars is neither gp.tupledict nor gp.Var")

  # dict will be indexable with the keys from gurobi_vars
  ret = {}
  if roundBinaries:
    ret = {k : roundBinaryValue(gurobi_vars[k].X) for k in gurobi_vars}
  elif roundInteger:
    ret = {k: roundIntegerValue(gurobi_vars[k].X) for k in gurobi_vars}
  else:
    ret = {k : gurobi_vars[k].X for k in gurobi_vars}
  return ret
