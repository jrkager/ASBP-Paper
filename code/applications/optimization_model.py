import gurobipy as gp
from gurobipy import GRB
import time

class InfeasibleException(Exception):
    pass

class OptimizationModel(gp.Model):
    def __init__(self, *argc, **argv):
        self._vars = dict()
        self._wasReset = False
        self._accRuntime = 0
        self._accProctime = 0
        super().__init__()
        self.Params.LogToConsole = 0
        for k in argv:
          if hasattr(self.Params, k):
            setattr(self.Params, k, argv[k])

    def optimize(self, *argc, **argv):
      start_proc = time.process_time()
      super().optimize(*argc, **argv)
      end_proc = time.process_time()
      self._accRuntime += self.runtime
      self._accProctime += end_proc - start_proc

    def _trackVars(self, name, var):
        if name:
            self._vars[name] = var
        else:
            i = 0
            while "var"+str(i) in self._vars:
                i += 1
            self._vars["var"+str(i)] = var
            print("tracked as", "var"+str(i))
        return var

    def addVar(self, *argc, **argv):
        return self._trackVars(argv.get("name"), super().addVar(*argc, **argv))

    def addVars(self, *argc, **argv):
        return self._trackVars(argv.get("name"), super().addVars(*argc, **argv))

    def run(self, callback = None, print_status = False, force_rerun = False):
        if self.status == GRB.LOADED or force_rerun or self._wasReset:
            self.optimize(callback = callback)
            self._wasReset = False
        if print_status:
          self.printStatus(raiseExceptions=True)

    def printStatus(self, raiseExceptions=False):
        if self.status == GRB.LOADED:
            print("Model loaded")
        if self.status == GRB.OPTIMAL:
            print("Model optimal")
        if self.status == GRB.INFEASIBLE:
            print("Model is infeasible")
            if raiseExceptions: raise InfeasibleException()
        if self.status == GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded")
            if raiseExceptions: raise InfeasibleException("Model is infeasible or unbounded")
        if self.status == GRB.UNBOUNDED:
            print("Model is unbounded (and maybe infeasible)")
            if raiseExceptions: raise InfeasibleException("Model is unbounded (and maybe infeasible)")

    def getOptimum(self, callback=None, print_status = False, force_rerun = False):
        self.run(callback=callback, print_status=print_status, force_rerun=force_rerun)
        return self.objVal

    def resetParams(self):
        self._wasReset = True
        temp_OutputFlag = self.Params.OutputFlag
        super().resetParams()
        self.Params.OutputFlag = temp_OutputFlag

    def resetParamsButLogAndThreads(self):
        logf = self.Params.LogFile
        logtc = self.Params.LogToConsole
        logflag = self.Params.OutputFlag
        n_threads = self.Params.Threads
        super().resetParams()
        self.Params.LogFile = logf
        self.Params.LogToConsole = logtc
        self.Params.OutputFlag = logflag
        self.Params.Threads = n_threads

    def printMIPGap(self):
        print("{:.1f}%".format(100 * self.MIPGap))
        return self.MIPGap

    def getSolutionVars(self):
        self.run()
        return self._vars

    def hasRun(self):
      return self.status != GRB.LOADED
