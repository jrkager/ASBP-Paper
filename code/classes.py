import typing
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from applications import Application


class SecondStagePlaceholder:
    def __init__(self, time_over = False, accTime = 0, accProctime = None):
        self._time_over = time_over
        self.status = None
        self.mipgap = float("inf")
        self._accRuntime = accTime # time spent solving mip
        self._accProctime = accTime if accProctime is None else accTime


class TimeoutException(Exception):  # Custom exception class
  def __init__(self, message = None, reached_gap = None):
    message = message or ""
    message += f" reached gap = {reached_gap*100:.2f}%"
    super().__init__(message)
    self.reached_gap = reached_gap


@dataclass
class AlgorithmStats:
    TIME_MASTER: float = 0
    TIME_MASTER_PROC: float = 0
    TIME_HEUR: float = 0
    TIME_HEUR_PROC: float = 0
    TIME_SS: float = 0
    TIME_SS_PROC: float = 0
    TIME_TOT: float = 0
    TIME_TOT_PROC: float = 0
    JUMPS: int = 0
    CCALLS: int = 0
    ITERATIONS: int = 0
    OPENED: int = np.nan
    COST: float = np.nan
    first_stage: any = None
    reached_gap = np.nan

    @staticmethod
    def InfStats():
        return AlgorithmStats(TIME_MASTER=np.inf,TIME_MASTER_PROC=np.inf,
                               TIME_HEUR=np.inf,TIME_HEUR_PROC=np.inf,
                               TIME_TOT=np.inf,TIME_TOT_PROC=np.inf,
                               TIME_SS=np.inf,TIME_SS_PROC=np.inf,
                               ITERATIONS=np.nan)


@dataclass
class AlgorithmParams:
    app: Application = None
    # one of these two should be passed. If None, start_sc = [] will be used
    start_sc: int = None
    start_fs: Any = None
    MASTER_P: float = None
    desired_gap: float = 0
    logfile: typing.TextIO = None
    n_threads: float = 0
    # sample_number: int = None # is now a param of instance
    data_name: str = None
    HEURTIMELIMIT: float = 0.1
    total_timelimit: float = np.inf # in seconds
    time_remaining: float = None # not used
    mastermodel_logfolder: str = None
    ssmodel_logfolder: str = None
    save_fs: bool = False
    log_heuristic_res: bool = False
    heuristic: Callable[[...], None] = None
    master_callback: Callable[[...], None] = None
