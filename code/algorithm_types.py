from dataclasses import dataclass, field

from enum import Enum

class ChooseNextTypes(Enum):
    # these options are not implemented at the moment (HIGHEST_UB is always the standard)
    HIGHEST_UB = 1
    HIGHEST_GAP = 2
    HIGHEST_LB = 3

class InterruptConsequence(Enum):
    CHOOSE_AGAIN = 1
    BREAK_ITER = 2
    TERMINATE = 3

class AlgorithmTypes(Enum):
    # Interrupt reasons
    OTHER_CHOICE = 1
    OPTIMAL = 2
    ONLY_ONE = 3
    WORST_TIMELIMITS_REACHED = 4
    ONLY_D = 5
    D_OPT_IS_WORST = 6
    TIMEOUT_EXC = 7

    # Algorithm OPTIONS
    FIX_TIMELIMIT = 8
    VAR_TIMELIMIT = 9
    NO_Z_BOUND = 10
    NO_LOWER_BOUND = 11
    GLOBAL_GAP = 12 # stop even when <= z + pi, and not just when <= z, per default turned on

    # Choose other Algorithm than ours
    TOENISSEN = 13
    RODRIGUES = 14

    # Extra
    NONE = 15

    def consequence(self) -> InterruptConsequence:
        if self in [self.OTHER_CHOICE, self.NONE,
                    self.FIX_TIMELIMIT, self.VAR_TIMELIMIT, self.OPTIMAL]:
            return InterruptConsequence.CHOOSE_AGAIN
        if self in [self.ONLY_ONE, self.WORST_TIMELIMITS_REACHED]:
            return InterruptConsequence.BREAK_ITER
        if self in [self.ONLY_D, self.D_OPT_IS_WORST, self.TIMEOUT_EXC]:
            return InterruptConsequence.TERMINATE
        return None

    def str_reason(self, end='\n'):
        if self.consequence() == InterruptConsequence.CHOOSE_AGAIN:
            return f"Choose again. Reason: {self.name}" + end
        if self.consequence() == InterruptConsequence.BREAK_ITER:
            return f"Break iteration. Reason: {self.name}" + end
        if self.consequence() == InterruptConsequence.TERMINATE:
            return f"Terminate algorithm. Reason: {self.name}" + end


@dataclass
class AlgorithmOptions:
    VAR_TIMELIMIT_FACTOR: float = None
    VAR_TIMELIMIT_MINIMUM: float = None
    FIX_TIMELIMIT: float = None

    def __str__(self):
        ret = ""
        for k, v in self.__dict__.items():
            if v is not None:
                ret += str(k) + " = " + str(v)
                ret += ", "
        if len(ret):
            ret = ret[:-2]
        return ret


@dataclass
class AlgorithmType:
    choose_next: list[ChooseNextTypes] = field(default_factory=list)
    alg_types: list[AlgorithmTypes] = None
    options: AlgorithmOptions = field(default_factory=AlgorithmOptions)

    def is_set(self, criteria):
        if criteria in self.alg_types:
            return True
        return False

    def __str__(self):
        return (f"[{','.join(cn.name for cn in self.choose_next)}], "
                f"[{','.join(cn.name for cn in self.alg_types)}], "
                f"{self.options}")