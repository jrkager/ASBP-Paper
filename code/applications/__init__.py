from dataclasses import dataclass
from typing import Any
from .optimization_model import OptimizationModel

class SecondStageModelType(OptimizationModel):
    pass

@dataclass
class Application:
    inst: Any = None
    MasterModel: OptimizationModel = None
    SecondStageModel: SecondStageModelType = None


@dataclass
class InstanceStrings:
    ALG_INTRO_TEXT: str = None
    UNIQUE_IDENTIFIER: str = None