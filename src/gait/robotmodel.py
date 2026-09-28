from dataclasses import dataclass
import numpy as np

import pinocchio

@dataclass
class RobotModel:
    model: pinocchio.Model
    data: pinocchio.Data # this data is edited across each pinnochio call


