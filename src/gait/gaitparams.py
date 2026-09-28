from dataclasses import dataclass, replace
import numpy as np

@dataclass(frozen=True)
class GaitParams:
    alpha: np.ndarray

    q_plus: np.ndarray
    q_minus: np.ndarray

    theta_plus: float
    theta_minus: float

    vertical_idx: int

    def __post_init__(self):
        self.alpha.setflags(write=False)
        self.q_plus.setflags(write=False)
        self.q_minus.setflags(write=False)