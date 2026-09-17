import mujoco as mj
import mujoco.viewer
import numpy as np
import pinocchio
import numpy.typing as npt
import time
from pathlib import Path

from src.testing.gait_test import get_and_validate_gait
from src.testing.mj_pin_test import pinocchio_extractor
from src.gait.gait import Gait

src_dir = Path(__file__).resolve().parent.parent

urdf_path = src_dir / "urdf" / "rabbit_floating.urdf"
pin_model = pinocchio_extractor(urdf_path)
assert pin_model is not None 
pin_data = pin_model.createData()
q = pinocchio.randomConfiguration(pin_model)
data = pin_model.createData()

seed_alpha_flipped = np.array([0.0043, 0.500, -0.5043, 0.500])

gait = Gait(model=pin_model, q_model=q, init_stance="R",
            seed_alpha=seed_alpha_flipped)

"""
Lift consistency check: PASS (Sept 8, 26)
"""
q_e = gait.lift_q(gait.q_minus)
qdot_minus = np.array([-0.15, 1.65, -1.00, 0.65, 0.50]) # reference qdot_minus for testing ONLY.

pinocchio.forwardKinematics(pin_model, data, q_e)
pinocchio.updateFramePlacements(pin_model, data)
stance_frameId = pin_model.getFrameId("RightFoot")
assert (data.oMf[stance_frameId].translation[2]) < 1e-12 # type: ignore
J_st = gait.get_swing_jacobian(q_e, "R")
assert np.linalg.norm(J_st @ gait.lift_qdot(q_e, qdot_minus))

"""
Relabelling Consistency
"""

