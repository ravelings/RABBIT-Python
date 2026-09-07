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

seed_alpha = np.array([0.5043, -0.500, -0.0043, -0.500])
seed_alpha_flipped = np.array([0.0043, 0.500, -0.5043, 0.500])   # was [0.5043, -0.500, -0.0043, -0.500] -> flipped knee
q = pinocchio.randomConfiguration(pin_model)
data = pin_model.createData()

gait = Gait(model=pin_model, q_model=q, init_stance="R",
            seed_alpha=seed_alpha_flipped)

mjcf_path = src_dir / "mjcf" / "rabbit.xml"

mj_model = mj.MjModel.from_xml_path(str(mjcf_path))

states_R, sw_R = get_and_validate_gait(gait, pin_data, "R")
states_L, sw_L = get_and_validate_gait(gait, pin_data, "L")

def mj_simulate_no_physics(model: mj.MjModel, states_R: npt.NDArray[np.float64], sw_R: npt.NDArray[np.float64], 
                        states_L: npt.NDArray[np.float64], sw_L: npt.NDArray[np.float64], fps: float = 60.0, steps=2):
    """
    (PHYSICS OFF)
    Simulates the states onto a matching model.

    Args:
        model: MjModel of the robot.
        states: Array of states to simulate the model over. Each row
            takes the form ``[b_x, b_z, b_y, sh, sk, wh, wk]``.
    """
    data = mj.MjData(model)
    ## Validate shapes
    joint_names = np.array([model.joint(i).name for i in range(model.njnt)])
    assert states_R.shape == states_L.shape
    assert joint_names.shape[-1] == states_R.shape[-1], "ERROR: Shape mismatch"

    ## Obtains the end positions for each step
    x_R = sw_R[-1, 0]
    x_L = sw_L[-1, 0]

    x_off = 0.0 # accumulator for distance travelled

    ## Simulation + Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera("track").id
        while viewer.is_running():
            for step in range(steps):
                states, x_dist = (states_R, x_R) if step % 2 == 0 else (states_L, x_L)
                for k in range(len(states_R)):
                    if not viewer.is_running():
                        break
                    data.qpos[:] = states[k]
                    data.qpos[0] += x_off
                    data.qvel[:] = 0.0
                    mj.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(1.0 / fps)

                x_off += x_dist # moves the x position to the total displacement for the next leg

mj_simulate_no_physics(model=mj_model, states_R=states_R, sw_R=sw_R, states_L=states_L, sw_L=sw_L, fps=120.0)



