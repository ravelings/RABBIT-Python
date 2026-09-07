import mujoco
import pinocchio
import numpy as np

from pathlib import Path
from src.extractor import pinocchio_extractor

JOINT_NAMES = ["base_slide_x", "base_slide_z", "base_pitch_y",
               "LeftHip", "LeftKnee", "RightHip", "RightKnee"]

def pin_quantities(q: np.ndarray, model: pinocchio.Model, data: pinocchio.Data) -> dict:
    pinocchio.centerOfMass(model, data, q)
    pinocchio.framesForwardKinematics(model, data, q)
    lf = data.oMf[model.getFrameId("LeftFoot")].translation.copy() # type: ignore
    rf = data.oMf[model.getFrameId("RightFoot")].translation.copy() # type: ignore
    M = pinocchio.crba(model, data, q)
    store_dict = {}
    store_dict["torso_com"] = data.com[0] # type: ignore
    store_dict["lf"] = lf
    store_dict["rf"] = rf 
    store_dict["M"] = np.diag(M).copy()

    return store_dict

def mj_quantities(q: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    assert data.qpos.shape == q.shape 

    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    com = data.subtree_com[model.body("Torso").id].copy()
    lf = data.site("LeftFoot").xpos.copy()
    rf = data.site("RightFoot").xpos.copy()
    full_M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full_M)

    store_dict = {}
    store_dict["torso_com"] = com # type: ignore
    store_dict["lf"] = lf
    store_dict["rf"] = rf 
    store_dict["M"] = np.diag(full_M).copy()

    return store_dict

src_dir = Path(__file__).resolve().parent.parent
urdf_path = src_dir / "urdf" / "rabbit_floating.urdf"

pin_model = pinocchio_extractor(urdf_path)
assert pin_model is not None 
pin_data = pin_model.createData()

mjcf_path = src_dir / "mjcf" / "rabbit.xml"

mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
mj_data = mujoco.MjData(mj_model)

pin_idx = [
    pin_model.joints[pin_model.getJointId(n)].idx_q # type: ignore
    for n in JOINT_NAMES
]

mj_idx = [
    mj_model.joint(n).qposadr[0] for n in JOINT_NAMES
]

