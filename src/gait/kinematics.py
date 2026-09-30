
import pinocchio
import numpy as np
import numpy.typing as npt
from src.logger import logger

from src.gait.robotmodel import RobotModel

JOINT_ORDER = ["RightHip", "RightKnee", "LeftHip", "LeftKnee"]

## Permutations
P_LEFT = np.array([2, 3, 0, 1, 4])   # Left Stance
P_RIGHT = np.array([0, 1, 2, 3, 4])  # Right Stance
SWAP = np.array([2, 3, 0, 1, 4])     # Stance-independent swap

"""
Builds the gait configuration vector q_gait
[StanceHip, StanceKnee, SwingHip, SwingKnee, Torso wrt. Vertical]
Where all angles are relative to their parent links (except q1).
"""


def unpack_q_gait(q_gait: npt.NDArray[np.float64]):
    """
    Unpacks q_gait into its named components.
    Centralizes the index order so only this needs to change if the
    q_gait format changes again.
    Returns:
        sh, sk, wh, wk, qt
    """
    sh, sk, wh, wk, qt = q_gait
    return sh, sk, wh, wk, qt


def model_to_gait(q_model: npt.NDArray[np.float64], stance: str):
    """
    Converts extended configuration (7 DOF) vector `q_model` into the
    gait configuration vector (5 DOF) `q_gait`
    """
    _, _, qt, lh, lk, rh, rk = q_model
    q_gait = np.array([rh, rk, lh, lk, qt])

    return q_gait[P_LEFT if stance == "L" else P_RIGHT]


def reorder(q_gait: npt.NDArray[np.float64], stance: str) -> npt.NDArray[np.float64]:
    """
    Builds Pinocchio's configuration vector q from
    q_gait vector.

    q[0:1] = [0.0, 0.0] due to the URDF's floating base
    """
    sh, sk, wh, wk, t = unpack_q_gait(q_gait)

    return (np.array([0., 0., t, wh, wk, sh, sk, ]) if stance == "R"
        else np.array([0., 0., t, sh, sk, wh, wk]))


def get_stance_foot_id(model: pinocchio.Model, stance: str) -> int:
    return model.getFrameId("RightFoot" if stance == "R" else "LeftFoot")


def lift_q(r: RobotModel,
        q_gait: npt.NDArray[np.float64], stance: str) -> npt.NDArray[np.float64]:
    """
    Lifts q_gait (5 DOF) into q_model (7 DOF) by pinning stance foot at origin
    """
    model = r.model
    data = r.data

    q_e = np.zeros(7)
    q_e[2:] = reorder(q_gait, stance)[2:]
    pinocchio.framesForwardKinematics(model, data, q_e)

    p = data.oMf[get_stance_foot_id(model, stance)].translation  # type: ignore
    q_e[0] -= p[0]
    q_e[1] -= p[2]
    pinocchio.framesForwardKinematics(model, data, q_e)
    return q_e


def lift_qdot(r: RobotModel,
        q_e: npt.NDArray[np.float64], qdot_gait: npt.NDArray[np.float64],
        stance: str) -> npt.NDArray[np.float64]:
    """
    Lifts qdot_gait (5 DOF) into q_model (7 DOF) by pinning stance foot at origin
    through J_st @ qdot == 0
    """
    model = r.model 
    data = r.data
    st_id = get_stance_foot_id(model, stance)

    pinocchio.computeJointJacobians(model, data, q_e)
    ## Extracts J_st for the stance leg elements
    J_st = pinocchio.getFrameJacobian(model, data, st_id,
                                    pinocchio.LOCAL_WORLD_ALIGNED)[[0, 2], :]
    qdot_e = np.zeros(7)
    qdot_e[2:] = reorder(qdot_gait, stance)[2:]
    qdot_e[:2] = -J_st[:, 2:] @ qdot_e[2:]

    assert np.linalg.norm(J_st @ qdot_e) < 1e-12

    return qdot_e


def build_gait_q(model: pinocchio.Model, q_model: npt.NDArray[np.float64], stance: str) -> npt.NDArray[np.float64]:
    assert model.joints is not None, "FATAL ERROR: Invalid model"
    assert stance == "L" or stance == "R", "FATAL ERROR: Invalid stance"

    try:
        base_pitch_y_id = model.getJointId("base_pitch_y")
        vertical_idx = model.joints[base_pitch_y_id].idx_q
        vertical_q = q_model[vertical_idx]  # q1

        joint_idx = [
            model.joints[model.getJointId(name)].idx_q
            for name in JOINT_ORDER
        ]

        joints_q = q_model[joint_idx]
        q_arranged = np.concatenate([joints_q, [vertical_q]])
        ## Permutate relative to stance to obtain q_gait form
        q_gait = q_arranged[P_LEFT if stance == "L" else P_RIGHT]

        return q_gait

    except KeyError as e:
        logger.info(f"FATAL ERROR: Joint Name mistmatch: {str(e)}")
        raise KeyError(f"FATAL ERROR: Joint Name mistmatch: {str(e)}")

    except IndexError as e:
        logger.info(f"FATAL ERROR: {str(e)}")
        raise IndexError(f"FATAL ERROR: q vector shape mismatch: {str(e)}")
    except Exception as e:
        logger.info(f"Warning: {str(e)}")
        raise Exception(f"Unkown error: {str(e)}")


def initialize_q_minus(seed_alpha: np.ndarray, L1: float, L2: float) -> npt.NDArray[np.float64]:
    """
    Initializes q_minus from seed_alpha by building q_torso
    through forward kinematics by solving the constraint for q_torso (q_t)
    using the equation:
    L2​cos(qt​+sh)+L1​cos(qt​+sh+sk)-L2​cos(qt​+wh)-L1​cos(qt​+wh+wk)=0
    """
    sh, sk, wh, wk = seed_alpha

    C = L1 * np.cos(sh) + L2 * np.cos(sh+sk) \
        - L1 * np.cos(wh) - L2 * np.cos(wh+wk)

    S = L1 * np.sin(sh) + L2 * np.sin(sh+sk) \
        - L1 * np.sin(wh) - L2 * np.sin(wh+wk)
    qt = np.arctan2(C, S)
    z_hip = L1 * np.cos(qt+sh) + L2 * np.cos(qt+sh+sk)
    if z_hip < 0:
        qt -= np.sign(qt) * np.pi

    return np.concatenate([seed_alpha, [qt]])


def get_swing_jacobian(r: RobotModel, q_model: npt.NDArray[np.float64], stance: str) -> npt.NDArray[np.float64]:
    model = r.model
    data = r.data   
    sw_id = model.getFrameId("LeftFoot" if stance == "R" else "RightFoot")
    pinocchio.computeJointJacobians(model, data, q_model)
    J_sw = pinocchio.getFrameJacobian(model, data, sw_id,
                                pinocchio.LOCAL_WORLD_ALIGNED)[[0, 2], :]
    assert np.linalg.matrix_rank(J_sw) == 2
    return J_sw

def get_stance_jacobian(r: RobotModel, q_model: npt.NDArray[np.float64], stance: str) -> npt.NDArray[np.float64]:
    model = r.model
    data = r.data
    st_id = model.getFrameId("RightFoot" if stance == "R" else "LeftFoot")
    pinocchio.computeJointJacobians(model, data, q_model)
    J_st = pinocchio.getFrameJacobian(model, data, st_id,
                                pinocchio.LOCAL_WORLD_ALIGNED)[[0, 2], :]
    assert np.linalg.matrix_rank(J_st) == 2
    return J_st


def get_D(r: RobotModel,
        q_model: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Args:
        q_model: current configuration based off of Pinocchio's model.
    Returns:
        Symmetrical inertia matrix D evaluated at q_model.
    """
    D = pinocchio.crba(r.model, r.data, q_model).copy()
    return np.triu(D) + np.triu(D, 1).T


def fk(q_gait: np.ndarray, L1: float, L2: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the forward kinematics for hip positions wrt. the stance foot
    as the origin.
    """
    sh, sk, wh, wk, qt = unpack_q_gait(q_gait)

    def hip_to_foot(h, k) -> np.ndarray:
        """
        Returns the position vector from the foot to the hip given h (hip) and knee (h)
        positions
        """
        a1, a2 = qt + h, qt + h + k
        return np.array([-L1 * np.sin(a1) - L2 * np.sin(a2),
                        -L1 * np.cos(a1) - L2 * np.cos(a2)])

    hip = -hip_to_foot(sh, sk)
    sw_p = hip + hip_to_foot(wh, wk)

    return hip, sw_p


def ik(hip: np.ndarray, foot: np.ndarray, qt, knee_sign, L1: float, L2: float):
    """
    Computes the inverse kinematics for the relative hip and knee angle
    given hip and foot positions
    """
    d = foot - hip
    r = np.linalg.norm(d)
    assert r < L1 + L2 - 1e-6, "FATAL ERROR: impossible leg length"
    # c = (r^2 - L1^2 - L2^2) / (2 * L1*  L2)
    c = np.clip((r**2 - L1**2 - L2**2) / (2*L1*L2), -1.0, 1.0)
    qk = knee_sign * np.arccos(c)
    gamma = np.arctan2(-d[0], -d[1])  # the angle of the leg relative to the vertical
    beta = np.arctan2(L2 * np.sin(qk), L1 + L2 * np.cos(qk))

    return gamma - beta - qt, qk
