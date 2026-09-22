
import pinocchio
import numpy as np
import numpy.typing as npt

from src.gait import kinematics
from src.gait import bezier as bezier_curve

C_THETA = np.array([1, 1, 0, 0, 1])  # extracts the angle of the stance tibia.

N = 5      # dim(q_gait)
N_ACT = 4  # actuated dimensions

H_0 = np.eye(N)[:-1]
H = np.vstack([H_0, C_THETA])


def theta(q_gait: npt.NDArray[np.float64]) -> float:
    """
    Builds the phase variable theta:

    theta(q) = cT * q

    where c voids swing leg angles.
    """
    return float(q_gait @ C_THETA)


def calculate_tau(theta_samples: npt.NDArray[np.float64] | float,
        q_plus: npt.NDArray[np.float64], q_minus: npt.NDArray[np.float64]):
    """
    tau function parameterizes state

    tau = (theta_gait - theta_plus) / (theta_minus - theta_plus)
    """
    theta_plus = theta(q_plus)
    theta_minus = theta(q_minus)

    dtheta = theta_minus - theta_plus
    assert abs(dtheta) > 1e-6, "FATAL ERROR: No gait progression generated"

    return (theta_samples - theta_plus) / dtheta


def get_dh_dq(alpha: npt.NDArray[np.float64], theta_val: float,
        q_plus: npt.NDArray[np.float64], q_minus: npt.NDArray[np.float64]):
    tau = calculate_tau(theta_val, q_plus, q_minus)
    assert isinstance(tau, float)

    dtau_dtheta = 1 / (theta(q_minus) - theta(q_plus))
    dhd_dtheta = bezier_curve.bezier_derivative(alpha, tau)

    return H_0 - dhd_dtheta * dtau_dtheta


def get_kappa1(epsilon1: npt.NDArray[np.float64]):
    ### Partial Derivative at epsilon1
    dtheta_dq = C_THETA
    dh_dq = get_dh_dq()


def impact_map(model: pinocchio.Model, data: pinocchio.Data,
        q_minus: npt.NDArray[np.float64], qdot_minus: npt.NDArray[np.float64], stance: str):
    """
    Calculates the state after impact.

    Returns:
        qdot_plus: 5 DOF configuration velocity after impact
        F_ext: Vector of external forces acting on the swing leg at impact
    """
    q_e = kinematics.lift_q(model, data, q_minus, stance)
    qdot_e = kinematics.lift_qdot(model, data, q_e, qdot_minus, stance)

    D = kinematics.get_D(model, data, q_e)
    J_sw = kinematics.get_swing_jacobian(model, data, q_e, stance)

    A = np.block([[D, -J_sw.T],
                  [J_sw, np.zeros((2, 2))]])
    b = np.concatenate([D @ qdot_e, np.zeros(2)])

    sol = np.linalg.solve(A, b)
    qdot_e_plus = sol[:7]
    qdot_plus = kinematics.model_to_gait(qdot_e_plus, stance)
    F2 = sol[7:]  # impulse Ns

    F_ext = D @ (qdot_e_plus - qdot_e)
    assert np.allclose(F_ext, J_sw.T @ F2)  # Eq. (3.15) VS (3.18)

    return qdot_plus, F_ext


def get_alpha_1(model: pinocchio.Model, data: pinocchio.Data,
        q_minus: npt.NDArray[np.float64], alphas: npt.NDArray[np.float64],
        theta_p: float, theta_m: float, stance: str, M: int = 5) -> npt.NDArray[np.float64]:

    A = M * (alphas[:, -1] - alphas[:, -2])
    B = theta_m - theta_p
    assert abs(B) > 1e-6, "ERROR: Theta did not advance over step"

    H0v = A / B
    v = np.zeros(5)
    ## Rebuild torso: vt = 1 - vsh - vsk
    v[:4] = H0v
    v[4] = 1 - H0v[0] - H0v[1]

    w_plus, _ = impact_map(model, data, q_minus, v, stance)
    nu = C_THETA @ w_plus

    alphas[:, 1] = alphas[:, 0] + (B / (M * nu)) * H_0 @ w_plus

    return alphas


def build_targets(q_plus: npt.NDArray[np.float64], q_minus: npt.NDArray[np.float64],
        knee_sign, L1: float, L2: float,
        N: int = 15, clearance: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds the target trajectory Y in task space by linear interpolation
    between the start and end points of seed_alpha (start and end assumed to be the same)
    """
    hip_p, sw_p = kinematics.fk(q_plus, L1, L2)
    hip_m, sw_m = kinematics.fk(q_minus, L1, L2)
    qt_p, qt_m = q_plus[-1], q_minus[-1]
    Y = np.zeros((N, 4))
    theta_samples = np.zeros(N)
    for i, u in enumerate(np.linspace(0.0, 1.0, N)):
        hip = (1-u)*hip_p + u*hip_m
        qt = (1-u)*qt_p + u*qt_m
        foot = (1-u)*sw_p + u*sw_m
        foot[1] += 4*clearance*u*(1-u)  # swing arc, zero at both ends
        sh, sk = kinematics.ik(hip, np.zeros(2), qt, knee_sign, L1, L2)
        wh, wk = kinematics.ik(hip, foot, qt, knee_sign, L1, L2)
        q_gait = np.array([sh, sk, wh, wk, qt])
        theta_samples[i] = theta(q_gait)
        Y[i] = [sh, sk, wh, wk]

    tau = calculate_tau(theta_samples, q_plus, q_minus)

    return Y, tau


def sweep(model: pinocchio.Model, data: pinocchio.Data,
        alpha: npt.NDArray[np.float64], theta_p: float,
        theta_m: float, stance: str, N=201):
    """
    Sweeps the Bezier trajectory with control points alpha and
    start/end stance to obtain data on the foot height over
    N timesteps tau

    Returns:
        list(tau): timesteps
        list(st_p): array of stance foot heights over tau
        list(sw_p): array of swing foot heights over tau
        list(theta): array of actuated angles over tau
    """
    assert data.oMf is not None

    st_id = model.getFrameId("RightFoot" if stance == "R" else "LeftFoot")
    sw_id = model.getFrameId("LeftFoot" if stance == "R" else "RightFoot")
    taus = np.linspace(0., 1., N)
    sw_P, st_P, thetas, states = [], [], [], []
    for tau in taus:
        b = bezier_curve.bezier(alpha, tau)
        th = theta_p + tau * (theta_m - theta_p)  # rearranged for theta
        qt = th - b[0] - b[1]  # calculates torso angle
        q_gait = np.concatenate((b, [qt]))
        q = kinematics.reorder(q_gait, stance)
        pinocchio.framesForwardKinematics(model, data, q)
        st_p = data.oMf[st_id].translation.copy()
        ## Anchors stance foot to the origin
        q[0] -= st_p[0]
        q[1] -= st_p[2]
        ## FK on the new coordinates
        pinocchio.framesForwardKinematics(model, data, q)
        sw_P.append(data.oMf[sw_id].translation.copy())
        st_P.append(data.oMf[st_id].translation.copy())
        thetas.append(theta(q_gait))
        states.append(q.copy())

    return taus, np.array(sw_P), np.array(st_P), np.array(thetas), np.array(states)
