
import pinocchio
import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid

from src.gait import kinematics
from src.gait import bezier as bezier_curve
from src.gait import dynamics
from src.gait.gaitparams import GaitParams
from src.gait.robotmodel import RobotModel

C_THETA = np.array([1, 1, 0, 0, 1])  # extracts the angle of the stance tibia.

N = 5      # dim(q_gait)
N_ACT = 4  # actuated dimensions

H_0 = np.eye(N)[:-1]
H = np.vstack([H_0, C_THETA])

## qr = PI_* @ q_gait, where qr = [Torso, RightHip, RightKnee, LeftHip, LeftKnee]
## and q_gait = [StanceHip, StanceKnee, SwingHip, SwingKnee, Torso] (see unpack_q_gait)
PI_R = np.eye(N)[[4, 0, 1, 2, 3]]  # Right stance: stance legs -> Right slots
PI_L = np.eye(N)[[4, 2, 3, 0, 1]]  # Left stance: stance legs -> Left slots
H_INV = np.linalg.inv(H)


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

def q_on_Z(p: GaitParams, theta: float):
    """
    Extracts q from ξ₁ from Westervelt et al., with ξ₁ ≡ θ evaluated on Z.

    q = H⁻¹[hd(ξ₁); ξ₁]
    """
    tau = calculate_tau(theta, p.q_plus, p.q_minus) 
    assert isinstance(tau, float)

    hd = bezier_curve.bezier(p.alpha, tau)

    return H_INV @ np.append(hd, theta)

def get_dh_dq(p: GaitParams, theta: float):
    """∂h/∂q (θ), shape (n-1, n)."""
    tau = calculate_tau(theta, p.q_plus, p.q_minus)
    assert isinstance(tau, float)
    dtheta_dq = C_THETA

    dtau_dtheta = 1 / (p.theta_minus - p.theta_plus)
    db_dtau = bezier_curve.bezier_derivative(p.alpha, tau) # equi to ∂hd/∂τ

    return H_0 - dtau_dtheta * np.outer(db_dtau, dtheta_dq)

def reduce_inertia(r: RobotModel, D: npt.NDArray[np.float64], q_s: npt.NDArray[np.float64], stance: str):
    """
    Reduces D_e (from Pinnochio) to the reduced gait model D_s

    Args:
        D_e: Extended Inertia Matrix
        q_s: Reduced Configuration Vector
    """
    q_e = kinematics.reorder(q_s, stance)
    J_r = kinematics.get_stance_jacobian(r, q_e, stance)[:, 2:] # [I2x2, J_r]
    PI = PI_R if stance == "R" else PI_L 
    T = np.vstack([-J_r, np.eye(J_r.shape[-1])]) @ PI

    return T.T @ D @ T

def get_gamma0(p: GaitParams, r: RobotModel, theta: float, stance: str):
    q_s = q_on_Z(p, theta)
    q_e = kinematics.reorder(q_s, stance)

    D = dynamics.get_D(r.model, r.data, q_e)
    D_s = reduce_inertia(r, D, q_s, stance)
    return D_s[-1]

def kappa1(p: GaitParams, r: RobotModel, theta: float,
           stance: str):
    """κ₁(ξ₁) from Westervelt et al., with ξ₁ ≡ θ evaluated on Z.

    κ₁ = ∂θ/∂q @ [∂h/∂q; γ₀(q)]⁻¹ @ [0; 1] | Z
    """
    ### Partial Derivative at theta
    dtheta_dq = C_THETA
    dh_dq = get_dh_dq(p, theta)
    
    gamma0 = get_gamma0(p, r, theta, stance)

    A = np.vstack([dh_dq, gamma0])
    B = np.eye(A.shape[0])[-1]

    return float(dtheta_dq @ np.linalg.solve(A, B))
    
def kappa2(p: GaitParams, r: RobotModel, theta: float,
        stance: str):
    """
    κ₂(ξ₁) from Westervelt et al., with ξ₁ ≡ θ evaluated on Z.

    κ₂ = ∂V/∂q_N = transpose(e_N) @ ∇V(q) | Z ,where ∇ V is the generalized gravity matrix
    """
    q = q_on_Z(p, theta)
    q_e = kinematics.lift_q(r, q, stance)
    G = pinocchio.computeGeneralizedGravity(r.model, r.data, q_e)

    return float(-G[p.vertical_idx])

def V_zero(p: GaitParams, r: RobotModel, stance: str, n_pts: int = 500):
    """
    V_zero over [θ⁺, θ⁻], from eq (5.70).
    """
    xi = np.linspace(p.theta_plus, p.theta_minus, n_pts)
    k1 = np.array([kappa1(p, r, theta, stance) for theta in xi])
    k2 = np.array([kappa2(p, r, theta, stance) for theta in xi])

    if np.any(np.abs(k1) < 1e-8):
        raise ValueError("κ₁ vanishes on [θ⁺, θ⁻]: zero dynamics singular")

    V = -cumulative_trapezoid(k2 / k1, xi, initial=0.0)
    assert isinstance(V, np.ndarray)

    return V, xi

def get_q_0(p: GaitParams, state: str):
    """
    Obtains q_0⁻ by Hq_0⁻ = [a_M; θ⁻]
    Args:
        state: "plus" or "minus"
    """
    theta = p.theta_minus if state == "minus" else p.theta_plus
    B = np.append(p.alpha[:, -1], p.theta_minus)

    return np.linalg.solve(H, B)

def get_delta0(p: GaitParams, r: RobotModel, stance: str) -> float:
    """
    Calculates δ₀ = γ₀(q₀+) @ ∆(q₀-)λqdot
    """
    q_minus = get_q_0(p, "minus")
    theta_minus = theta(q_minus)
    dh_dq = get_dh_dq(p, theta_minus)
    gamma0_minus = get_gamma0(p, r, theta_minus, stance)
    A = np.vstack([dh_dq, gamma0_minus])
    lam_q = np.linalg.solve(A, np.eye(A.shape[0])[:, -1])

    theta_plus = theta(get_q_0(p, "plus"))
    gamma_plus = get_gamma0(p, r, theta_plus, stance)
    delta_q_lam_q, _ = impact_map(r, q_minus, lam_q, stance)

    return gamma_plus @ delta_q_lam_q

def verify_stability(p: GaitParams, r: RobotModel, V: npt.NDArray[np.float64], xi: npt.NDArray[np.float64], stance: str):
    delta0_2 = (get_delta0(p, r, stance))** 2

    print(f" Delta0 = {delta0_2:.4f}")
    print(f"Condition 1: 0< delta0^2 < 1: {delta0_2 > 1e-8 and delta0_2 < 1.00}")

    sum = ( (delta0_2) / (1 - delta0_2) ) * V[-1] + V.max()

    print(f" sum = {sum:.4f}")
    print(f"Nontrivial Periodic Orbit: sum < 0: {sum < 1e-8}")
          
def impact_map(r: RobotModel ,
        q_minus: npt.NDArray[np.float64], qdot_minus: npt.NDArray[np.float64], stance: str):
    """
    Calculates the state after impact.

    Returns:
        qdot_plus: 5 DOF configuration velocity after impact
        F_ext: Vector of external forces acting on the swing leg at impact
    """
    model = r.model 
    data = r.data
    q_e = kinematics.lift_q(r, q_minus, stance)
    qdot_e = kinematics.lift_qdot(r, q_e, qdot_minus, stance)

    D = kinematics.get_D(r, q_e)
    J_sw = kinematics.get_swing_jacobian(r, q_e, stance)

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

def get_alpha_1(r: RobotModel,
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

    w_plus, _ = impact_map(r, q_minus, v, stance)
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

    assert isinstance(tau, np.ndarray)

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
