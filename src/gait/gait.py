
import pinocchio
import numpy as np
import numpy.typing as npt
from src.logger import logger
from scipy.special import comb

JOINT_ORDER = ["RightHip", "RightKnee", "LeftHip", "LeftKnee"]
C_THETA = np.array([1, 1, 1, 0, 0])

## Permutations
P_LEFT = np.array([0, 1, 2, 3, 4]) # Left Stance
P_RIGHT = np.array([0, 3, 4, 1, 2]) # Right Stance
SWAP = np.array([0, 3, 4, 1, 2]) # Stance-independent swap

"""
Builds the gait configuration vector q_gait
[Torso wrt. Vertical, StanceHip, StanceKnee, SwingHip, SwingKnee]
Where all angles are relative to their parent links (except q1).
"""

class Gait:
    def __init__(self, model: pinocchio.Model, 
                q_model: np.ndarray, init_stance: str, 
                seed_alpha: np.ndarray) -> None:
        """
        Args:
            model: Pinocchio model of robot.
            q_model: Model Configuration Vector q from Pinocchio. Also written as q_e
            init_stance: Initial stance of the robot.
            seed_alpha: The last column of Bezier Coefficients to initialize gait.
            seed_alpha: [StanceHip, StanceKnee, SwingHip, SwingKnee]

        Vector Forms:

            Gait Configuration Vector q_gait: [Torso wrt. Vertical, StanceHip, StanceKnee, SwingHip, SwingKnee]
            Model Configuration Vector q_model: [Torso, RightHip, RightKnee, LeftHip, LeftKnee]
        """
        assert init_stance == "R" or init_stance == "L", "FATAL ERROR: Invalid stance"

        self.model = model 
        self.data = model.createData()
        ## Store L1 and L2
        assert model.jointPlacements is not None

        knee_id = model.getJointId("LeftKnee")
        self.L1 = np.linalg.norm(model.jointPlacements[knee_id].translation)
        foot_frame_id = model.getFrameId("LeftFoot")
        assert model.frames is not None

        self.L2 = np.linalg.norm(model.frames[foot_frame_id].placement.translation)

        self.q_model = q_model 
        self.stance = init_stance

        ## Initialization 
        self.q_gait = self.build_gait_q()
        self.q_minus = self.initialize_q_minus(seed_alpha)
        ### \/ DOUBLE CHECK @ OPERATOR
        self.q_plus = self.q_minus[SWAP]

        knee_stance, knee_swing = seed_alpha[1], seed_alpha[3]

        self.alpha = np.zeros((4, 6)) # Alpha takes the shape (4, 6)
        self.alpha[:, -1] = seed_alpha
        self.alpha[:, 0] = self.q_plus[1:]

        assert abs(knee_stance) > 1e-3, "FATAL ERROR: stance knee seeded straight"   
        assert abs(knee_swing)  > 1e-3, "FATAL ERROR: swing knee seeded straight"
        assert np.sign(knee_stance) == np.sign(knee_swing), \
            "FATAL ERROR: knees bend opposite ways in seed"

        self.knee_sign = np.sign(knee_stance)

    def init_alpha(self):
        targets, tau = self.build_targets()
        return self.alpha_lstsq(tau, targets, self.alpha)

    def _reorder(self, q_gait: npt.NDArray[np.float64], stance: str) -> npt.NDArray[np.float64]:
        """
        Builds Pinocchio's configuration vector q from
        q_gait vector.

        q[0:1] = [0.0, 0.0] due to the URDF's floating base
        """
        t, sh, sk, wh, wk = q_gait

        return (np.array([0., 0., t, wh, wk, sh, sk]) if stance == "R"
            else np.array([0., 0., t, sh, sk, wh, wk]))
    def _get_stance_foot_id(self, stance: str) -> int:
        return self.model.getFrameId("RightFoot" if stance == "R" else "LeftFoot")

    def lift_q(self, q_gait: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Lifts q_gait (5 DOF) into q_model (7 DOF) by pinning stance foot at origin
        """
        q_e = np.zeros(7)
        q_e[2:] = self._reorder(q_gait, self.stance)
        pinocchio.framesForwardKinematics(self.model, self.data, q_e)
        p = self.data.oMf[self._get_stance_foot_id(self.stance)].translation # type: ignore
        q_e[0] -= p[0]
        q_e[1] -= p[1]
        pinocchio.framesForwardKinematics(self.model, self.data, q_e)
        return q_e

    def lift_qdot(self, q_e: npt.NDArray[np.float64], qdot_gait: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Lifts qdot_gait (5 DOF) into q_model (7 DOF) by pinning stance foot at origin
        through J_st @ qdot == 0
        """
        st_id = self._get_stance_foot_id(self.stance)

        pinocchio.computeJointJacobians(self.model, self.data, q_e)
        ## Extracts J_st for the stance leg elements
        J_st = pinocchio.getFrameJacobian(self.model, self.data, st_id,
                                        pinocchio.LOCAL_WORLD_ALIGNED)[[0, 2], :]
        qdot_e = np.zeros(7)
        qdot_e[2:] = self._reorder(qdot_gait, self.stance)
        qdot_e[:2] = -J_st[:, 2:] @ qdot_e[2:]

        assert np.linalg.norm(J_st @ qdot_e) < 1e-12

        return qdot_e

    def build_gait_q(self) -> npt.NDArray[np.float64]:
        assert self.model.joints is not None, "FATAL ERROR: Invalid model"
        assert self.stance == "L" or self.stance == "R", "FATAL ERROR: Invalid stance"

        try:
            base_pitch_y_id = self.model.getJointId("base_pitch_y")
            vertical_idx = self.model.joints[base_pitch_y_id].idx_q
            vertical_q = self.q_model[vertical_idx] # q1

            joint_idx = [
                self.model.joints[self.model.getJointId(name)].idx_q
                for name in JOINT_ORDER
            ]

            joints_q = self.q_model[joint_idx]
            q_arranged = np.concatenate([[vertical_q], joints_q])
            ## Permutate relative to stance to obtain q_gait form
            q_gait = q_arranged[P_LEFT if self.stance == "L" else P_RIGHT]

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

    def initialize_q_minus(self, seed_alpha: np.ndarray) -> npt.NDArray[np.float64]:
        """
        Initializes q_minus from seed_alpha by building q_torso
        through forward kinematics by solving the constraint for q_torso (q_t)
        using the equation:
        L2​cos(qt​+sh)+L1​cos(qt​+sh+sk)-L2​cos(qt​+wh)-L1​cos(qt​+wh+wk)=0
        """
        sh, sk, wh, wk = seed_alpha

        C = self.L1 * np.cos(sh) + self.L2 * np.cos(sh+sk) \
            - self.L1 * np.cos(wh) - self.L2 * np.cos(wh+wk)
        
        S = self.L1 * np.sin(sh) + self.L2 * np.sin(sh+sk) \
            - self.L1 * np.sin(wh) - self.L2 * np.sin(wh+wk)
        qt = np.arctan2(C, S)
        z_hip = self.L1 * np.cos(qt+sh) + self.L2 * np.cos(qt+sh+sk)
        if z_hip < 0:
            qt -= np.sign(qt) * np.pi
        
        return np.concatenate([[qt], seed_alpha])

    def theta(self, q_gait: npt.NDArray[np.float64]) -> float:
        """
        Builds the phase variable theta:
    
        theta(q) = cT * q
    
        where c voids swing leg angles.
        """
        return float(q_gait @ C_THETA)
        """
        Pins the stance foot to the origin by making stance foot velocity = 0 through
        `J_st @ q_dot == 0`
        """
        st_id = self.model.getFrameId("RightFoot" if stance == "R" else "LeftFoot")

        pinocchio.computeJointJacobians(self.model, self.data, q_model)
        ## Extracts J_st for the stance leg elements
        J_st = pinocchio.getFrameJacobian(self.model, self.data, st_id,
                                        pinocchio.LOCAL_WORLD_ALIGNED)[[0, 2], :]
        qdot_e = np.zeros(7)
        qdot_e[2:] = qdot_model
        qdot_e[:2] = -J_st[:, 2:] @ qdot_e[2:]

        assert np.linalg.norm(J_st @ qdot_e) < 1e-12

        return qdot_e

    def get_swing_jacobian(self, q_model: npt.NDArray[np.float64], stance) -> npt.NDArray[np.float64]:
        sw_id = self.model.getFrameId("LeftFoot"  if stance == "R" else "RightFoot")
        pinocchio.computeJointJacobians(self.model, self.data, q_model)
        J_sw = pinocchio.getFrameJacobian(self.model, self.data, sw_id,
                                    pinocchio.LOCAL_WORLD_ALIGNED)[[0, 2], :]
        assert np.linalg.matrix_rank(J_sw) == 2
        return J_sw

    def get_D(self, q_model: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Args:
            q_model: current configuration based off of Pinocchio's model.
        Returns:
            Symmetrical inertia matrix D evaluated at q_model.
        """
        D = pinocchio.crba(self.model, self.data, q_model).copy()
        return np.triu(D) + np.triu(D, 1).T

    def impact_map(self, q_minus, qdot_minus):
        q_e = self.lift_q(q_minus)
        qdot_e = self.lift_qdot(q_e, qdot_minus)

        D = self.get_D(q_e)
        J_sw = self.get_swing_jacobian(q_e, self.stance)

        A = np.block([[D, J_sw.T],
                      [J_sw, np.zeros((2, 2))]])
        b = np.concatenate([D @ qdot_e, np.zeros(2)])

        sol = np.linalg.solve(A, b)
        qdot_e_plus = sol[:7]
        F_ext = sol[7:] # impulse Ns



    def fk(self, q_gait: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the forward kinematics for hip positions wrt. the stance foot
        as the origin.
        """
        qt, sh, sk, wh, wk = q_gait # not to be confused with self.q_gait

        """
        Returns the position vector from the foot to the hip given h (hip) and knee (h)
        positions
        """
        def hip_to_foot(h, k) -> np.ndarray:
            a1, a2 = qt + h, qt + h + k 
            return np.array([-self.L1 * np.sin(a1) - self.L2* np.sin(a2), 
                            -self.L1 * np.cos(a1) - self.L2 * np.cos(a2)])

        hip = -hip_to_foot(sh, sk)
        sw_p = hip + hip_to_foot(wh, wk)

        return hip, sw_p

    def ik(self, hip: np.ndarray, foot: np.ndarray, qt, knee_sign):
        """
        Computes the inverse kinematics for the relative hip and knee angle
        given hip and foot positions
        """
        d = foot - hip 
        r = np.linalg.norm(d)
        assert r < self.L1 + self.L2 - 1e-6, "FATAL ERROR: impossible leg length"
        # c = (r^2 - L1^2 - L2^2) / (2 * L1*  L2)
        c = np.clip((r**2 - self.L1**2 - self.L2**2) / (2*self.L1*self.L2), -1.0, 1.0)
        qk = knee_sign * np.arccos(c)
        gamma = np.arctan2(-d[0], -d[1]) # the angle of the leg relative to the vertical
        beta = np.arctan2(self.L2 * np.sin(qk), self.L1 + self.L2 * np.cos(qk))

        return gamma - beta - qt, qk
    
    def bernstein(self, tau, M=5):
        """
        Calculates the Bernstein polynomial from tau

        B(s) = (M; k) * s^k * (1-s)^(M-k)
        """
        tau = np.atleast_1d(tau)[:, None]
        k = np.arange(M+1)

        return comb(M,k) * tau**k * (1-tau)**(M-k)

    def build_targets(self, N=15, clearance=0.10) -> tuple[np.ndarray, np.ndarray]:
        """
        Builds the target trajectory Y in task space by linear interpolation 
        between the start and end points of seed_alpha (start and end assumed to be the same)
        """
        hip_p, sw_p = self.fk(q_gait=self.q_plus)
        hip_m, sw_m = self.fk(q_gait=self.q_minus)
        qt_p, qt_m = self.q_plus[0], self.q_minus[0]
        Y = np.zeros((N, 4)) 
        theta_samples = np.zeros(N)
        for i, u in enumerate(np.linspace(0.0, 1.0, N)):
            hip  = (1-u)*hip_p + u*hip_m
            qt   = (1-u)*qt_p  + u*qt_m
            foot = (1-u)*sw_p  + u*sw_m
            foot[1] += 4*clearance*u*(1-u)          # swing arc, zero at both ends
            sh, sk = self.ik(hip, np.zeros(2), qt, self.knee_sign)
            wh, wk = self.ik(hip, foot, qt, self.knee_sign)
            q_gait = np.array([qt, sh, sk, wh, wk])
            theta_samples[i] = self.theta(q_gait)
            Y[i] = [sh, sk, wh, wk]

        tau = self.calculate_tau(theta_samples, self.q_plus, self.q_minus)

        return Y, tau

    def alpha_lstsq(self, tau: npt.NDArray[np.float64], 
                        target: npt.NDArray[np.float64], alpha: npt.NDArray[np.float64]):
        """
        Uses least square fit to approximate the Bezier Curve by solving
        Y (target) = B_fixed + B_free (unkown)
        """
        B = self.bernstein(tau)
        fixed = [0, 5] # temporary placeholder for impact map invariance
        R = target - B[:, fixed] @ alpha[:, fixed].T
        sol, res, rank, sv = np.linalg.lstsq(B[:, 1:5], R, rcond=None)
        alpha[:, 1:5] = sol.T
        logger.info(f"Rank: {rank} ")
        return alpha

    def bezier(self, alpha: npt.NDArray[np.float64], tau: float,):
        """
        Calculates the M degree Bezier Polynomial given control points alpha
        and input tau using:
        
        sum from k=0 to k=M: alpha * (M; k) tau^k  (1-tau)^(M-k)
        """

        M = alpha.shape[1] - 1 # 5
        B = np.array([comb(M, k) * tau**k * (1 - tau)**(M-k) for k in range(M+1)])
        return alpha @ B

    def calculate_tau(self, theta_samples: np.ndarray, q_plus: np.ndarray, q_minus: np.ndarray):
        
        
        """
        tau function parameterizes state 

        tau = (theta_gait - theta_plus) / (theta_minus - theta_plus)

        """
        theta_plus = self.theta(q_plus)
        theta_minus = self.theta(q_minus)

        dtheta = theta_minus - theta_plus
        assert abs(dtheta) > 1e-6, "FATAL ERROR: No gait progression generated"

        return (theta_samples - theta_plus) / dtheta

    def sweep(self, model: pinocchio.Model, data: pinocchio.Data,
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
        sw_id = model.getFrameId("LeftFoot"  if stance == "R" else "RightFoot")
        taus = np.linspace(0., 1., N)
        sw_P, st_P, thetas, states = [], [], [], []
        for tau in taus:
            b = self.bezier(alpha, tau)
            theta = theta_p + tau * (theta_m - theta_p) # rearranged for theta
            qt = theta - b[0] - b[1] # calculates torso angle 
            q_gait = np.concatenate(([qt], b))
            q = self._reorder(q_gait, stance) 
            pinocchio.framesForwardKinematics(model, data, q)
            st_p = data.oMf[st_id].translation.copy()
            ## Anchors stance foot to the origin
            q[0] -= st_p[0]
            q[1] -= st_p[2]
            ## FK on the new coordinates
            pinocchio.framesForwardKinematics(self.model, data, q)
            sw_P.append(data.oMf[sw_id].translation.copy())
            st_P.append(data.oMf[st_id].translation.copy())
            thetas.append(self.theta(q_gait))
            states.append(q.copy())

        return taus, np.array(sw_P), np.array(st_P), np.array(thetas), np.array(states)
