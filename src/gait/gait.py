
import pinocchio
import numpy as np
import numpy.typing as npt

from src.gait import kinematics
from src.gait import bezier
from src.gait import hzd
from src.gait.gaitparams import GaitParams
from src.gait.robotmodel import RobotModel

"""
Builds the gait configuration vector q_gait
[StanceHip, StanceKnee, SwingHip, SwingKnee, Torso wrt. Vertical]
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
            Gait Configuration Vector q_gait: [StanceHip, StanceKnee, SwingHip, SwingKnee, Torso wrt. Vertical]
            Model (7DOF) Configuration Vector q_model: [Base_x, Base_z, Torso, LeftHip, LeftKnee, RightHip, RightKnee]
        """
        assert init_stance == "R" or init_stance == "L", "FATAL ERROR: Invalid stance"

        self.model = model
        self.data = model.createData()
        self.robot_model = RobotModel(self.model, self.data)
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
        self.q_gait = kinematics.build_gait_q(self.model, self.q_model, self.stance)
        q_minus = kinematics.initialize_q_minus(seed_alpha, self.L1, self.L2)
        q_plus = q_minus[kinematics.SWAP]
        theta_p = hzd.theta(q_plus)
        theta_m = hzd.theta(q_minus)

        knee_stance, knee_swing = seed_alpha[1], seed_alpha[3]

        assert abs(knee_stance) > 1e-3, "FATAL ERROR: stance knee seeded straight"
        assert abs(knee_swing)  > 1e-3, "FATAL ERROR: swing knee seeded straight"
        assert np.sign(knee_stance) == np.sign(knee_swing), \
            "FATAL ERROR: knees bend opposite ways in seed"

        self.knee_sign = np.sign(knee_stance)

        alpha = np.zeros((4, 6)) # Alpha takes the shape (4, 6)
        alpha[:, -1] = seed_alpha
        alpha[:, 0] = q_plus[:-1]

        targets, tau = hzd.build_targets(q_plus, q_minus, self.knee_sign, self.L1, self.L2)
        alpha = bezier.alpha_lstsq(tau, targets, alpha)
        alpha_new = hzd.get_alpha_1(self.robot_model, q_minus, alpha, theta_p, theta_m, self.stance)
        assert alpha.shape == alpha_new.shape

        ### Misc Init
        base_pitch_y_id = model.getJointId("base_pitch_y")
        assert model.joints is not None
        vertical_idx = model.joints[base_pitch_y_id].idx_q

        self.params = GaitParams(
            alpha=alpha,
            q_plus=q_plus,
            q_minus=q_minus,
            theta_plus=hzd.theta(q_plus),
            theta_minus=hzd.theta(q_minus),
            vertical_idx=vertical_idx
        )

    def poncare(self):
        V_zero, xi = hzd.V_zero(self.params, self.robot_model, self.stance)
        hzd.verify_stability(self.params, self.robot_model, V_zero, xi, self.stance)

    
