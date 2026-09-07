from src.gait.gait import Gait 
import numpy as np
from pathlib import Path
from src.extractor import pinocchio_extractor
import pinocchio

import matplotlib.pyplot as plt

src_dir = Path(__file__).resolve().parent.parent
urdf_path = src_dir / "urdf" / "rabbit_floating.urdf"
pin_model = pinocchio_extractor(urdf_path)
assert pin_model is not None 
pin_data = pin_model.createData()

"""
seed_alpha = np.array([0.5043, -0.500, -0.0043, -0.500])
seed_alpha_flipped = np.array([0.0043, 0.500, -0.5043, 0.500])   # was [0.5043, -0.500, -0.0043, -0.500] -> flipped knee
q = pinocchio.randomConfiguration(pin_model)
data = pin_model.createData()

gait = Gait(model=pin_model, q_model=q, init_stance="R",
            seed_alpha=seed_alpha_flipped)
"""

def get_and_validate_gait(gait: Gait, data, stance) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:

    alpha = gait.init_alpha()

    theta_p, theta_m = gait.theta(gait.q_plus), gait.theta(gait.q_minus)

    taus, sw_P, st_P, thetas, states = gait.sweep(pin_model, data, alpha,
                                    theta_p, theta_m, stance)

    print(f"State Shape: {states.shape}")

    """    
    assert np.allclose(st_P[:, 2], 0., atol=1e-12), "anchor broken"
    print("endpoint b(0) err:", np.abs(gait.bezier(alpha, 0.) - gait.q_plus[1:]).max())
    print("endpoint b(1) err:", np.abs(gait.bezier(alpha, 1.) - gait.q_minus[1:]).max())
    print("swing z endpoints:", sw_P[0, 2], sw_P[-1, 2], " min interior:", sw_P[1:-1, 2].min())
    print("step length:", sw_P[-1, 0] - sw_P[0, 0])

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    ax[0].plot(taus, sw_P[:, 2]); ax[0].axhline(0, c='k', lw=.5); ax[0].set(xlabel="τ", ylabel="swing z [m]")
    ax[1].plot(sw_P[:, 0], sw_P[:, 2]); ax[1].axhline(0, c='k', lw=.5); ax[1].set(xlabel="x [m]", ylabel="z [m]"); ax[1].axis('equal')
    ax[2].plot(taus, thetas); ax[2].set(xlabel="τ", ylabel="θ(q)")
    plt.tight_layout(); plt.show()
    """

    return states, sw_P