
import numpy as np
import numpy.typing as npt
from src.logger import logger
from scipy.special import comb


def bernstein(tau, M: int = 5):
    """
    Calculates the Bernstein polynomial from tau

    B(s) = (M; k) * s^k * (1-s)^(M-k)
    """
    tau = np.atleast_1d(tau)[:, None]
    k = np.arange(M+1)

    return comb(M, k) * tau**k * (1-tau)**(M-k)


def bezier(alpha: npt.NDArray[np.float64], tau: float):
    """
    Calculates the M degree Bezier Polynomial given control points alpha
    and input tau using:

    sum from k=0 to k=M: alpha * (M; k) tau^k  (1-tau)^(M-k)
    """
    M = alpha.shape[1] - 1  # 5
    B = np.array([comb(M, k) * tau**k * (1 - tau)**(M-k) for k in range(M+1)])
    return alpha @ B


def bezier_derivative(alpha: npt.NDArray[np.float64], tau: float):
    """
    Calculates the derivative of M degree Bezier Polynomial given control points alpha
    and input tau using:

    M * sum from k=0 to M-1: (alpha_k+1 - alpha_k) * comb(M-1;k) * tau^k * (1-tau)^(M-1-k)
    """
    M = alpha.shape[1] - 1  # 5
    B = np.array([(alpha[k+1] - alpha[k]) * comb(M-1, k) * tau**k * (1 - tau)**(M-1-k) for k in range(M-1)])
    return M @ B


def alpha_lstsq(tau: npt.NDArray[np.float64],
                target: npt.NDArray[np.float64], alpha: npt.NDArray[np.float64]):
    """
    Uses least square fit to approximate the Bezier Curve by solving
    Y (target) = B_fixed + B_free (unkown)
    """
    B = bernstein(tau)
    fixed = [0, 5]  # temporary placeholder for impact map invariance
    R = target - B[:, fixed] @ alpha[:, fixed].T
    sol, res, rank, sv = np.linalg.lstsq(B[:, 1:5], R, rcond=None)
    alpha[:, 1:5] = sol.T
    logger.info(f"Rank: {rank} ")
    return alpha
