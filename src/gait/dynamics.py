
import pinocchio
import numpy as np
import numpy.typing as npt


def get_D(model: pinocchio.Model, data: pinocchio.Data,
        q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Args:
        q: current configuration based off of Pinocchio's model.
    Returns:
        Symmetrical inertia (mass) matrix D evaluated at q.
    """
    D = pinocchio.crba(model, data, q).copy()
    return np.triu(D) + np.triu(D, 1).T


def get_G(model: pinocchio.Model, data: pinocchio.Data,
        q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Args:
        q: current configuration based off of Pinocchio's model.
    Returns:
        Generalized gravity vector G evaluated at q.
    """
    return pinocchio.computeGeneralizedGravity(model, data, q).copy()
