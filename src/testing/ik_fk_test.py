import numpy as np

L1 = 0.4
L2 = 0.4

q = np.array([0, 0.5043, -0.500, -0.0043, -0.500])
sign = -1.0


def fk(q_gait: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qt, sh, sk, wh, wk = q_gait # not to be confused with self.q_gait

    """
    Returns the position vector from the foot to the hip given h (hip) and knee (h)
    positions
    """
    def hip_to_foot(h, k) -> np.ndarray:
        a1, a2 = qt + h, qt + h + k 
        return np.array([-L1 * np.sin(a1) - L2* np.sin(a2), 
                        -L1 * np.cos(a1) - L2 * np.cos(a2)])

    hip = -hip_to_foot(sh, sk)
    sw_p = hip + hip_to_foot(wh, wk)

    return hip, sw_p

def ik(hip: np.ndarray, foot: np.ndarray, qt, knee_sign):
    d = foot - hip 
    r = np.linalg.norm(d)
    assert r < L1 + L2 - 1e-6, "FATAL ERROR: impossible leg length"
    # c = (r^2 - L1^2 - L2^2) / (2 * L1*  L2)
    c = np.clip((r**2 - L1**2 - L2**2) / (2*L1*L2), -1.0, 1.0)
    qk = knee_sign * np.arccos(c)
    gamma = np.arctan2(-d[0], -d[1]) # the angle of the leg relative to the vertical
    beta = np.arctan2(L2 * np.sin(qk), L1 + L2 * np.cos(qk))

    return gamma - beta - qt, qk

hip, sw_p = fk(q)

print(f"Hip: {hip}, Foot To Hip Distance: {np.linalg.norm(hip)}")
print(f"Swing_foot: {sw_p}")

sw_h, sw_k = ik(hip, sw_p, q[0], sign) 

print(f"Swing Hip Angle: {sw_h}, Actual: {q[3]}")
print(f"Swing Knee Angle: {sw_k}, Actual: {q[4]}")