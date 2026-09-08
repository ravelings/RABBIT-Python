# RABBIT in Python (WIP)
RABBIT is a 5 linked planar biped developed by [Chevallereau et al.](https://ieeexplore.ieee.org/document/1234651). This repository is my recreation of its dynamical modelling and controller in modern Python.

This project is part of a larger project of recreating RABBIT using 3D printed components. Major milestones will be posted in the `README` in order of recency.

## Bézier curve generation through `seed_alpha`. September 4th, 2026
Alpha is a (4X7) matrix containing control points of the Bézier curve for each actuated joint.

To "kickstart" the alpha NLP optimization (later), we would need to provide it with initial values. Additionally, I wanted to see what a basic Bézier curve gait would look like.

To achieve this, I seeded the last column of alpha, as well as the first column by permutating the seed by swapping stance and swing leg angles. 
Then, linear interpolation with a quadratic correction on the swing foot is used to approximate 15 columns of target angles $Y$ over the trajectory. Where $Y$ takes the form 
`[stance_hip; stance_knee; swing_hip; swing_knee]`

Given a quantity $x$ and a slider $u \in [0, 1]$, our interpolated value $p$ at $u$ is:

$$
p(u) = (1-u)x + ux
$$

The quadratic correction applied onto the swing foot with a safety clearance scalar $c$ is given by:

$$
p_\text{sw} (u) = 4cu(1-u)
$$

After $Y$ is obtained, we need to obtain the entire $\alpha$ in order to generate a gait. Now given that the Bezier curve is linear in relation
to its control point $b$ (proof is left as exercise to reader). So, given $B_i$ as the Bernstein polynomial's output and control point $\alpha_i$ at phase $\tau_i$ (note this is all actuated joint),
then our Bezier curve evaluated at $\tau_i$ is simply: $B_i \ \alpha^{\top}_i$. This will give us the configuration of all actuated joints. Hence, we get the optimization problem:

$$
\min_\alpha Y - B \ \alpha^\top
$$

The following gait generation is generated with 200 timesteps of Bézier curve outputs between the starting stance and ending stance.
We then make MuJoCo follow the generated gait at 120FPS:

<img width="400" height="500" alt="python_xpFIYBkT8a" src="https://github.com/user-attachments/assets/02bc8289-6383-400c-950f-d951049f6125" />

Note that the gait generated is purely for testing purposes as it contains no real dynamics involved.
