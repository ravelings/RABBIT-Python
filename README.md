# RABBIT in Python (WIP)
RABBIT is a 5 linked planar biped developed by [Chevallereau et al.](https://ieeexplore.ieee.org/document/1234651). This repository is my recreation of its dynamical modeling and controller in modern Python by following the book [Feedback Control of Dynamic Bipedal Locomotion](https://web.eecs.umich.edu/~grizzle/biped_book_web/)

This project is part of a larger project of recreating RABBIT using 3D printed components. Major milestones will be posted in the `README` in order of recency.

Please bear in mind that as a college Sophomore working on this project, there could be mathematical mistakes that I do not notice as I am trying my best to learn the topic and implementation at the same time.

## The Impact Map. September 10th, 2026

### Assumptions
1. Impact results from the contact of the swing leg end with the ground.
2. Impact is **instantaneous**
3. Impact results in **no rebound or slipping** of the swing leg
4. At impact, stance leg lifts from the ground without **interaction**
5. External forces during impact can be represented by impulses
6. Actuators cannot generate impulses, so ignored.
7. Impulsive force may result in instantaneous change in **velocity**, but **not** configuration

### The Impact Map
I will denote `q_gait` as $q_s$ and `q_model` (7 DOF) as $q_e$.

The goal of the impact map $\Delta$ is to map pre-impact state $x^-$ onto the post-impact state $x^+$. Or more in other terms $\Delta(q_e^-) \mapsto q_e^+$: 

It is also worth knowing that since impact does not change instantaneous configuration (assumption 7), then $q_e^+ = q_e^-$.
### Equation 1: Swing Foot Velocity
Let $p_2(q_e)$ map $q_e$ into $(x, z)$ swing foot positions such that $\mathbb{R}^7 \mapsto \mathbb{R}^2$. And let the swing foot Jacobian $E_2(q_e) := \frac{\partial}{\partial q_e}p_2 (q_e)$. From assumption 3, we know that the foot **does not slip** or **rebound**, therefore our post-impact velocity must be equal to zero:

$$
E_2(q_e^-)\dot{q_e}^+ = 0
$$

### Equation 2: External Forces
We will start with the 7 DOF model of the robot, then we will integrate to obtain the external force vector $F_\text{ext}$ as a change in momentum. It turns out that, at the instant of impact, the bounded terms $C$, $G$ and our actuator torque $Bu$ become zero, and we are left with:

$$
D_e(q_e^+)\dot{q_e}^+ - D_e(q_e^-)\dot{q_e}^- = F_\text{ext}
$$

If we define $F_2$ as the forces acting on the swing foot, then it also follows that:

$$
F_\text{ext} = E_2^{\top}(q_e^-)F_2
$$

### Stacked Matrix
Now we have two equation with the same quantity, we can stack them in a matrix, resulting in a KKT system, and what I would call the **impact equation:**

$$
\begin{bmatrix}
D_e(q_e^-) && -E_2^{\top}(q_e^-) \\
E_2(q_e^-) && 0_{2\times2}
\end{bmatrix}
\begin{bmatrix}
\dot{q_e}^+ \\ F_2
\end{bmatrix} =
\begin{bmatrix}
D_e(q_e^-)\dot{q_e}^- \\ 0_{2 \times 1}
\end{bmatrix}
$$

This equation can solved using `np.linalg.solve` demonstrated in the following code:
```python
A = np.block([[D, -J_sw.T],
			  [J_sw, np.zeros((2, 2))]])
b = np.concatenate([D @ qdot_e, np.zeros(2)])

sol = np.linalg.solve(A, b)
```

### Obtaining $\alpha_1$ 
#### Defining the Coordinate Inverse
Recall the output $y$:

$$
y = h_0(q) - h_d \odot \theta(q)
$$

Where $h_0$ is our actual output and $h_d$ is our desired output.

And that $h_0(q) := H_0 \ q$ and $\theta(q) := c \ q$

Now, according to our hypothesis that 1: the output only depends on the configuration variables and two: coordinate transformations are invertible. These two hypotheses are apparently only satisfied if $H := [H_0 \ ; \ c]$ is full rank. Then, the coordinate inverse is given by:

$$
q = H^{-1} \begin{bmatrix} h_d(\theta) \\ \theta \end{bmatrix}
$$

Recall that $h_d(\theta)$ is defined in terms of our Bézier coefficients $\alpha$, then our pre-impact output $h_d(\theta^+) = \alpha_0$ and post-impact $h_d(\theta^-) = \alpha_M$ where $M$ is the degree of our polynomial (5 in our case).
#### Differentiating and substituting
If we differentiate $q$ in terms of the Bézier coefficients, we get the following:
at $\dot{y} = 0$ and $s = 0$:

$$
\dot{q}^+ = H^{-1} \left[ \frac{M}{\theta^- - \theta^+}(\alpha_1 - \alpha_0) \right] \ \dot{\theta}^+
$$

(Eq 6.16a)
And:

$$
\dot{q}^- = H^{-1} \left[ \frac{M}{\theta^- - \theta^+}(\alpha_M - \alpha_{M-1}) \right] \ \dot{\theta}^-
$$

(Eq 6.16b)

Rearranging for $\alpha_1$:

$$
\alpha_1 = \alpha_0 + \frac{\theta^- - \theta^+}{M} \frac{H \dot{q}^+}{\dot{\theta}^+}
$$

Notice that to obtain $\dot{q}^+$, we need to use the impact map, but also notice that the impact map requires us to supply a velocity $\dot{q}^-$, which we don't have at the beginning of optimization. A trick to obtain an accurate answer is to use the ratio:

$$
v^- = \frac{\dot{q}^-}{\dot{\theta}^-}
$$

Notice that $c\dot{q}^- / \dot{\theta}^- = 1$. We are normalizing the velocity by $\dot{\theta}^-$. We can obtain this ratio by rearranging (Eq 6.16b).

Recall $\Delta$ maps $q^- \mapsto q^+$, then by linearity, $\Delta(v^-) = \frac{\dot{q}^+}{\dot{\theta}^-}$. Another way to look at this is treating $1 / \dot{\theta}^-$  as a scalar multiplied to the impact equation. For convenience, we will define $\omega^+ := \Delta(v^-)$.
And it becomes clear that:

$$
\dot{q}^+ = \omega^+ \dot{\theta}^-
$$

And we define another term:

$$
\nu:= c \ \omega^+ = \dot{\theta}^+
$$

(Note that this is not $v$, but the greek letter nu $\nu$)

Finally, substituting for $\dot{q}^+$ and $\dot{\theta}^+$ into the equation for $\alpha_1$, we obtain:

$$
\alpha_1 = \alpha_0 + \frac{\theta^- - \theta^+}{M\nu} H_0 \ \omega^+
$$

This equation in code form is as follows:
```python
A = M * (alphas[:, -1] - alphas[:, -2])
B = theta_m - theta_p 
assert abs(B) > 1e-6, "ERROR: Theta did not advance over step"

H0v = A / B
v = np.zeros(5)
## Rebuild torso: vt = 1 - vsh - vsk
v[1:] = H0v
v[0] = 1 - H0v[0] - H0v[1]

w_plus, _ = self.impact_map(q_minus, v)
nu = C_THETA @ w_plus

alphas[:, 1] = alphas[:, 0] + ( B / (M * nu) ) * H_0 @ w_plus # set alpha 1
```


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
