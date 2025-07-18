# Nonlinear Trajectory Optimization - Lecture 3

## Getting started

### Review
- _Second order methods_ are important: We should always do this.
- Problem structure gave us a way to solve optimal control problems efficiently, with "shooting" or "sparsity".
- Direct optimal control is best if we aren't worried about real-time deployment.
- Recall the Newton's method with equality constraints, and the KKT system

### Goals
- We can go in two directions:
    1. second-order indirect (iLQR, DDP)
    2. second-order direct (Direct Collocation, Multiple Shooting).
- We will focus on second-order direct.

## I. Nonlinear Programming

- A nonlinear program is a cost, equality constraint, and inequality constraint,
$$\begin{align}
    \min_z &\quad f(z) \\
    \text{s.t.}&\quad c(z) = 0 \\
    &\quad d(z) \le 0
\end{align}$$

- Grab an off-the-shelf solver: IPOPT (free), SNOPT, KNITRO (commercial)

### SQP (Sequential Quadratic Programming)

- Take a second order Taylor expansion of the cost and linearize the constraints (locally about a _guess_):

$$\begin{align}
    \min_z &\quad J(z) + g \Delta z + \tfrac{1}{2} \Delta z^T H \Delta z \\
    \text{s.t.}&\quad c(z) + C \Delta z = 0 \\
    &\quad d(z) + D \Delta z \le 0
\end{align}$$

- Lagrangian, $\mathcal{L}(z, \lambda, \mu) = J(z) + \lambda^T c(z) + \mu^T d(x)$
- Gradient: $g = \frac{\partial \mathcal{L}}{\partial z}$, Hessian: $H = \frac{\partial^2 \mathcal{L}}{\partial z^2}$, Jacobians: $C = \frac{\partial c}{\partial z}$, $D = \frac{\partial d}{\partial z}$
- Solve a QP to compute the primal-dual search direction, 
$$\Delta s = \begin{bmatrix} 
        \Delta z \\ \Delta \lambda \\ \Delta \mu
    \end{bmatrix}$$

#### Comments
- Play lots of tricks to leverage _sparsity_.
- $\text{SQP} \subset \text{SCP}$: Sequential _Convex_ Programming puts constraints directly into the solver, without linearization.
- We don't actually need rollouts: _Direct Collocation_ maximally leverages this constraint structure.

### Splines

- Cubic splines,
$$\begin{align}
    x(t) &= a_0 + a_1 t + a_2 t^2 + a_3 t^3 \\
    \dot{x}(t) &= \phantom{a_0 + } a_1 + 2 a_2 t + 3 a_3 t^2
\end{align}$$

- _Hermite splines_ use the left and right endpoint ($x_n = x(t)$, $x_{n+1}=x(t + h)$), and their derivatives,

$$\begin{align}
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        1 & h & h^2 & h^3 \\
        0 & 1 & 2h & 3 h^2
    \end{bmatrix}
    \begin{bmatrix} a_0 \\ a_1 \\ a_2 \\ a_3 \end{bmatrix}
    &= 
    \begin{bmatrix} x_n \\ \dot{x}_n \\ x_{n+1} \\ \dot{x}_{n+1} \end{bmatrix} \\
\Rightarrow
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        -\tfrac{3}{h^2} & -\tfrac{2}{h} & \tfrac{3}{h^2} & -\tfrac{1}{h} \\
        \tfrac{2}{h^3} & \tfrac{1}{h^2} & -\tfrac{2}{h^3} & \tfrac{1}{h^2}
    \end{bmatrix}
    \begin{bmatrix} x_n \\ \dot{x}_n \\ x_{n+1} \\ \dot{x}_{n+1} \end{bmatrix}
    &= 
    \begin{bmatrix} a_0 \\ a_1 \\ a_2 \\ a_3 \end{bmatrix}
\end{align}$$

- The _collocation point_ enforces the dynamics constraint using the value at some other point (the spline required free variables!),
$$\begin{align}
    x(t + \tfrac{h}{2}) &= \tfrac{1}{2}(x_n + x_{n+1}) + \tfrac{h}{8} (\dot{x}_n - \dot{x}_{n+1}) \\
    &= \tfrac{1}{2}(x_n + x_{n+1}) + \tfrac{h}{8} (f(x_n, u_n) - f(x_{n+1}, u_{n+1}))\\
    \, \\
    u(t + \tfrac{h}{2}) &= \tfrac{1}{2} (u_n + u_{n+1}) \\
    \, \\
    \dot{x}(t + \tfrac{h}{2}) &= -\tfrac{3}{2h}(x_n - x_{n+1}) - \tfrac{1}{4} (\dot{x}_n + \dot{x}_{n+1}) \\
     &= -\tfrac{3}{2h}(x_n - x_{n+1}) - \tfrac{1}{4} (f(x_n, u_n) + f(x_{n+1}, u_{n+1})) 
\end{align}$$

- Putting the dynamics into the constraint,
$$\begin{align}
    C_n(z) &= C_n(x_n, u_n, x_{n+1}, u_{n+1}) \\
    &= f(x_{n + 1/2}, u_{n + 1/2}) - \dot{x}_{n + 1/2} \\
    &= f\left( \tfrac{1}{2}(x_n + x_{n+1}) + \tfrac{h}{8} (f(x_n, u_n) - f(x_{n+1}, u_{n+1})), \tfrac{1}{2} (u_n + u_{n+1}) \right) \\&\quad- \left(-\tfrac{3}{2h}(x_n - x_{n+1}) - \tfrac{1}{4} (f(x_n, u_n) + f(x_{n+1}, u_{n+1})) \right) 
    \\&\quad \overset{!}{=} 0
\end{align}$$

- Achieves 3rd order accuracy (RK4 is Runge-Kutta 4th Order).

- Requires fewer $f$ calls than RK3. Exercise: How?

## II. IPOPT

From the IPOPT documentation (https://coin-or.github.io/Ipopt/OUTPUT.html):

- **inf_pr**: The unscaled constraint violation at the current point. This quantity is the infinity-norm (max) of the (unscaled) constraints ($g_L≤g(x)≤g_U$ in (NLP)). During the restoration phase, this value remains the constraint violation of the original problem at the current point. The option inf_pr_output can be used to switch to the printing of a different quantity.

- **inf_du**: The scaled dual infeasibility at the current point. This quantity measure the infinity-norm (max) of the internal dual infeasibility, Eq. (4a) in the implementation paper [12], including inequality constraints reformulated using slack variables and problem scaling. During the restoration phase, this is the value of the dual infeasibility for the restoration phase problem.

- **lg(mu)**: log10 of the value of the barrier parameter μ.

- **||d||**: The infinity norm (max) of the primal step (for the original variables x and the internal slack variables s). During the restoration phase, this value includes the values of additional variables, p and n (see Eq. (30) in [12]).

- **lg(rg)**: log10 of the value of the regularization term for the Hessian of the Lagrangian in the augmented system (δw in Eq. (26) and Section 3.1 in [12]). A dash ("-") indicates that no regularization was done.

- **alpha_du**: The stepsize for the dual variables (αzk in Eq. (14c) in [12]).