# Trajectory Optimization - Lecture 2

```@example trajopt
using Optim
using LinearAlgebra
using SparseArrays
using CairoMakie
using Piccolo

# let's define a shorthand for kron
const ⊗ = kron
```

## Getting started

### Review
- Gradient descent
- Newton's method and KKT conditions
- Regularization
- Newton approximations
- Line search

### Goals
- Introduce trajectory optimization
- Solve the LQR problem three ways
- Describe nonlinear trajectory optimization

## I. Trajectory optimization

- The solution is a definite _trajectory_, not a feedback policy.

$$\begin{align}
\min_{x_{1:N}, u_{1:N}} &\quad J(x_{1:N}, u_{1:N}) = \sum_{n=1}^N \ell_n(x_n, u_n) + \ell_N(x_N, u_N) \\
\text{s.t.} &\quad x_{n+1} = f(x_n, u_n, n)
\end{align}$$

### Named Trajectories

**Terminology**

- _Snapshot matrix_, $Z = \begin{bmatrix} | & | & & | \\ z_1 & z_2 & & z_T \\ | & | & & | \end{bmatrix}$
- _Knot point_, $z_1 = \begin{bmatrix} x_1 \\ u_1 \end{bmatrix}$

```@example trajopt
N = 10 # Number of knot points
traj = rand(NamedTrajectory, N)
```

```@example trajopt
traj.x
```

```@example trajopt
traj.u
```

```@example trajopt
plot(traj, :x)
```

## II. Linear Quadratic Regulator

- LQR is the "simple harmonic oscillator" of control

$$\begin{align}
\min_{x_{1:N}, u_{1:N{-}1}} &\quad J = \sum_{n=1}^{N-1} \tfrac{1}{2} x_n^T Q_n x_n + \tfrac{1}{2} u_n^T R_n u_n + \tfrac{1}{2} x_N^T Q_N x_N \\
\text{s.t.} &\quad x_{n+1} = A_n x_n + B_n u_n \\
&\quad Q_n \succeq 0,\, R_n \succ 0
\end{align}$$

- Quick check: Why does $R$ need to be positive definite?

### Linear systems

#### Zero-order hold

- Zero-order hold can be used to convert continuous, linear, time-invariant (LTI) systems to discrete LTI systems.

$$\begin{align}
\dot{x}(t) &= A x(t) + B u(t)  \\
\overset{h}{\longrightarrow} x(t+h) &= A_h x(t) + B_h u(t) \\
&= \left( \sum_{n\ge0} \tfrac{1}{n!} A^n h^n \right) x + \left( \sum_{n\ge1} \tfrac{1}{n!} A^{n-1} B h^n \right) u \\
&\approx (I + h A) x + h B u
\end{align}$$

- Matrix exponential trick:

$$\exp\left(\begin{bmatrix} A & B \\ 0 & 0 \end{bmatrix} h \right)
= \begin{bmatrix} A_h & B_h \\ 0 & I \end{bmatrix}$$

```@example trajopt
# Define continuous LTI system matrices
A = [0.0 1.0; -1.0 -0.1]
B = [0.0; 1.0]
h = 0.1  # Time step

function continuous_to_discrete(A, B, h)
    # Construct augmented matrix for matrix exponential
    augmented_matrix = [
        A B; 
        zeros(size(B, 2), size(A, 1)) zeros(size(B, 2), size(B, 2))
    ]

    # Compute matrix exponential
    exp_matrix = exp(augmented_matrix * h)

    # Extract discrete LTI system matrices
    A_h = exp_matrix[1:size(A, 1), 1:size(A, 2)]
    B_h = exp_matrix[1:size(A, 1), size(A, 2)+1:end]

    return A_h, B_h
end

# Extract discrete LTI system matrices
A_h, B_h = continuous_to_discrete(A, B, h);
```

```@example trajopt
A_h
```

```@example trajopt
I + A * h
```

```@example trajopt
B_h
```

```@example trajopt
B * h
```

#### Double Integrator

- Double integrator (Newton's second law, $F = ma$)

$$m \frac{d}{dt} \begin{bmatrix} q \\ \dot{q} \end{bmatrix} 
= \begin{bmatrix} 0 & m \\ 0 & 0 \end{bmatrix} \begin{bmatrix} q \\ \dot{q} \end{bmatrix}
+ \begin{bmatrix} 0 \\ u \end{bmatrix}$$

```@example trajopt
function double_integrator(m)
    A_c = [0.0 1.0; 0.0 0.0]
    B_c = [0.0; 1.0 / m]
    return A_c, B_c
end

# simulate a discrete LTI system
function simulate_dlti(u::AbstractMatrix, A, B, x1)
    N = size(u, 2) + 1
    x = zeros(size(A, 2), N)
    x[:, 1] = x1
    for k in 1:N-1
        x[:, k + 1] = A * x[:, k] + B * u[:, k]
    end
    return x
end

function simulate_dlti(u::AbstractVector, A, B, x1)
    simulate_dlti(reshape(u, 1, length(u)), A, B, x1)
end
```

```@example trajopt
m = 2.0 # Mass
A_c , B_c = double_integrator(m)
h = 0.05  # Time step
A, B = continuous_to_discrete(A_c, B_c, h);
```

```@example trajopt
A ≈ I + A_c * h
```

```@example trajopt
B ≈ B_c * h + [h^2 / 2m; 0]
```

### Indirect optimal control: Naive way

- Indirect optimal control is also known as "single shooting"

- The naive way is to perform gradient descent without any problem structure.

$$\min_{u_{1:N{-}1}} \quad J(u_{1:N{-}1}) = \sum_{n=1}^{N-1} \ell_n(x_n(u_{1:n{-}1}), u_n) + \ell_N(x_N(u_{1:N{-}1}), u_N)$$

- We will start with the double integrator and solve the LQR problem,
$$\begin{align}
\min_{u_{1:N{-}1}} &\quad J(u_{1:{N{-}1}}) = \sum_{n=1}^{N-1} \tfrac{1}{2} x_n(u_{1:n{-}1})^T Q_n x_n(u_{1:n{-}1}) + \tfrac{1}{2} u_n^T R_n u_n + \tfrac{1}{2} x_N(u_{1:N{-}1})^T Q_N x_N(u_{1:N{-}1}) \\
\text{s.t.} &\quad Q_n \succeq 0,\, R_n \succ 0
\end{align}$$

```@example trajopt
m = 0.1 # Mass
A_c, B_c = double_integrator(m)
h = 0.1  # Time step
A, B = continuous_to_discrete(A_c, B_c, h)
x1 = [1.0; 2.0]  # Initial state

Q = 1e-4I
R = 1e-1I
QN = 1e2I

function J(
    x::AbstractMatrix,
    u::AbstractVecOrMat;
    Q = 1e-2I, 
    R = 1e-4I, 
    QN = 1e2I
)
    u = isa(u, AbstractMatrix) ? u : reshape(u, 1, length(u))
    
    N = size(u, 2) + 1    
    J = 0.0
    for n in 1:N-1
        xₙ = x[:, n]
        uₙ = u[:, n]
        J += 1/2 * (xₙ' * Q * xₙ + uₙ' * R * uₙ)
    end
    J += 1/2 * (x[:, N]' * QN * x[:, N])
    return J
end

function J(u::AbstractVecOrMat; A=A, B=B, x1=x1, kwargs...)
    x = simulate_dlti(u, A, B, x1)
    return J(x, u; kwargs...)
end
```

#### Gradient descent

```@example trajopt
N = 40
u0 = randn(N - 1)
J(u0; A=A, B=B, x1=x1, Q=Q, R=R, QN=QN)
```

```@example trajopt
fig, ax = series(simulate_dlti(u0, A, B, x1), labels=["q", "q̇"])
axislegend(ax)
fig
```

```@example trajopt
res = optimize(u -> J(u; A=A, B=B, x1=x1, Q=Q, R=R, QN=QN), u0, GradientDescent(), 
               Optim.Options(iterations=50))
```

```@example trajopt
fig, ax = series(simulate_dlti(res.minimizer, A, B, x1), labels=["q", "q̇"])
axislegend(ax)
fig
```

```@example trajopt
stairs(res.minimizer)
```

#### Newton's method

```@example trajopt
res_newton = optimize(u -> J(u; A=A, B=B, x1=x1, Q=Q, R=R, QN=QN), u0, Newton(),
                     Optim.Options(iterations=20))
```

```@example trajopt
fig, ax = series(simulate_dlti(res_newton.minimizer, A, B, x1), labels=["q", "q̇"])
axislegend(ax)
fig
```

```@example trajopt
stairs(res_newton.minimizer)
```

### Indirect optimal control: Pontryagin

**Tagline**

"optimize, then discretize"

**Ideas**

- We can do this in a smart way by using the temporal structure of the problem.

$$\begin{align}
\min_{x_{1:N}, u_{1:{N{-}1}}} & \quad J(x_{1:N}, u_{1:{N{-}1}}) = \sum_{n=1}^{N-1} \tfrac{1}{2} x_n^T Q_n x_n + \tfrac{1}{2} u_n^T R_n u_n + \tfrac{1}{2} x_N^T Q_N x_N \\
\text{s.t.} &\quad x_{n+1} = A_n x_n + B_n u_n \\
&\quad Q_n \succeq 0,\, R_n \succ 0
\end{align}$$

- Lagrangian:

$$L(x_{1:N}, u_{1:{N{-}1}}, \lambda_{2:N}) = \sum_{n=1}^{N-1} \tfrac{1}{2} x_n^T Q_n x_n + \tfrac{1}{2} u_n^T R_n u_n + \lambda_{n+1}^T(A_n x_n + B_n u_n - x_{n+1}) + \tfrac{1}{2} x_N^T Q_N x_N$$

- KKT conditions:

$$\begin{align}
    \frac{\partial L}{\partial \lambda_n} &= (A_n x_n + B_n u_n - x_{n+1})^T \overset{!}{=} 0 \\
    \frac{\partial L}{\partial x_n} &= x_n^T Q_n + \lambda_{n+1}^T A_n - \lambda_{n}^T \overset{!}{=} 0 \\
    \frac{\partial L}{\partial x_N} &= x_N^T Q_N - \lambda_{N}^T \overset{!}{=} 0 \\
    \frac{\partial L}{\partial u_n} &= u_n^T R_n + \lambda_{n+1}^T B_n \overset{!}{=} 0
\end{align}$$

- Rewrite:

$$\begin{align}
    x_{n+1} &= A_n x_n + B_n u_n \\
    \lambda_{n} &= A_n^T \lambda_{n+1} + Q_n x_n \\
    \lambda_N &= Q_N x_N \\
    u_n &= -R_n^{-1} B_n^T \lambda_{n+1}
\end{align}$$

- There is a _forward_ equation and a _backward_ equation. This computation is _backpropagation_ through time.
- We inherit many of the same problems, e.g., vanishing / exploding gradients.

```@example trajopt
N = 60
m = 1.0 # Mass
A_c, B_c = double_integrator(m)
h = 0.1  # Time step
A, B = continuous_to_discrete(A_c, B_c, h)
x1 = [1.0; 1.0]  # Initial state

Q = 1e-4I
R = 1e-2I
QN = 1e1I

# Initial guess
u = zeros(1, N - 1)
Δu = ones(1, N - 1)
x = simulate_dlti(u, A, B, x1)
λ = zeros(size(x, 1), N)

# Line search parameters
α = 1.0 # Step size
b = 1e-2 # Tolerance
α_min = 1e-16 # Minimum step size
loop = 0
max_iterations = 20

# Verbose
verbose = false

while maximum(abs, Δu) > 1e-2 && α > α_min && loop < max_iterations
    verbose ? println("Iteration: ", loop) : nothing

    # Backward pass to compute λ and Δu
    λ[:, N] .= QN * x[:, N]
    for n = N-1:-1:1
        Δu[:, n] .= - R\B' * λ[:, n+1] - u[:, n]
        λ[:, n] .= Q * x[:, n] + A' * λ[:, n+1]
    end

    # Forward pass (with line search) to compute x
    global α = 1.0
    b = 1e-2 # Tolerance
    unew = u + α .* Δu
    xnew = simulate_dlti(unew, A, B, x1)
    
    while J(xnew, unew) > J(x, u) - b * α * norm(Δu)^2
        α = 0.5 * α
        unew = u + α .* Δu
        xnew = simulate_dlti(unew, A, B, x1)

        if verbose && α < α_min
            println("\tLine search failed to find a suitable step size")
            break
        end
    end

    u .= unew
    x .= xnew #added this dot so that global x is modified (https://discourse.julialang.org/t/local-and-global-variables-with-the-same-name-error-undefvarerror-x-not-defined/53951)
    verbose ? println("\tα = ", α) : nothing

    global loop = loop + 1
end
```

```@example trajopt
series(simulate_dlti(u, A, B, x1))
```

```@example trajopt
stairs(u[1, :])
```

**Exercise: Neural ODEs**

- Check out this [SciML note](https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/). Can we connect what we learned about Pontryagin to this work?

### Direct optimal control

**Tagline**

"discretize, then optimize"

**Ideas**

- Package the optimization variables into a trajectory
- Set up a quadratic program defined over the trajectory
- Observe the presence of sparsity

$$\begin{align}
\min_{x_{1:N}, u_{1:{N{-}1}}} &\quad J(x_{1:N}, u_{1:{N{-}1}}) = \sum_{n=1}^{N-1} \tfrac{1}{2} x_n^T Q_n x_n + \tfrac{1}{2} u_n^T R_n u_n + \tfrac{1}{2} x_N^T Q_N x_N \\
\text{s.t.} &\quad x_{n+1} = A_n x_n + B_n u_n \\
&\quad Q_n \succeq 0,\, R_n \succ 0
\end{align}$$

```@example trajopt
m = 0.1 # Mass
A_c, B_c = double_integrator(m)
h = 0.1  # Time step
A, B = sparse.(continuous_to_discrete(A_c, B_c, h))
x1 = [1.0; 1.0]  # Initial state

N = 100
x_dim = size(A, 2)
u_dim = size(B, 2)

Q = sparse(1e-4I(x_dim))
R = sparse(1e-2I(u_dim))
QN = sparse(1e2I(x_dim))
```

Define 
$$\begin{align}
Z &= \begin{bmatrix} 
    x_1 & x_2 & \cdots & x_N \\
    u_1 & u_2 & \cdots & u_N 
\end{bmatrix} \\
\Rightarrow \vec{Z} &= \begin{bmatrix} x_1 \\ u_1 \\ x_2 \\ u_2 \\ \vdots \\ x_N \\ u_N \end{bmatrix}
\end{align}$$

and drop the first state (known) and last control (not used),
$$z = \vec{Z}[\text{length}(x_1){:}\text{end}{-}\text{length}(u_N)]$$

```@example trajopt
z_dim = (N-2) * (x_dim + u_dim) + u_dim + x_dim
```

Write the cost function as $J = \tfrac{1}{2} z^T H z$, where
$$H = \begin{bmatrix}
        R_1 & 0 & 0 & & 0 \\
        0 & Q_2 & 0 & \cdots & 0 \\
        0 & 0 & R_2 & & 0 \\
        & \vdots & & \ddots & \vdots \\
        0 & 0 & 0 & \cdots & Q_N \\
    \end{bmatrix}$$

```@example trajopt
# Recall our shorthand for kron
H = blockdiag(sparse(R), I(N-2) ⊗ blockdiag(sparse(Q), sparse(R)), sparse(QN))
```

Write the dynamics constraint, $Cz = d$, where
$$C = \begin{bmatrix}
    B_1 & -I & 0 & 0 & & 0 \\
    0 & A_2 & B_2 & -I & \cdots & 0 \\
    \vdots & \vdots & \ddots & \ddots & \ddots & 0 \\
    0 & 0 & \cdots & A_{N-1} & B_{N-1} & -I
\end{bmatrix}, \qquad
d = \begin{bmatrix}
    -A_1 x_1 \\
    0 \\
    0 \\
    \vdots \\
    0
\end{bmatrix}$$

```@example trajopt
C = I(N-1) ⊗ [B -I(x_dim)]
for k = 1:N-2
    C[(k * x_dim) .+ (1:x_dim), (k * (x_dim + u_dim) - x_dim) .+ (1:x_dim)] = A
end
C
```

```@example trajopt
# Check the structure of C
k = 6
(
    C[(k * x_dim) .+ (1:x_dim), (k * (x_dim + u_dim) - x_dim) .+ (1:(2x_dim + u_dim))] 
    == [A B -I(x_dim)]
)
```

```@example trajopt
d = [-A * x1; zeros((N-2) * x_dim)];
```

Putting it all together, 
$$\begin{align}
    \min_z &\quad \tfrac{1}{2} z^T H z \\
    \text{s.t.}&\quad C z = d
\end{align}$$

- Lagrangian:
$$L(z, \lambda) = \tfrac{1}{2} z^T H z + \lambda^T (C z - d)$$

- KKT conditions:
$$\begin{align}
    & \nabla_z L = H z + C^T \lambda \overset{!}{=} 0 \\
    & \nabla_\lambda L = Cz - d \overset{!}{=} 0 \\
\end{align}$$

- Matrix form:
$$\Rightarrow \begin{bmatrix} H & C^T \\ C & 0 \end{bmatrix} 
    \begin{bmatrix} z \\ \lambda \end{bmatrix} 
    = \begin{bmatrix} 0 \\ d \end{bmatrix}$$

- Quick check: How many iterations will this take to solve?

```@example trajopt
P = [H C'; C zeros(size(C, 1), size(C, 1))]

res_qp = P \ [zeros(z_dim); d]

# Extract the minimizer
z_minimizer = res_qp[1:z_dim]
u_minimizer = z_minimizer[1:x_dim + u_dim:end];
```

```@example trajopt
fig, ax = series(simulate_dlti(u_minimizer, A, B, x1), labels=["q", "q̇"])
axislegend(ax)
fig
```

```@example trajopt
stairs(u_minimizer)
```

- Quick check: What about the temporal structure?
