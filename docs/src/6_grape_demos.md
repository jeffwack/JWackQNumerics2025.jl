# GRAPE Demos

```@example grape
using Piccolo
using Optim
using LinearAlgebra
using SparseArrays
using CairoMakie

# useful
const ⊗ = kron
```

## Goals

- Learn the quantum isomorphisms that map variables to real-valued state vectors
- Study how gradient descent and Newton's method can be used to optimize quantum controls.

## I. Isomorphisms

**Piccolo isomorphisms**
- The standard quantum states are _kets_, $|\psi\rangle$, and _Unitaries_, $U$.
- Open quantum system require _density matrices_, $\rho$, and _quantum channels_, $\Phi$.
- Standard quantum states have an open system counterpart,

$$\begin{align}
    \text{closed} &\longrightarrow \text{open}  \\ \hline
    |\psi\rangle &\longrightarrow |\psi\rangle \langle \psi | \\
    U &\longrightarrow U \cdot U^\dagger 
\end{align}$$

🚧 ⚠️ If you are seeing a lot of boxes like Ũ⃗, it is _very_ useful to have the [JuliaMono](https://juliamono.netlify.app/) fonts for Piccolo. Install and [change the default font family](https://code.visualstudio.com/docs/terminal/appearance).

```@example grape
# Ok, so it's not technically a wavefunction
ψ = [1; 2] + im * [3; 4]

ψ̃ = ket_to_iso(ψ)
```

```@example grape
iso_to_ket(ψ̃)
```

```@example grape
# We often need to convert a complex matrix U to a real vector, Ũ⃗. 
U = [1 5; 2 6] + im * [3 7; 4 8]
```

Remember what you learned about Julia arrays! Why would I write the matrix this way?

```@example grape
Ũ⃗ = operator_to_iso_vec(U)
```

```@example grape
iso_vec_to_operator(Ũ⃗)
```

Physics check: What's an efficiency that we might be able to leverage when storing $\rho$ that you don't see here?

```@example grape
# Warning: The isomorphism `density_to_iso_vec` is not the same as `operator_to_iso_vec`.
ρ = [1 2; 3 4] + im * [5 6; 7 8]
ρ̃⃗ = density_to_iso_vec(ρ)
```

**Exercise** 
- Just how big are these vectors for a single qubit state? A two qubit state? 
- What about quantum channels?

## II. Quantum dynamics

**Quantum systems**

First up, we are going to look at some dynamics convenience functions in Piccolo.

- Let's flip a qubit from the ground state to the excited state.
- Introduce the isomorphisms that make quantum dynamics real-valued.  
- Use [PiccoloQuantumObjects](https://docs.harmoniqs.co/PiccoloQuantumObjects/dev/) to make a quantum system.
- Use a rollout to integrate the quantum system forward in time.

$$H(u(t)) = \underbrace{u_1(t) XI + u_2(t) YI}_\text{qubit 1} 
    + \underbrace{u_3(t) IX + u_4(t) IY}_\text{qubit 2} + \underbrace{u_5(t) XX}_\text{coupling}$$

```@example grape
H_drives = [
    PAULIS.X ⊗ PAULIS.I,
    PAULIS.Y ⊗ PAULIS.I,
    PAULIS.I ⊗ PAULIS.X,
    PAULIS.I ⊗ PAULIS.Y,
    PAULIS.X ⊗ PAULIS.X
]

system = QuantumSystem(H_drives)
```

- Quantum systems contain the operators we need, including the real valued versions.

```@example grape
get_drift(system)
```

- Quick check: What do we expect to see?

```@example grape
get_drives(system)[1]
```

```@example grape
system.H(randn(system.n_drives))
```

- Quick check: How big will this operator be?

```@example grape
system.G(randn(system.n_drives))
```

- We can use a system to perform a rollout.

```@example grape
# Timing information (e.g. 20 ns superconducting qubit gate)
T = 40
Δt = 0.5
timesteps = fill(Δt, T)
```

```@example grape
# Controls
controls = randn(system.n_drives, T + 1);
```

```@example grape
unitary_rollout(controls, timesteps, system)
```

```@example grape
# Entangling gate
U_goal = GATES.CX

# How'd we do?
println("ℱ = ", unitary_rollout_fidelity(U_goal, controls, timesteps, system))
```

**We have all the pieces we need to solve!**

Let's put Piccolo to work.

```@example grape
# Piccolo (we'll learn more about this later)
prob = UnitarySmoothPulseProblem(system, U_goal, T, Δt);
```

```@example grape
# save these initial controls for later
a_init = prob.trajectory.a
plot(prob.trajectory, :a)
```

```@example grape
solve!(
    prob, 
    max_iter=20, print_level=1, verbose=false, options=IpoptOptions(eval_hessian=false)
)

ℱ = unitary_rollout_fidelity(prob.trajectory, system)

println("The fidelity is ", ℱ)
```

```@example grape
a_final = prob.trajectory.a
plot(prob.trajectory, :a)
```

## III. GRAPE

The [GRAPE algorithm](https://doi.org/10.1016/j.jmr.2004.11.004) comes from NMR in 2004, and there is a [Julia version](https://github.com/JuliaQuantumControl/GRAPE.jl). We'll reproduce GRAPE in this example.

```@example grape
# We work with timesteps between knot points
timesteps = fill(Δt, T)

# Let's use our previous function to compute the fidelity
GRAPE(controls) = abs(1 - unitary_rollout_fidelity(U_goal, controls, timesteps, system))
```

### Automatic differentiation
- It's quick to test! Compare different algorithms, e.g., `Newton()`, `GradientDescent()`, `LBFGS()`
- If you switch from gradient descent to a quasi-Newton method, you get to [write another paper](https://www.sciencedirect.com/science/article/abs/pii/S1090780711002552).

```@example grape
result_GRAPE = optimize(GRAPE, collect(a_init), LBFGS())
```

```@example grape
a_GRAPE = Optim.minimizer(result_GRAPE)
println("The fidelity is ", unitary_rollout_fidelity(U_goal, a_GRAPE, timesteps, system))
```

- What do we think we'll see here?

```@example grape
series(cumsum(timesteps), a_GRAPE)
```

### Analytic gradients

**Calculus practice**
- We can combine forward and backward rollouts to compute the gradients,
$$\begin{align}
    \frac{\partial U(T)}{\partial u_k(t)} &= U(T, t) (-i H_k \Delta t) U(t) \\
   \Rightarrow \langle\psi_\text{goal} | \frac{\partial U(T)}{\partial u_k(t)} |\psi_\text{init.}\rangle &= -i \Delta t \langle\psi_\text{goal}^\text{bwd.}(t) | H_k |\psi_\text{init.}^\text{fwd.}(t) \rangle.
\end{align}$$

**Exercise**
- Implement gradient descent using the analytic gradients.
- Sometimes, there are insights you can only get by opening up the black box, e.g. [d-GRAPE](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.042122).

## IV. Function Spaces

- Pick a function basis for the controls and optimize the coefficients. Some choices are [trig functions](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.84.022326) or [Slepians](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.062346).
- Our optimization parameters are now coefficients of the basis,
$$u(t) = u_0 + \sum_{j=1}^{n} c_j a_j(t)$$
- The modes $a_j(t)$ stay fixed, and the coefficients $c_j$ are optimized.

```@example grape
# First n = 5 entries in a Fourier series, including the constant term
n = 5
fourier_series = [cos.(π * j * (0:T-1) / T .- π/2) for j in 0:n-1]

function get_controls(coefficients)
    a(c) = sum(cⱼ * aⱼ for (cⱼ, aⱼ) in zip(c, fourier_series))
    return stack([a(c) for c in eachrow(coefficients)], dims=1)
end

function GRAFS(coefficients)
    controls = get_controls(coefficients)
    return abs(1 - unitary_rollout_fidelity(U_goal, controls, timesteps, system))
end
```

```@example grape
c_init = rand(system.n_drives, n)
result_GRAFS = optimize(GRAFS, c_init, LBFGS())
```

```@example grape
a_GRAFS = Optim.minimizer(result_GRAFS)
println("The fidelity is ", 1 - unitary_rollout_fidelity(U_goal, a_GRAFS, timesteps, system))
```

```@example grape
series(cumsum(timesteps), get_controls(a_GRAFS))
```

- These shapes are a lot nicer! But performance depends a lot on the expressivity and initial condition.

```@example grape
c_init = randn(system.n_drives, n)
result_GRAFS_2 = optimize(GRAFS, c_init, LBFGS())
```

```@example grape
a_GRAFS_2 = Optim.minimizer(result_GRAFS_2)
println("The fidelity is ", 1 - unitary_rollout_fidelity(U_goal, a_GRAFS_2, timesteps, system))
```

```@example grape
f = Figure()
ax = Axis(f[1,1])
series!(ax, cumsum(timesteps), get_controls(a_GRAFS))
ax = Axis(f[2,1])
series!(ax, cumsum(timesteps), get_controls(a_GRAFS_2))
f
```

**Exercise: A filtering approach**

- Pass the controls through a spectral filter: Look up Slepians and consider how to bound the bandwidth by choice of basis.
- How might we shape the bandwidth of the controls? (Remember, we can just rely on automatic differentiation!)

## V. States in costs

**Exercise:**

- Let's switch to a transmon, which has more than two levels and can be _leaky_.

$$H(u(t)) = \tfrac{1}{2} \eta a^\dagger a^\dagger a a + u_1(t) (a + a^\dagger) - i u_2(t) (a - a^\dagger)$$

- The optimizer can exploit the higher levels!

- Add a leakage penalty to a guard state. _Notice that working with states can be awkward._

```@example grape
T = 40
Δt = 0.25
timesteps = fill(Δt, T)

function Transmon(n)
    a = annihilate(n)
    x = a + a'
    p = -im * (a - a')
    η = 0.1
    return QuantumSystem(1/2 * a'a'a*a, [x, p])
end

transmon_2 = Transmon(2)
transmon_4 = Transmon(4)
```

```@example grape
function TransmonGRAFS(
    goal::AbstractPiccoloOperator, coefficients, timesteps, sys::AbstractQuantumSystem
)
    controls = get_controls(coefficients)
    return abs(1 - unitary_rollout_fidelity(goal, controls, timesteps, sys))
end
```

- Quick aside: _Embedded operators_

```@example grape
U_emb(n) = EmbeddedOperator(GATES.X, 1:2, n)
```

```@example grape
U_emb(4).operator
```

```@example grape
unembed(U_emb(4))
```

```@example grape
sys2, U2 = Transmon(2), U_emb(2)
c_init = randn(sys2.n_drives, n)
result_GRAFS_3 = optimize(a -> TransmonGRAFS(U2, a, timesteps, sys2), c_init, LBFGS())
```

```@example grape
a_GRAFS_3 = get_controls(Optim.minimizer(result_GRAFS_3))
println("The fidelity is ", unitary_rollout_fidelity(U2, a_GRAFS_3, timesteps, sys2))
```

- Quick check: What might happen now?

```@example grape
println(
    "The fidelity is ", unitary_rollout_fidelity(U_emb(4), a_GRAFS_3, timesteps, Transmon(4))
)
```

**TODO:** 
- Add an L2 penalty to states that are not in the computational basis.
- Use a modified GRAPE cost to penalize leakage while maintaining fidelity.
- Study how leakage and fidelity change with the penalty.
- Study how the anharmonicity η affects leakage.
