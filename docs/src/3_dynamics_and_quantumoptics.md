# Dynamics and QuantumOptics.jl - Stefan Krastanov

This package should be thought of as a domain specific linear algebra wrapper.

## Operator basics

```@example qo
using QuantumOpticsBase

cutoff = 25

F = FockBasis(cutoff)

Ψ = fockstate(F,5)
```

```@example qo
S = SpinBasis(1//2)
s = spindown(S)
```

```@example qo
G = GenericBasis(100)
basisstate(G,2)
```

```@example qo
C = F ⊗ S

sparse(dm(Ψ⊗s))

```

```@example qo
projector(Ψ⊗s)
```

```@example qo
using BenchmarkTools
P = projector(Ψ)
Ps = sparse(P)

@benchmark P*Ψ
```
```@example qo
@benchmark Ps*Ψ
```

We can use in-place or mutating functions to improve performance. This is one of
the first things to think about optimizing when looking for speedups.
```@example qo
using LinearAlgebra: mul!

buffer = copy(Ψ)
@benchmark mul!(buffer, Ps, Ψ)
```

## Composite spaces
We can create composite spaces and operators with the '\otimes' symbol
representing the tensor product.

```@example qo
psi = fockstate(F,4)⊗spinup(S)
```

```@example qo
create(F)⊗identityoperator(S)
```
This does not scale well! We have a function called embed to address this
problem

```@example qo
embed(F⊗S, 1, create(F))
```

We can take a partial trace with ptrace()

```@example qo
l = spindown(S)
h = spinup(S)

bell = (l⊗l + h⊗h) / √2

ptrace(bell,2)
```

## Coherent states

```@example qd
using QuantumOptics
```

QuantumOptics depends on DifferentialEquations, so has a much longer precompile
time compared to QuantumOpticsBase, which defines the methods for working with
quantum states and operators without any time evolution. 

We will work with the quantum harmonic oscillator and look at coherent states in
phase space.

```@example qd
cutoff = 32
B = FockBasis(cutoff)
ω = 1.0
a = destroy(B)

H = ω*a'*a
```

Now we can make a coherent state
```@example qd
α = 4
ψ = coherentstate(B, α)
```

```@example qd
a*ψ ≈ α*ψ
```
TODO: what is going wrong here?


```@example qd
using CairoMakie
quad = -10:0.1:10

w = wigner(ψ,quad,quad)

fig = Figure()
ax = Axis(fig[1,1], aspect = DataAspect())
heatmap!(quad,quad,w)
fig
```

## Schrodinger dynamics

```@example qd
ts = 0:0.1:3*2*π

_, ψs = timeevolution.schroedinger(ts,ψ,H)

nothing #hide
```

```@example qd
x = (a' + a)/√2
p = 1im*(a' - a)/√2

fig = Figure()
ax = Axis(fig[1,1], aspect = DataAspect())
lines!(ax,ts,real.(expect.((x,),ψs)))
lines!(ax,ts,real.(expect.((p,),ψs)))
fig
```

```@example qd
lines(norm.(ψs))
```

First, the naive way to create a time-dependent Hamiltonian
```@example qd
function Hdynamic(t,psi)
    return H+p*4*sin(8*ω*t)
end

_, ψs = timeevolution.schroedinger_dynamic(ts,ψ,Hdynamic)

fig = Figure()
ax = Axis(fig[1,1], aspect = DataAspect())
lines!(ax,ts,real.(expect.((x,),ψs)))
lines!(ax,ts,real.(expect.((p,),ψs)))
fig
```

SciMLOperators has an interesting way to create time-dependent operators in an
allocation-efficient way.

```@example qd
using BenchmarkTools
Hlazy = H + TimeDependentSum(t-> 4*sin(8*ω*t),p)
@benchmark _, ψs = timeevolution.schroedinger_dynamic(ts, ψ, Hlazy)
```
 And the old way:
```@example qd
@benchmark _, ψs = timeevolution.schroedinger_dynamic(ts,ψ,Hdynamic)
```

## Trying out different solvers

```@example qd
using OrdinaryDiffEqLowOrderRK: DP5
using OrdinaryDiffEqTsit5: Tsit5
using OrdinaryDiffEqVerner: Vern8
```

[A good discussion on Schrodinger dynamics and solver tolerance](https://discourse.julialang.org/t/setting-abstol-and-reltol-when-solving-the-schrodinger-equation-with-ordinarydiffeq/125534)

```@example qd
@benchmark _, ψs = timeevolution.schroedinger_dynamic(ts, ψ, Hlazy; alg =
DP5())
```
```@example qd
@benchmark _, ψs = timeevolution.schroedinger_dynamic(ts, ψ, Hlazy; alg =
Tsit5())
```
```@example qd
@benchmark _, ψs = timeevolution.schroedinger_dynamic(ts, ψ, Hlazy; alg =
Vern8())
```

## Master equation evolution

```@example qd
γ = 0.1
_, ρs = timeevolution.master(ts,ψ,H,[√γ*a])

fig = Figure()
ax = Axis(fig[1,1], aspect = DataAspect())
lines!(ax,ts,real.(expect.((x,),ρs)))
lines!(ax,ts,real.(expect.((p,),ρs)))
fig
```

TODO: Lindblad is nonlinear with respect to collapse operators? 


