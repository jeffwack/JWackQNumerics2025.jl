# Quantum trajectories 
Claim: you should not use master equation solvers. You don't pay the exponential
cost when working with classical probability distributions, instead we use Monte
Carlo simulations. How can we do the same with quantum systems?

## Good resources

[QuantumOptics documentation](https://docs.qojulia.org/timeevolution/mcwf/)

[QuTiP documentation](https://qutip.org/docs/4.5/guide/dynamics/dynamics-monte.html)

[Mølmer, Castin, and Dalibard](https://opg.optica.org/josab/fulltext.cfm?uri=josab-10-3-524&id=59382)

## Monte Carlo wavefunction in QuantumOptics.jl

```@example qt
using QuantumOptics

cutoff = 32
B = FockBasis(cutoff)
a = destroy(B)
ω = 1.0
H = ω*a'*a

γ = 0.1
L = [√γ*a]

ψ = fockstate(B, 18)

ts = 0:0.01:3*2*π

_, ρs = timeevolution.master(ts,ψ,H,L)

_, ψs = timeevolution.mcwf(ts,ψ,H,L)
nothing #hide
```

```@example qt
using CairoMakie
fig = Figure()
ax = Axis(fig[1,1], aspect = DataAspect())

lines!(ax,ts,real.(expect.((a'*a,),ρs)))
lines!(ax,ts,real.(expect.((a'*a,),ψs)))

fig
```

Is the environment photon counting or doing homodyne readout?

We can make lots of trajectories and average them...

```@example qt
fig = Figure()
ax = Axis(fig[1,1], aspect = DataAspect())

_, ρs = timeevolution.master(ts,ψ,H,L)

lines!(ax,ts,real.(expect.((a'*a,),ρs)))

trajectories = []
n_traj = 100

for _ in 1:n_traj 
    _, ψs = timeevolution.mcwf(ts,ψ,H,L)
    push!(trajectories,ψs)
    lines!(ax,ts,real.(expect.((a'*a,),ψs)), color = (:gray,0.1))
end

sols_avg = sum([dm.(ψs) for ψs in trajectories])/n_traj
lines!(ax,ts, real.(expect.((a'*a,), sols_avg)))

fig
```
## Cat states

```@example qt
α = 4.0

l0 = (coherentstate(B,α) + coherentstate(B,-α)) / √2
l1 = (coherentstate(B,1im*α) + coherentstate(B,-1im*α)) / √2

mix0 = (dm(coherentstate(B,α)) + dm(coherentstate(B,-α))) / 2
mix1 = (dm(coherentstate(B,1im*α)) + dm(coherentstate(B,-1im*α))) / 2
nothing #hide
#TODO: make 4 panel Wigner function plot
```

```@example qt
H = ω*a'*a
ts = 0:0.01:π/2

_, ψs = timeevolution.schroedinger(ts,l0,H)

lines(ts, real.(expect.((projector(l0),), ψs)))
lines!(ts, real.(expect.((projector(l1),), ψs)))
current_figure()
```
We can think of this as a logical gate, it exchanges our two states. 

We can add loss:

```@example qt
γ = 0.01
L = [√γ*a]
_,ρs = timeevolution.master(ts,l0,H,L);
lines!(ts, real.(expect.((projector(l0),),ρs)), linestyle=:dash)
lines!(ts, real.(expect.((projector(l1),),ρs)), linestyle=:dash)
current_figure()
```

We can use non Hermitian time evolution as a worst case analysis. 
```@example qt
Hnotherm = H - im/1 *L[1]'*L[1]
_,ψnhs = timeevolution.schroedinger(ts,l0,Hnotherm);
lines!(ts, real.(expect.((projector(l0),),ψnhs)), linestyle=:dot)
lines!(ts, real.(expect.((projector(l1),),ψnhs)), linestyle=:dot)
current_figure()
```
