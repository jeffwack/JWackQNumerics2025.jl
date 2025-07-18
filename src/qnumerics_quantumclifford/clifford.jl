
# The main package for these demos:
using QuantumClifford

# For visualizing the states under study, we have Makie recipes:
using CairoMakie

# We will also use QuantumOpticsBase to convert to state vector repr
# for visualization in a more familiar formalism:
using QuantumOpticsBase

# and BenchmarkTools to measure performance:
using BenchmarkTools

# # Pauli operators

# to define an operator, we can use the `P"..."` syntax
P"Z"

# to convert to state vector repr, we can use the `Operator` constructor
P"Z" |> Operator # from QuantumOptics

# same for Pauli X, and tensor products
P"X" |> Operator

# shorthand syntax for tensor products
P"XX" == P"X" ⊗ P"X"

# low level function to check for commutativity
comm(P"X", P"Z")

# One of the value adds of this package is that behind
# the convenient syntax, the low-level representation
# of the operators is very compact and fast.

a_billion = 1_000_000_000
l = random_pauli(a_billion)
r = random_pauli(a_billion)

@benchmark comm(l, r)

# Two bits per Pauli; multiple qubits packed in a single Int primitive type.

p = P"XIZY"

bitstring.(UInt8.(p.xz))

bits(arr) = bitstring.(UInt8.(arr));

# ## Pauli multiplication is bitwise ⊻ (XOR)

P"X" * P"Z"

#

(P"XYI".xz, P"ZZI".xz) .|> bits

#

(P"XYI" * P"ZZI").xz .|> bits

#

(P"XYI".xz .⊻ P"ZZI".xz) .|> bits

# # Stabilizer states
#
# Instead of defining a state by its amplitudes in a given basis,
# we can define it by specifying a commuting set of operators for which
# the state is an eigenstate with known eigenvalues.
# That is known as the Heienberg picture.
# We call these operators "stabilizing operators".

S"X"

#

S"X" |> Ket

#

S"Z" |> Ket

#

S"Z" ⊗ S"Z"

#

S"ZI IZ" |> Ket

#

S"ZI -IZ" |> Ket

#

S"XX ZZ" |> Ket

# It is helpful to think of them as a system of linear equations,
# a set of constraints that have only one solution,
# the unique state being stabilized.
#
# There are many ways to write equivalent set of equations,
# e.g. just by doing row operations on the matrix of coefficients.

state = S"ZI IZ"

#

row1 = state[1]
row2 = state[2]
new_state = Stabilizer([row1, row1*row2])

#

state |> Ket

#

new_state |> Ket

# You can do (binary) Gaussian elimination to get a canonical form.

canonicalize!(new_state)

# There is a variety of different ways to do the Gaussian elimination
# (do you run first X, then Z, interleave, etc), and each is useful in different contexts.

state = random_stabilizer(10, 20);
fig = Figure();
stabilizerplot_axis(fig[1,1], state);
fig

#

canonicalize!(state);
fig = Figure();
stabilizerplot_axis(fig[1,1], state);
fig

#

canonicalize_rref!(state);
fig = Figure();
stabilizerplot_axis(fig[1,1], state);
fig

#

canonicalize_clip!(state);
fig = Figure();
stabilizerplot_axis(fig[1,1], state);
fig

#

canonicalize_gott!(state);
fig = Figure();
stabilizerplot_axis(fig[1,1], state);
fig

# # Unitary dynamics of a Stabilizer state
#
# If we have a Unitary `U` and a state `|ψ⟩`, then `U|ψ⟩` is the new state after evolution.
#
# If we use the Heisenberg picture, we have an Operator `O` used as a stand-in to represent
# its eigenstate `|ψ⟩`. Then the stand-in for the evolved state is `U O U†`.
#
# On its this is not a more efficient representation.
# It is computationally worse -- we need to track a bunch of matrix-matrix multiplications,
# instead of a single matrix-vector multiplication.
# However things change, if we:
#     - stick only to operators that are efficient to implicitly represent
#     - and have an efficient rule how to update the operator when we apply a gate
#
# then we can avoid the exponential cost of the typical state vector representation.
#
# If the operators we use in the Heisenberg picture are only Pauli operators,
# then we do have everything we need for efficient representation,
# but what we lose is that we now support only Unitaries from the Clifford group,
# i.e. the unitary subgroup that maps Pauli operators to Pauli operators.

# ## Clifford action as binary matrix multiplication
#
# We can write Pauli operators as binary vectors,
# and we can decompose such vectors into weighted sums of basis vectors.

P"XYZ"

#

P"XII" * P"IXI" * P"IZI" * P"IIZ"

# Above we have the "sum" of four Pauli "binary basis vectors".
#
# The confusing thing is that these three are the same:
#     - multiplication of Pauli matrices
#     - addition (mod 2) of the binary representation of the Pauli matrices
#     - bitwise XOR of the binary representation of the Pauli matrices

# The application of Clifford unitaries is now simply a binary matrix multiplication.

tCNOT

# Preparing a Bell pair

tCNOT * S"XI IZ"

# Starting with a complicated state

tCNOT * S"XY"

#

P"XI" * P"IX" * P"IZ"

#

tCNOT

#

P"XX" * P"IX" * P"ZZ"

# Let's see it again, but in binary

gf2CNOT = stab_to_gf2(tab(tCNOT))'

#

gf2stab = stab_to_gf2(S"XY")'

#

gf2CNOT * gf2stab .% 2

#

stab_to_gf2(tCNOT*S"XY")

# # Named Small Gates
#
# Usually you just run a circuit involving a single- and two-qubit gates.
# For those we have specialized fast gate implementations.

state = random_stabilizer(100);
@benchmark apply!(state, tCNOT, [1,2])

#

@benchmark apply!(state, sCNOT(1,2))

# # Pauli Measurements
#
# Besides unitary dynamics, we also care about measurements.
#
# ## Measuring a non-commuting operator
#
# The easy case is when we measure an operator that does not commute with all the stabilizers that define the state.
# Just remove the non-commuting stabilizer and put in the measurement operator (with a random phase).

project!(S"X", P"Z")[1] # project! gives a bunch of information; here just get the final state

# But what if there are more than one non-commuting stabilizers?
# Do row operations until only one is left.
#
# Doing it manually to see the steps:

state = S"XXX ZZI IZZ"
meas = P"IXI"

QuantumClifford.mul_left!(state, 3, 2)

#

state[2] = meas
state

# Using the built-in implementation:

project!(S"XXX ZZI IZZ", P"IXI")[1]

# ## Measuring a commuting operator
#
# Now there is a definite answer, and you need to use Gaussian elimination to figure out
# what product of existing stabilizers is equal to the measurement operator (up to a phase),
# and the phase "difference" would be the measurement result

state = S"XXX ZZI IZZ"
meas = P"YYX"

QuantumClifford.mul_left!(state, 1, 2)

#

project!(S"XXX ZZI IZZ", P"YYX")[3] # project! gives a bunch of information; here just get the phase

# # The Destabilizer Formalism
#
# We can keep track for "basis" non-commuting terms in order to skip the need for
# Gaussian elimination, speeding up the simulation of commuting measurements.
# To each stabilizer operator, we assign a destabilizer which commutes with everyone
# but the given stabilizer operator.

state = S"XXX ZZI IZZ"
dstate = MixedDestabilizer(state)

#

stabilizerview(dstate)

#

destabilizerview(dstate)

#

meas = P"YYX"

#

state = random_stabilizer(100);
dstate = MixedDestabilizer(state);
meas = state[10]*state[30]*state[60];

@elapsed project!(state, meas) # TODO too many allocations

#

@elapsed project!(dstate, meas) # TODO some allocations

# # Single Qubit Measurements
#
# Similarly to the single-qubit named gates, we have faster implementations for single-qubit measurements.

dstate = random_destabilizer(500)
meas = single_x(500,4)
named_meas = sMX(4)

@benchmark apply!(_d, meas) setup=(_d=copy(dstate)) evals=1 # TODO this is faster!?

#

@benchmark apply!(_d, named_meas) setup=(_d=copy(dstate)) evals=1

# # Mixed Stabilizer States
#
# We can just give fewer than n constraints for n qubits. Such tableaux represent
# "mixed stabilizer states" (which is a specific term of art, different from just any arbitrary
# mixture of stabilizer states). A Mixed Stabilizer State is an equal mixture of all 2⁽ⁿ⁻ᶜ⁾
# stabilizer states that obey the given c<n constraints.

mdstate = MixedDestabilizer(S"ZZI IZZ")

using LinearAlgebra: rank

rank(mdstate)

# Measurements can increase the rank

apply!(mdstate, PauliMeasurement(P"XXX"))

rank(mdstate)

# Partial traces can lower the rank

traceout!(mdstate, [1])

rank(mdstate)

# # Row- and Column-orientation
#
# Lastly, as usual with linear algebra operations (albeit binary linear algebra here)
# the way the matrices are stored matters a lot to performance.

s = random_destabilizer(600)
c = random_clifford(600)
p = random_pauli(600)

typeof(tab(s))

#

typeof(fastcolumn(tab(s)))

#

# TODO not interesting
ccol = CliffordOperator(fastcolumn(tab(c))); # TODO there should be a method for clifford operator
@btime apply!(s,c);

#

@benchmark canonicalize!(_s) setup=(_s=copy(s)) evals=1

#

@benchmark canonicalize!(_s) setup=(_s=fastcolumn(copy(s))) evals=1

#

@benchmark apply!(_s, sCNOT(1,2)) setup=(_s=copy(s)) evals=1

#

@benchmark apply!(_s, sCNOT(1,2)) setup=(_s=fastcolumn(copy(s))) evals=1
