# Introduction to state vector simulation

In this lesson, we're going to implement a simple state vector simulator and steadily made it less simple (and more performant) as a way to examine some performance guidelines for Julia and numerical physics.

First, let's remind ourselves of some basic facts about state vectors.

## State vectors, gates, and observables

### State vector basics

A *state vector* is a length-$D^N$ complex vector representing probability amplitudes for a quantum system of $N$ particles, each with $D$ possible states. So, for *qubits*, which have 2 possible states ($|0\rangle$ and $|1\rangle$), the state vector has $2^N$ elements, and for *qutrits*, which have 3 possible states, the state vector would have $3^N$ elements.

**In this lecture we'll focus entirely on the qubit case, as it's most common.**

For just a single qubit, we have a 2-element vector:

```math
|\psi_1\rangle = \begin{bmatrix} \psi_0 \\ \psi_1 \end{bmatrix}
```

Where this corresponds to 

```math
|\psi_1\rangle = \psi_0 | 0 \rangle + \psi_1 | 1 \rangle
```

For two qubits, we have a 4-element vector:

```math
|\psi_2\rangle = \begin{bmatrix} \psi_{00} \\ \psi_{01} \\ \psi_{10} \\ \psi_{11} \end{bmatrix}
```

Where this corresponds to

```math
|\psi_2\rangle = \psi_{00} | 00 \rangle + \psi_{01} | 01 \rangle + \psi_{10} | 10 \rangle + \psi_{11} | 11 \rangle
```

For $N$ qubits, the state vector has $2^N$ elements. We can think of the index within the vector as corresponding to a base-2 representation of the computational basis. So for a vector with index $i$, we have the state $|i\rangle = |i_{N-1} ... i_0\rangle$.

### Gate application

*Gates* are unitary matrices that transform state vectors. For a single qubit system, gates are $2 \times 2$ matrices. For an $N$-qubit system, gates are $2^N \times 2^N$ matrices.

In Julia, we'll represent our single-qubit gates as follows:

```@example sv
using LinearAlgebra
using Chairmarks
using Profile
using AliasTables
```

```@example sv
const X = float.(complex.([0 1; 1 0]))
const Y = float.([0 -im; im 0])
const Z = float.(complex.([1 0; 0 -1]))
const I = float.(complex.([1 0; 0 1]))
const H = float.(complex.(1/√2 * [1 1; 1 -1]))
```

Most quantum gates act on a single qubit, but the state vector involves many qubits. We can extend a single-qubit gate to act on the entire state vector by taking a tensor product with the identity matrix for all other qubits.

For example, if we have a 3-qubit state vector and we want to apply gate $G$ to the second qubit (qubit 1), we need to apply $I \otimes G \otimes I$ to the state vector. In Julia, the tensor product is implemented as `kron`.

```@example sv
function apply_gate_naive(ψ::Vector, gate::Matrix, gate_qubit::Int)
    n_qubits  = Int( log2(length(ψ)) )
    all_gates = [I for qubit in 1:n_qubits]
    all_gates[n_qubits - gate_qubit] = gate # Julia is one indexed!
    full_gate = reduce(kron, all_gates)
    return full_gate * ψ
end
```

Let's test this with a simple example. We'll create a 3-qubit state vector and apply an X gate to the second qubit:

```@example sv
# Create a 3-qubit state vector in the |000⟩ state
ψ = zeros(ComplexF64, 8)
ψ[1] = 1.0  # |000⟩ corresponds to index 1
```

```@example sv
# Apply X gate to qubit 1 (second qubit)
ψ_new = apply_gate_naive(ψ, X, 1)
```

```@example sv
# This should give us |010⟩, which corresponds to index 3
findall(x -> abs(x) > 1e-10, ψ_new)
```

### Observables

*Observables* are Hermitian matrices that correspond to measurable quantities. For a single qubit, common observables are the Pauli matrices $X$, $Y$, and $Z$.

The *expectation value* of an observable $O$ on a state vector $|\psi\rangle$ is:

```math
\langle O \rangle = \langle \psi | O | \psi \rangle
```

For a multi-qubit system, we need to extend the observable to act on the entire state vector using tensor products, just like we did for gates.

```@example sv
function expectation_value_naive(ψ::Vector, observable::Matrix, observable_qubit::Int)
    n_qubits  = Int( log2(length(ψ)) )
    all_observables = [I for qubit in 1:n_qubits]
    all_observables[n_qubits - observable_qubit] = observable
    full_observable = reduce(kron, all_observables)
    return real(ψ' * full_observable * ψ)
end
```

Let's test this with our X gate example:

```@example sv
# Expectation value of Z on qubit 1 for |000⟩
expectation_value_naive(ψ, Z, 1)
```

```@example sv
# Expectation value of Z on qubit 1 for |010⟩
expectation_value_naive(ψ_new, Z, 1)
```

## Performance considerations

The naive approach works, but it's not very efficient. The main problem is that we're creating a large matrix ($2^N \times 2^N$) for each gate application, which becomes prohibitively expensive for large systems.

Let's create a larger system to see the performance issues:

```@example sv
n_qubits = 8
ψ_large = zeros(ComplexF64, 2^n_qubits)
ψ_large[1] = 1.0
```

```@example sv
# This will be slow and memory-intensive
@b apply_gate_naive(ψ_large, X, 4)
```

For a 16-qubit system, we need to create a $65536 \times 65536$ matrix, which uses about 34 GB of memory! This is clearly not sustainable.

## Optimized approaches

The key insight is that we don't need to create the full matrix. Instead, we can directly manipulate the state vector using more efficient algorithms.

### Approach 1: Tensor reshaping

One approach is to reshape the state vector into a tensor and apply gates more efficiently:

```@example sv
function apply_gate_reshaped1(ψ::Vector{ComplexF64}, gate::Matrix{ComplexF64}, gate_qubit::Int)
    n_qubits = Int(log2(length(ψ)))
    
    # Reshape the state vector into a tensor
    tensor_shape = ntuple(i -> 2, n_qubits)
    ψ_tensor = reshape(ψ, tensor_shape)
    
    # Apply the gate to the specified qubit
    result = similar(ψ_tensor)
    
    # We need to contract the gate with the appropriate tensor dimension
    # This is a simplified version - full implementation would be more complex
    
    return vec(result)
end
```

A more direct approach is to use the structure of the tensor product:

```@example sv
function apply_gate_reshaped2(ψ::Vector{ComplexF64}, gate::Matrix{ComplexF64}, gate_qubit::Int)
    n_qubits = Int(log2(length(ψ)))
    
    # Create a copy of the state vector
    ψ_new = copy(ψ)
    
    # The key insight is that we can operate on chunks of the state vector
    chunk_size = 2^gate_qubit
    stride = 2^(gate_qubit + 1)
    
    for i in 1:chunk_size:length(ψ)
        # Apply the gate to this chunk
        if i + chunk_size - 1 <= length(ψ)
            # Get the two components for this qubit
            α = ψ[i:i+chunk_size-1]
            β = ψ[i+chunk_size:min(i+stride-1, length(ψ))]
            
            # Apply the gate
            ψ_new[i:i+chunk_size-1] = gate[1,1] * α + gate[1,2] * β
            if i + stride - 1 <= length(ψ)
                ψ_new[i+chunk_size:i+stride-1] = gate[2,1] * α + gate[2,2] * β
            end
        end
    end
    
    return ψ_new
end
```

### Approach 2: Bit manipulation

A more efficient approach uses bit manipulation to directly compute which elements of the state vector need to be modified:

```@example sv
function apply_gate_shifting1(ψ::Vector{ComplexF64}, gate::Matrix{ComplexF64}, gate_qubit::Int)
    n = length(ψ)
    ψ_new = copy(ψ)
    
    for i in 0:(n-1)
        # Check if the gate_qubit is 0 or 1 in the binary representation of i
        if (i >> gate_qubit) & 1 == 0
            # The qubit is 0, so we need to find the corresponding 1 state
            j = i | (1 << gate_qubit)  # Set the gate_qubit to 1
            
            # Apply the gate
            α = ψ[i+1]  # +1 because Julia is 1-indexed
            β = ψ[j+1]
            
            ψ_new[i+1] = gate[1,1] * α + gate[1,2] * β
            ψ_new[j+1] = gate[2,1] * α + gate[2,2] * β
        end
    end
    
    return ψ_new
end
```

Let's test this implementation:

```@example sv
ψ_test = zeros(ComplexF64, 8)
ψ_test[1] = 1.0
```

```@example sv
# Apply X gate to qubit 1
ψ_result = apply_gate_shifting1(ψ_test, X, 1)
```

```@example sv
# Compare with naive implementation
ψ_naive = apply_gate_naive(ψ_test, X, 1)
```

```@example sv
# Check if they're the same
maximum(abs.(ψ_result - ψ_naive))
```

Great! Now let's benchmark the performance:

```@example sv
@b apply_gate_shifting1(ψ_large, X, 4)
```

This is much faster than the naive approach! But we can do even better.

### Optimized bit manipulation

We can optimize the bit manipulation approach by eliminating redundant work:

```@example sv
function apply_gate_shifting2(ψ::Vector{ComplexF64}, gate::Matrix{ComplexF64}, gate_qubit::Int)
    n = length(ψ)
    ψ_new = copy(ψ)
    
    mask = 1 << gate_qubit
    
    for i in 0:(n-1)
        if (i & mask) == 0  # Only process when the target qubit is 0
            j = i | mask    # Corresponding index with target qubit = 1
            
            α = ψ[i+1]
            β = ψ[j+1]
            
            ψ_new[i+1] = gate[1,1] * α + gate[1,2] * β
            ψ_new[j+1] = gate[2,1] * α + gate[2,2] * β
        end
    end
    
    return ψ_new
end
```

```@example sv
@b apply_gate_shifting2(ψ_large, X, 4)
```

We can make this even more efficient by using vectorized operations:

```@example sv
function apply_gate_shifting_linear(ψ::Vector{ComplexF64}, gate::Matrix{ComplexF64}, gate_qubit::Int)
    n = length(ψ)
    ψ_new = copy(ψ)
    
    # Create masks for efficient bit manipulation
    mask = 1 << gate_qubit
    
    # Process in chunks to utilize vectorization
    chunk_size = 2^(gate_qubit + 1)
    lower_chunk = 2^gate_qubit
    
    for start in 0:chunk_size:(n-1)
        # Process the chunk
        for i in start:(start + lower_chunk - 1)
            if i + lower_chunk < n
                j = i + lower_chunk
                
                α = ψ[i+1]
                β = ψ[j+1]
                
                ψ_new[i+1] = gate[1,1] * α + gate[1,2] * β
                ψ_new[j+1] = gate[2,1] * α + gate[2,2] * β
            end
        end
    end
    
    return ψ_new
end
```

```@example sv
@b apply_gate_shifting_linear(ψ_large, X, 4)
```

### Threading for additional performance

For very large systems, we can use threading to parallelize the computation:

```@example sv
function apply_gate_threaded(ψ::Vector{ComplexF64}, gate::Matrix{ComplexF64}, gate_qubit::Int)
    n = length(ψ)
    ψ_new = copy(ψ)
    
    chunk_size = 2^(gate_qubit + 1)
    lower_chunk = 2^gate_qubit
    
    Threads.@threads for start in 0:chunk_size:(n-1)
        for i in start:(start + lower_chunk - 1)
            if i + lower_chunk < n
                j = i + lower_chunk
                
                α = ψ[i+1]
                β = ψ[j+1]
                
                ψ_new[i+1] = gate[1,1] * α + gate[1,2] * β
                ψ_new[j+1] = gate[2,1] * α + gate[2,2] * β
            end
        end
    end
    
    return ψ_new
end
```

```@example sv
@b apply_gate_threaded(ψ_large, X, 4)
```

## Performance comparison

Let's compare all our implementations:

```@example sv
println("Naive approach:")
@b apply_gate_naive(ψ_large, X, 4)
```

```@example sv
println("Bit shifting approach:")
@b apply_gate_shifting_linear(ψ_large, X, 4)
```

```@example sv
println("Threaded approach:")
@b apply_gate_threaded(ψ_large, X, 4)
```

## Expectation values

We can apply similar optimizations to expectation value calculations:

```@example sv
function expectation_value_optimized(ψ::Vector{ComplexF64}, observable::Matrix{ComplexF64}, observable_qubit::Int)
    n = length(ψ)
    result = 0.0
    
    chunk_size = 2^(observable_qubit + 1)
    lower_chunk = 2^observable_qubit
    
    for start in 0:chunk_size:(n-1)
        for i in start:(start + lower_chunk - 1)
            if i + lower_chunk < n
                j = i + lower_chunk
                
                α = ψ[i+1]
                β = ψ[j+1]
                
                # Compute the expectation value contribution
                result += real(conj(α) * observable[1,1] * α + 
                              conj(α) * observable[1,2] * β +
                              conj(β) * observable[2,1] * α + 
                              conj(β) * observable[2,2] * β)
            end
        end
    end
    
    return result
end
```

```@example sv
# Test the optimized expectation value
expectation_value_optimized(ψ_large, Z, 4)
```

```@example sv
# Compare with naive implementation
expectation_value_naive(ψ_large, Z, 4)
```

## Multi-qubit gates

For multi-qubit gates (like CNOT), we need to extend our approach:

```@example sv
function apply_cnot(ψ::Vector{ComplexF64}, control_qubit::Int, target_qubit::Int)
    n = length(ψ)
    ψ_new = copy(ψ)
    
    control_mask = 1 << control_qubit
    target_mask = 1 << target_qubit
    
    for i in 0:(n-1)
        # Only apply if control qubit is 1
        if (i & control_mask) != 0
            # Find the index with target qubit flipped
            j = i ⊻ target_mask  # XOR flips the target bit
            
            # Swap the amplitudes
            ψ_new[i+1] = ψ[j+1]
            ψ_new[j+1] = ψ[i+1]
        end
    end
    
    return ψ_new
end
```

```@example sv
# Test CNOT gate
ψ_cnot_test = zeros(ComplexF64, 4)
ψ_cnot_test[3] = 1.0  # |10⟩ state
```

```@example sv
# Apply CNOT with control=1, target=0
ψ_cnot_result = apply_cnot(ψ_cnot_test, 1, 0)
```

## Sampling from quantum states

Often in quantum simulation, we need to sample from the probability distribution defined by the state vector:

```@example sv
function naive_sample(ψ::Vector{ComplexF64}, n_shots::Int)
    probabilities = abs2.(ψ)
    samples = Int[]
    
    for _ in 1:n_shots
        r = rand()
        cumulative = 0.0
        
        for (i, p) in enumerate(probabilities)
            cumulative += p
            if r <= cumulative
                push!(samples, i-1)  # Convert to 0-indexed
                break
            end
        end
    end
    
    return samples
end
```

```@example sv
# Test sampling
samples = naive_sample(ψ_large, 1000)
```

```@example sv
# Count the frequency of each outcome
using StatsBase
countmap(samples)
```

For better performance with many samples, we can use the alias method:

```@example sv
function alias_sample(ψ::Vector{ComplexF64}, n_shots::Int)
    probabilities = abs2.(ψ)
    
    # Create alias table
    alias_table = AliasTables.AliasTable(probabilities)
    
    # Sample from the alias table
    samples = [rand(alias_table) - 1 for _ in 1:n_shots]  # Convert to 0-indexed
    
    return samples
end
```

```@example sv
# Test alias sampling
alias_samples = alias_sample(ψ_large, 1000)
```

```@example sv
# Compare performance
@b naive_sample(ψ_large, 1000)
```

```@example sv
@b alias_sample(ψ_large, 1000)
```

## Mixed states and density matrices

For mixed states, we need to work with density matrices instead of state vectors:

```@example sv
function apply_gate_density_matrix(ρ::Matrix{ComplexF64}, gate::Matrix{ComplexF64}, gate_qubit::Int)
    n_qubits = Int(log2(size(ρ, 1)))
    
    # Create the full gate matrix
    all_gates = [I for qubit in 1:n_qubits]
    all_gates[n_qubits - gate_qubit] = gate
    full_gate = reduce(kron, all_gates)
    
    # Apply the gate: ρ_new = U * ρ * U†
    return full_gate * ρ * full_gate'
end
```

```@example sv
# Create a mixed state (50% |0⟩ and 50% |1⟩)
ρ_mixed = zeros(ComplexF64, 2, 2)
ρ_mixed[1,1] = 0.5  # |0⟩⟨0|
ρ_mixed[2,2] = 0.5  # |1⟩⟨1|
```

```@example sv
# Apply Hadamard gate
ρ_mixed_h = apply_gate_density_matrix(ρ_mixed, H, 0)
```

## Putting it all together: Circuit simulation

Let's create a simple function to simulate a quantum circuit:

```@example sv
function simulate_circuit(initial_state::Vector{ComplexF64}, gates::Vector{Tuple{Matrix{ComplexF64}, Int}})
    ψ = copy(initial_state)
    
    for (gate, qubit) in gates
        ψ = apply_gate_shifting_linear(ψ, gate, qubit)
    end
    
    return ψ
end
```

```@example sv
# Create a Bell state circuit: H on qubit 1, then CNOT(1,0)
initial_state = zeros(ComplexF64, 4)
initial_state[1] = 1.0  # |00⟩

# Define the circuit
circuit = [(H, 1)]  # Apply H to qubit 1
```

```@example sv
# Run the first part of the circuit
ψ_after_h = simulate_circuit(initial_state, circuit)
```

```@example sv
# Now apply CNOT manually (since we need a specialized function)
bell_state = apply_cnot(ψ_after_h, 1, 0)
```

```@example sv
# Verify this is a Bell state
println("Bell state amplitudes:")
for (i, amp) in enumerate(bell_state)
    if abs(amp) > 1e-10
        println("State |$(bitstring(i-1)[end-1:end])⟩: ", amp)
    end
end
```

## Performance profiling

Let's profile our optimized gate application to see where time is spent:

```@example sv
# Profile the optimized gate application
Profile.clear()
@profile for i in 1:1000
    apply_gate_shifting_linear(ψ_large, X, 4)
end
```

```@example sv
# Display profiling results
Profile.print()
```

## Final thoughts

This tutorial has shown how to implement efficient quantum state vector simulation in Julia. Key takeaways:

1. **Avoid creating large matrices**: Use bit manipulation instead of tensor products
2. **Vectorize operations**: Process data in chunks when possible
3. **Use threading**: For large systems, parallel processing can provide significant speedups
4. **Profile your code**: Use Julia's profiling tools to identify bottlenecks
5. **Consider specialized algorithms**: For sampling, use techniques like the alias method

The optimized implementations we've developed can handle reasonably large quantum systems (up to ~20-25 qubits) efficiently, making them suitable for many quantum algorithm simulations and educational purposes.

For production quantum simulation, you would typically use specialized libraries like `Yao.jl` or `PennyLane.jl`, but understanding these fundamental algorithms helps you use such libraries more effectively and debug performance issues.
