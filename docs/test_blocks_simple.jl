#!/usr/bin/env julia

using Optim
using LinearAlgebra
using SparseArrays
using CairoMakie
using Piccolo

const ⊗ = kron

println("Testing trajectory optimization blocks individually...")

# Simple function definitions
function double_integrator(m)
    A_c = [0.0 1.0; 0.0 0.0]
    B_c = [0.0; 1.0 / m]
    return A_c, B_c
end

function continuous_to_discrete(A, B, h)
    augmented_matrix = [A B; zeros(size(B, 2), size(A, 1)) zeros(size(B, 2), size(B, 2))]
    exp_matrix = exp(augmented_matrix * h)
    A_h = exp_matrix[1:size(A, 1), 1:size(A, 2)]
    B_h = exp_matrix[1:size(A, 1), size(A, 2)+1:end]
    return A_h, B_h
end

function simulate_dlti(u::AbstractMatrix, A, B, x1)
    N = size(u, 2) + 1
    x = zeros(size(A, 2), N)
    x[:, 1] = x1
    for k in 1:N-1
        x[:, k + 1] = A * x[:, k] + B * u[:, k]
    end
    return x
end

simulate_dlti(u::AbstractVector, A, B, x1) = simulate_dlti(reshape(u, 1, length(u)), A, B, x1)

function J_cost(x::AbstractMatrix, u::AbstractVecOrMat; Q = 1e-2I, R = 1e-4I, QN = 1e2I)
    u = isa(u, AbstractMatrix) ? u : reshape(u, 1, length(u))
    N = size(u, 2) + 1    
    cost = 0.0
    for n in 1:N-1
        xₙ = x[:, n]
        uₙ = u[:, n]
        cost += 1/2 * (xₙ' * Q * xₙ + uₙ' * R * uₙ)
    end
    cost += 1/2 * (x[:, N]' * QN * x[:, N])
    return cost
end

function J_cost(u::AbstractVecOrMat; A, B, x1, kwargs...)
    x = simulate_dlti(u, A, B, x1)
    return J_cost(x, u; kwargs...)
end

# Test individual expensive blocks
println("\n=== Testing optimization blocks ===")

# Setup
m = 0.1
A_c, B_c = double_integrator(m)
h = 0.1
A, B = continuous_to_discrete(A_c, B_c, h)
x1 = [1.0; 2.0]
Q = 1e-4I
R = 1e-1I
QN = 1e2I

println("Gradient descent with N=10:")
@time begin
    N = 10
    u0 = randn(N - 1)
    res = optimize(u -> J_cost(u; A=A, B=B, x1=x1, Q=Q, R=R, QN=QN), u0, GradientDescent(), 
                   Optim.Options(iterations=5))
end

println("Gradient descent with N=40:")
@time begin
    N = 40
    u0 = randn(N - 1)
    res = optimize(u -> J_cost(u; A=A, B=B, x1=x1, Q=Q, R=R, QN=QN), u0, GradientDescent(), 
                   Optim.Options(iterations=5))
end

println("Newton with N=10:")
@time begin
    N = 10
    u0 = randn(N - 1)
    res = optimize(u -> J_cost(u; A=A, B=B, x1=x1, Q=Q, R=R, QN=QN), u0, Newton(),
                   Optim.Options(iterations=3))
end

println("QP method with N=20:")
@time begin
    N = 20
    x_dim = size(A, 2)
    u_dim = size(B, 2)
    z_dim = (N-2) * (x_dim + u_dim) + u_dim + x_dim
    
    A_sparse, B_sparse = sparse(A), sparse(B)
    Q_sparse = sparse(1e-4I(x_dim))
    R_sparse = sparse(1e-2I(u_dim))
    QN_sparse = sparse(1e2I(x_dim))
    
    H = blockdiag(sparse(R_sparse), I(N-2) ⊗ blockdiag(sparse(Q_sparse), sparse(R_sparse)), sparse(QN_sparse))
    C = I(N-1) ⊗ [B_sparse -I(x_dim)]
    for k = 1:N-2
        C[(k * x_dim) .+ (1:x_dim), (k * (x_dim + u_dim) - x_dim) .+ (1:x_dim)] = A_sparse
    end
    d = [-A_sparse * x1; zeros((N-2) * x_dim)]
    P = [H C'; C zeros(size(C, 1), size(C, 1))]
    res_qp = P \ [zeros(z_dim); d]
end

println("QP method with N=100:")
@time begin
    N = 100
    x_dim = size(A, 2)
    u_dim = size(B, 2)
    z_dim = (N-2) * (x_dim + u_dim) + u_dim + x_dim
    
    A_sparse, B_sparse = sparse(A), sparse(B)
    Q_sparse = sparse(1e-4I(x_dim))
    R_sparse = sparse(1e-2I(u_dim))
    QN_sparse = sparse(1e2I(x_dim))
    
    H = blockdiag(sparse(R_sparse), I(N-2) ⊗ blockdiag(sparse(Q_sparse), sparse(R_sparse)), sparse(QN_sparse))
    C = I(N-1) ⊗ [B_sparse -I(x_dim)]
    for k = 1:N-2
        C[(k * x_dim) .+ (1:x_dim), (k * (x_dim + u_dim) - x_dim) .+ (1:x_dim)] = A_sparse
    end
    d = [-A_sparse * x1; zeros((N-2) * x_dim)]
    P = [H C'; C zeros(size(C, 1), size(C, 1))]
    res_qp = P \ [zeros(z_dim); d]
end

println("\n=== Testing complete ===")