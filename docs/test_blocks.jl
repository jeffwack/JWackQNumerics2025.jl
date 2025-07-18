#!/usr/bin/env julia

using Optim
using LinearAlgebra
using SparseArrays
using CairoMakie
using Piccolo

const ⊗ = kron

println("Testing trajectory optimization blocks individually...")

# Block 1: Basic setup
println("\n=== Block 1: Basic setup ===")
@time begin
    N = 10
    traj = rand(NamedTrajectory, N)
end

# Block 2: LTI system setup
println("\n=== Block 2: LTI system setup ===")
@time begin
    # Define continuous LTI system matrices
    A = [0.0 1.0; -1.0 -0.1]
    B = [0.0; 1.0]
    h = 0.1

    function continuous_to_discrete(A, B, h)
        augmented_matrix = [
            A B; 
            zeros(size(B, 2), size(A, 1)) zeros(size(B, 2), size(B, 2))
        ]
        exp_matrix = exp(augmented_matrix * h)
        A_h = exp_matrix[1:size(A, 1), 1:size(A, 2)]
        B_h = exp_matrix[1:size(A, 1), size(A, 2)+1:end]
        return A_h, B_h
    end

    A_h, B_h = continuous_to_discrete(A, B, h)
end

# Block 3: Double integrator simulation setup
println("\n=== Block 3: Double integrator setup ===")
@time begin
    function double_integrator(m)
        A_c = [0.0 1.0; 0.0 0.0]
        B_c = [0.0; 1.0 / m]
        return A_c, B_c
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

    function simulate_dlti(u::AbstractVector, A, B, x1)
        simulate_dlti(reshape(u, 1, length(u)), A, B, x1)
    end

    m = 2.0
    A_c, B_c = double_integrator(m)
    h = 0.05
    A, B = continuous_to_discrete(A_c, B_c, h)
end

# Block 4: Cost function setup
println("\n=== Block 4: Cost function setup ===")
@time begin
    function J(x::AbstractMatrix, u::AbstractVecOrMat; Q = 1e-2I, R = 1e-4I, QN = 1e2I)
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

    function J(u::AbstractVecOrMat; A=A, B=B, x1=[1.0; 2.0], kwargs...)
        x = simulate_dlti(u, A, B, x1)
        return J(x, u; kwargs...)
    end
end

# Block 5: Gradient descent optimization (POTENTIALLY EXPENSIVE)
println("\n=== Block 5: Gradient descent optimization ===")
@time begin
    m = 0.1
    A_c, B_c = double_integrator(m)
    h = 0.1
    A, B = continuous_to_discrete(A_c, B_c, h)
    x1 = [1.0; 2.0]
    
    Q = 1e-4I
    R = 1e-1I
    QN = 1e2I
    
    N = 20  # Reduced from 40
    u0 = randn(N - 1)
    res = optimize(u -> J(u; A=A, B=B, x1=x1, Q=Q, R=R, QN=QN), u0, GradientDescent(), 
                   Optim.Options(iterations=10))  # Limited iterations
end

# Block 6: Newton optimization (POTENTIALLY EXPENSIVE)
println("\n=== Block 6: Newton optimization ===")
@time begin
    res_newton = optimize(u -> J(u; A=A, B=B, x1=x1, Q=Q, R=R, QN=QN), u0, Newton(),
                         Optim.Options(iterations=5))  # Limited iterations
end

# Block 7: Pontryagin method (POTENTIALLY EXPENSIVE)
println("\n=== Block 7: Pontryagin method ===")
@time begin
    N = 50  # Reduced from 100
    m = 1.0
    A_c, B_c = double_integrator(m)
    h = 0.1
    A, B = continuous_to_discrete(A_c, B_c, h)
    x1 = [1.0; 1.0]
    
    Q = 1e-4I
    R = 1e-2I
    QN = 1e1I
    
    u = zeros(1, N - 1)
    Δu = ones(1, N - 1)
    x = simulate_dlti(u, A, B, x1)
    λ = zeros(size(x, 1), N)
    
    α = 1.0
    b = 1e-2
    α_min = 1e-16
    loop = 0
    max_iter = 5  # Reduced iterations
    
    while maximum(abs, Δu) > 1e-2 && α > α_min && loop < max_iter
        λ[:, N] .= QN * x[:, N]
        for n = N-1:-1:1
            Δu[:, n] .= - R\B' * λ[:, n+1] - u[:, n]
            λ[:, n] .= Q * x[:, n] + A' * λ[:, n+1]
        end
        
        α = 1.0
        unew = u + α .* Δu
        xnew = simulate_dlti(unew, A, B, x1)
        
        while J(xnew, unew) > J(x, u) - b * α * norm(Δu)^2
            α = 0.5 * α
            unew = u + α .* Δu
            xnew = simulate_dlti(unew, A, B, x1)
            if α < α_min; break; end
        end
        
        u .= unew
        x .= xnew
        loop += 1
    end
end

# Block 8: Direct QP method (POTENTIALLY EXPENSIVE)
println("\n=== Block 8: Direct QP method ===")
@time begin
    m = 0.1
    A_c, B_c = double_integrator(m)
    h = 0.1
    A, B = sparse.(continuous_to_discrete(A_c, B_c, h))
    x1 = [1.0; 1.0]
    
    N = 50  # Reduced from 100
    x_dim = size(A, 2)
    u_dim = size(B, 2)
    
    Q = sparse(1e-4I(x_dim))
    R = sparse(1e-2I(u_dim))
    QN = sparse(1e2I(x_dim))
    
    z_dim = (N-2) * (x_dim + u_dim) + u_dim + x_dim
    H = blockdiag(sparse(R), I(N-2) ⊗ blockdiag(sparse(Q), sparse(R)), sparse(QN))
    
    C = I(N-1) ⊗ [B -I(x_dim)]
    for k = 1:N-2
        C[(k * x_dim) .+ (1:x_dim), (k * (x_dim + u_dim) - x_dim) .+ (1:x_dim)] = A
    end
    
    d = [-A * x1; zeros((N-2) * x_dim)]
    P = [H C'; C zeros(size(C, 1), size(C, 1))]
    res_qp = P \ [zeros(z_dim); d]
end

println("\n=== Testing complete ===")