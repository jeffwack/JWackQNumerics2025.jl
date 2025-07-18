using QNumerics2025
using Documenter

DocMeta.setdocmeta!(QNumerics2025, :DocTestSetup, :(using QNumerics2025); recursive=true)

makedocs(;
    modules=[QNumerics2025],
    authors="Jeffrey Wack <jeffwack111@gmail.com> and contributors",
    sitename="QNumerics2025.jl",
    format=Documenter.HTML(;
        canonical="https://jeffwack.github.io/QNumerics2025.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Creating this package" => "creation.md",
        "Intro to Julia" => "1_intro_to_julia.md",
        "Introduction to State Vector Simulation" => "2_introduction_to_state_vector_simulation.md",
        "Dynamics and QuantumOptics.jl" => "3_dynamics_and_quantumoptics.md",
        "Quantum Trajectories" => "4_quantum_trajectories.md",
        "Optimization Methods" => "5_optimization_methods.md",
        "GRAPE Demos" => "6_grape_demos.md",
        "Trajectory Optimization" => "7_trajectory_optimization.md",
        "Nonlinear Trajectory Optimization" => "8_nonlinear_trajectory_optimization.md",
        "Quantum Trajectory Optimization" => "9_quantum_trajectory_optimization.md",
        "Clifford" => "10_clifford.md",
    ],
)

deploydocs(;
    repo="github.com/jeffwack/QNumerics2025.jl",
    devbranch="main",
)