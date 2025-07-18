#!/usr/bin/env julia

# Script to test individual file build times

files_to_test = [
    "1_intro_to_julia.md",
    "2_introduction_to_state_vector_simulation.md", 
    "3_dynamics_and_quantumoptics.md",
    "4_quantum_trajectories.md",
    "5_optimization_methods.md",
    "6_grape_demos.md",
    "7_trajectory_optimization.md",
    "8_nonlinear_trajectory_optimization.md",
    "9_quantum_trajectory_optimization.md",
    "10_clifford.md"
]

# Store results
results = []

# Base make.jl template
base_makejl = """using QNumerics2025
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
        "TEST_FILE" => "REPLACE_FILE",
    ],
)

deploydocs(;
    repo="github.com/jeffwack/QNumerics2025.jl",
    devbranch="main",
)
"""

for file in files_to_test
    println("Testing file: $file")
    
    # Move file from temp to src
    run(`mv temp_files/$file src/`)
    
    # Create custom make.jl
    custom_makejl = replace(replace(base_makejl, "REPLACE_FILE" => file), "TEST_FILE" => splitext(file)[1])
    write("make.jl", custom_makejl)
    
    # Time the build
    start_time = time()
    try
        run(`julia --project=. make.jl`)
        end_time = time()
        build_time = end_time - start_time
        push!(results, (file, build_time, "success"))
        println("  Build time: $(round(build_time, digits=2)) seconds")
    catch e
        end_time = time()
        build_time = end_time - start_time
        push!(results, (file, build_time, "failed"))
        println("  Build failed after: $(round(build_time, digits=2)) seconds")
    end
    
    # Move file back to temp
    run(`mv src/$file temp_files/`)
end

# Print results table
println("\n" * "="^60)
println("BUILD TIME RESULTS")
println("="^60)
println("File Name                                    | Time (s) | Status")
println("-"^60)
for (file, time, status) in results
    file_padded = rpad(file, 40)
    time_str = rpad(string(round(time, digits=2)), 8)
    println("$file_padded | $time_str | $status")
end