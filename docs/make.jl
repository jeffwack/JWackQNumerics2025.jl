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
        "Julia and linear algebra" => "monday.md",
        "Applying operators to quantum states" => "tuesday_morning.md",
        "Schrodinger and Lindblad evolution" => "tuesday_afternoon.md",
        "Monte Carlo wave function evolution" => "tuesday_evening.md",
    ],
)

deploydocs(;
    repo="github.com/jeffwack/QNumerics2025.jl",
    devbranch="main",
)
