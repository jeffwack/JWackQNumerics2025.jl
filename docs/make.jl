using JWackQNumerics2025
using Documenter

DocMeta.setdocmeta!(JWackQNumerics2025, :DocTestSetup, :(using JWackQNumerics2025); recursive=true)

makedocs(;
    modules=[JWackQNumerics2025],
    authors="Jeffrey Wack <jeffwack111@gmail.com> and contributors",
    sitename="JWackQNumerics2025.jl",
    format=Documenter.HTML(;
        canonical="https://jeffwack.github.io/JWackQNumerics2025.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Creating this package" => "creation.md",
        "Monday" => "monday.md",
        "Tuesday morning" => "tuesday.md",
        "Tuesday afternoon" => "tuesday_afternoon.md",
    ],
)

deploydocs(;
    repo="github.com/jeffwack/JWackQNumerics2025.jl",
    devbranch="main",
)
