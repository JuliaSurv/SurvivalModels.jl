using SurvivalModels
using Documenter

DocMeta.setdocmeta!(SurvivalModels, :DocTestSetup, :(using SurvivalModels); recursive=true)

makedocs(;
    modules=[SurvivalModels],
    authors="Oskar Laverny <oskar.laverny@univ-amu.fr> and contributors",
    sitename="SurvivalModels.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaSurv.github.io/SurvivalModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaSurv/SurvivalModels.jl",
    devbranch="main",
)
