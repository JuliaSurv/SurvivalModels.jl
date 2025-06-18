using SurvivalModels
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(SurvivalModels, :DocTestSetup, :(using SurvivalModels); recursive=true)

bib = CitationBibliography(
    joinpath(@__DIR__,"src","assets","references.bib"),
    style=:numeric
)


makedocs(;
    modules=[SurvivalModels],
    plugins=[bib],
    authors="Oskar Laverny <oskar.laverny@univ-amu.fr> and contributors",
    sitename="SurvivalModels.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaSurv.github.io/SurvivalModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "cox.md",
        "references.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaSurv/SurvivalModels.jl",
    devbranch="main",
)
