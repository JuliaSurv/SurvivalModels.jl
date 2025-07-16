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
        assets=String["assets/citations.css"],
    ),
    pages=[
        "Home" => "index.md",
        "Nonparametric" => [
            "Kaplan-Meier" => "nonparametric/kaplanmeier.md",
            "Log-Rank Test" => "nonparametric/logranktest.md",
        ],
        "Semiparametric" => [
            "Cox" => "semiparametric/cox.md",
        ],
        "Parametric" => [
            "General Hazard" => "parametric/generalhazard.md",
            # Add parametric models here as you implement them
            # "General Hazard" => "parametric/generalhazard.md",
        ],
        "Case Study" => "case_study.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaSurv/SurvivalModels.jl",
    devbranch="main",
)
