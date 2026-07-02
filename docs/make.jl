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
    # Only require exported symbols to be documented, so internal helpers
    # (e.g. `_initial_baseline_log_params`) can keep their source docstrings
    # without being surfaced in the public docs.
    checkdocs=:exports,
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
            "Model classes" => "parametric/models.md",
            "Baseline hazards" => "parametric/baselines.md",
            "Fitting, prediction & simulation" => "parametric/fitting.md",
            "Illustrative example" => "parametric/example.md",
        ],
        "Case Study" => "case_study.md",
        "References" => "references.md",
        "Index" => "apiindex.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaSurv/SurvivalModels.jl",
    devbranch="main",
)
