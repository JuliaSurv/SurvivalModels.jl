using TestItemRunner

@run_package_tests

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(
        SurvivalModels;
        ambiguities=false,
    )
end