using SurvivalModels
using Test
using Aqua

@testset "SurvivalModels.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SurvivalModels)
    end
    # Write your tests here.
end
