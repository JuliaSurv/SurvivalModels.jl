@testitem "Check Cox" begin

      # Required packages
      using Distributions, Random
      using StableRNGs
      using SurvivalModels: getβ, CoxV0, CoxV1, CoxV2, CoxV3, CoxV4, CoxV5
      rng = StableRNG(123)

      # Sample size
      n = 1000

      # Simulated design matrices
      dist = Normal()
      des = hcat(rand(rng, dist, n), rand(rng, dist, n))
      des_t = rand(rng, dist, n)

      # True parameters
      theta0 = [0.1,2.0,5.0]
      alpha0 = 0.5
      beta0 = [-0.5,0.75]

      # censoring
      cens = 10

      function qPGW(p, sigma, nu, gamma)
            val = sigma * ((1 .- log.(1 .- p)) .^ gamma .- 1) .^ (1 / nu)
            return val
        end

      function simPH(rng, n, des, theta, beta)
            #= Uniform variates =#
            distu = Uniform(0, 1)
            u = rand(rng, distu, n)
        
            #= quantile function =#
            function quantf(prob)
                  #= quantile value =#
                  sigma = theta[1]
                  nu = theta[2]
                  gamma = theta[3]
                  val = qPGW(prob, sigma, nu, gamma)
                return val
            end
            # Linear predictors
            exp_xalpha = 1.0
            exp_dif = exp.(-des * beta)
        
            # Simulating the times to event
            p0 = 1 .- exp.(log.(1 .- u) .* exp_dif)
            times = quantf.(p0) ./ exp_xalpha
        
            return times
        end
      # Data simulation
      simdat = simPH(rng, n, des, theta0, beta0)

      # status variable
      status = collect(Bool,(simdat .< cens))

      # Inducing censoring
      simdat = min.(simdat, cens)


      # Model fit
      for M in (CoxV0, CoxV1, CoxV2, CoxV3, CoxV4, CoxV5)
            β = getβ(M(simdat, status, des))
            @test β[1] ≈ -0.4926892848193542 atol=1e-2
            @test β[2] ≈ 0.6790626074990427 atol=1e-2
      end
end



@testitem "Check Cox on real data" begin

      # Required packages
      using Distributions, Random, RDatasets
      using SurvivalModels: getβ, CoxV0, CoxV1, CoxV2, CoxV3, CoxV4, CoxV5

      ovarian = dataset("survival","ovarian")

      for M in (CoxV0, CoxV1, CoxV2, CoxV3, CoxV4, CoxV5)
            β = fit(M, @formula(Surv(FUTime, FUStat) ~ Age + ECOG_PS), ovarian).β
            @test β[1] ≈ 0.16149 atol=1e-3
            @test β[2] ≈ 0.0187 atol=1e-3
      end

      colon = dataset("survival", "colon")
      colon.Time = Float64.(colon.Time)
      colon.Status = Bool.(colon.Status)
      model_colon = fit(Cox, @formula(Surv(Time, Status) ~ Age + Rx), colon)

      for M in (CoxV0, CoxV1, CoxV2, CoxV3, CoxV4, CoxV5)
            β = fit(M, @formula(Surv(Time, Status) ~ Age + Rx), colon).β
            @test β[1] ≈ -0.00205614 atol=1e-3
            @test β[2] ≈ -0.0200488	 atol=1e-3
            @test β[3] ≈ -0.439289   atol=1e-3
      end
end

@testitem "Verify baseline hazard " begin
    using DataFrames
    using SurvivalModels: baseline_hazard
    time = [1.0, 3.0, 5.0, 6.0, 2.0, 7.0, 9.0, 11.0] 
    status = [true, false, true, true, true, false, true, true] 
    sex = [1, 1, 1, 1, 0, 0, 0, 0]
    age = [57, 52, 48, 42, 39, 31, 26, 22]
    df = DataFrame(time = time, status = status, sex = sex, age = age)

    model = fit(Cox, @formula(Surv(time, status) ~ sex + age), df)
    result_false = baseline_hazard(model, centered = false)
    result_true = baseline_hazard(model, centered = true)
end


@testitem "Verify the correctness of the KaplanMeier implementation" begin
    using SurvivalModels: KaplanMeier

    # Example data: 5 subjects, 3 events, 2 censored
    # Times:    2, 3, 4, 5, 8
    # Status:   1, 1, 0, 1, 0
    T = Float64[2, 3, 4, 5, 8]
    Δ = Bool[1, 1, 0, 1, 0]

    km = KaplanMeier(T, Δ)

    @test km.t == [2, 3, 4, 5, 8]
    @test km.∂N == [1, 1, 0, 1, 0]
    @test km.Y == [5, 4, 3, 2, 1]

    S = [1.0]
    for i in 1:length(km.t)
        S_new = S[end] * (1 - km.∂Λ[i])
        push!(S, S_new)
    end
    S = S[2:end]
    @test isapprox(S[1], 0.8; atol=1e-8)
    @test isapprox(S[2], 0.6; atol=1e-8)
    @test isapprox(S[3], 0.6; atol=1e-8)
    @test isapprox(S[4], 0.3; atol=1e-8)
    @test isapprox(S[5], 0.3; atol=1e-8)

    # Now test with duplicate times (events and censored at same time)
    # Times:    2, 2, 3, 3, 4, 5, 8
    # Status:   1, 0, 1, 0, 0, 1, 0
    T2 = Float64[2, 2, 3, 3, 4, 5, 8]
    Δ2 = Bool[1, 0, 1, 0, 0, 1, 0]

    km2 = KaplanMeier(T2, Δ2)

    @test km2.t == [2, 3, 4, 5, 8]
    @test km2.∂N == [1, 1, 0, 1, 0]
    @test km2.Y == [7, 5, 3, 2, 1]

    # Survival probabilities:
    # S(2) = 1 * (1 - 1/7) = 6/7 ≈ 0.8571428571
    # S(3) = (6/7) * (1 - 1/5) = (6/7)*(4/5) = 24/35 ≈ 0.6857142857
    # S(4) = 24/35 (no event)
    # S(5) = (24/35) * (1 - 1/2) = (24/35)*(1/2) = 12/35 ≈ 0.3428571429
    # S(8) = 12/35 (no event)
    S2 = [1.0]
    for i in 1:length(km2.t)
        S_new = S2[end] * (1 - km2.∂Λ[i])
        push!(S2, S_new)
    end
    S2 = S2[2:end]
    @test isapprox(S2[1], 6/7; atol=1e-8)
    @test isapprox(S2[2], 24/35; atol=1e-8)
    @test isapprox(S2[3], 24/35; atol=1e-8)
    @test isapprox(S2[4], 12/35; atol=1e-8)
    @test isapprox(S2[5], 12/35; atol=1e-8)
end

@testitem "Verify the correctness of the LogRankTest implementation" begin
    using SurvivalModels: LogRankTest
    using Random

    # 1. Two groups, clear separation
    T = Float64[1, 2, 3, 4, 5, 6, 7, 8]
    Δ = Bool[1, 1, 1, 1, 1, 1, 1, 1]
    group = [1, 1, 1, 1, 2, 2, 2, 2]
    strata = ones(Int, 8)
    # Group 1: early events, Group 2: late events
    lrt = LogRankTest(T, Δ, group, strata)
    @test lrt.stat > 3.84  # Should be significant at p < 0.05
    @test lrt.pval < 0.05

    # 2. Two groups, identical data
    T2 = Float64[1, 2, 3, 4, 1, 2, 3, 4]
    Δ2 = Bool[1, 1, 1, 1, 1, 1, 1, 1]
    group2 = [1, 1, 1, 1, 2, 2, 2, 2]
    lrt2 = LogRankTest(T2, Δ2, group2, strata)
    @test lrt2.stat ≈ 0 atol=1e-8
    @test lrt2.pval ≈ 1 atol=1e-8

    # 3. Duplicate times, mix of events and censored
    T3 = Float64[2, 2, 3, 3, 4, 5, 8, 8]
    Δ3 = Bool[1, 0, 1, 0, 0, 1, 0, 1]
    group3 = [1, 1, 1, 2, 2, 2, 2, 2]
    lrt3 = LogRankTest(T3, Δ3, group3, strata)
    @test lrt3.stat ≥ 0
    @test 0.0 ≤ lrt3.pval ≤ 1.0

    # 4. All uncensored data 
    T4 = Float64[1, 2, 3, 4, 5, 6]
    Δ4 = trues(6)
    group4 = [1, 1, 1, 2, 2, 2]
    strata = ones(Int, 6)
    lrt4 = LogRankTest(T4, Δ4, group4, strata)
    @test lrt4.stat ≥ 0
    @test lrt4.pval ≈ 0 atol=1e-8

    # Two strata, two groups, identical within strata
    T = Float64[1, 2, 3, 4, 1, 2, 3, 4]
    Δ = Bool[1, 1, 1, 1, 1, 1, 1, 1]
    group = [1, 1, 2, 2, 2, 2, 1, 1]
    strata = [1, 1, 1, 1, 2, 2, 2, 2]

    lrt = LogRankTest(T, Δ, group, strata)
    @test lrt.stat ≈ 0 atol=1e-8
    @test lrt.pval ≈ 1 atol=1e-8

    # Now, make group 2 in stratum 2 have later events
    T2 = Float64[1, 2, 3, 4, 1, 2, 7, 8]
    lrt2 = LogRankTest(T2, Δ, group, strata)
    @test lrt2.stat > 0
    @test 0 < lrt2.pval < 1
end

@testitem "fit() interface matches direct constructor for KaplanMeier and LogRankTest" begin
    using SurvivalModels: KaplanMeier, LogRankTest
    using DataFrames, StatsModels

    # Data for KaplanMeier
    T = Float64[2., 3, 4, 5, 8]
    Δ = Bool[true, true, false, true, false]
    df = DataFrame(time=T, status=Δ)

    # Direct and fit interface for KaplanMeier
    km1 = KaplanMeier(T, Δ)
    km2 = fit(KaplanMeier, @formula(Surv(time, status) ~ 1), df)
    @test km1.t == km2.t
    @test km1.∂N == km2.∂N
    @test km1.Y == km2.Y
    @test all(isapprox.(km1.∂Λ, km2.∂Λ; atol=1e-12))

    # Data for LogRankTest
    T = Float64[1, 2, 3, 4, 1, 2, 3, 4]
    Δ = Bool[1, 1, 1, 1, 1, 1, 1, 1]
    group = [1, 1, 2, 2, 1, 1, 2, 2]
    strata = [1, 1, 1, 1, 2, 2, 2, 2]
    df2 = DataFrame(time=T, status=Δ, group=group, strata=strata)

    # Direct and fit interface for LogRankTest
    lrt1 = LogRankTest(T, Δ, group, strata)
    lrt2 = fit(LogRankTest, @formula(Surv(time, status) ~ Strata(strata) + group), df2)
    @test isapprox(lrt1.stat, lrt2.stat; atol=1e-8)
    @test lrt1.df == lrt2.df
    @test isapprox(lrt1.pval, lrt2.pval; atol=1e-8)
end

@testitem "GeneralHazardModel direct construction and simulation" begin
    using SurvivalModels, Distributions, Random
    using SurvivalModels: GeneralHazardModel, GHMethod, PHMethod, AFTMethod, AHMethod, simGH

    n = 1000
    Random.seed!(123)
    X1 = randn(n, 2)
    X2 = randn(n, 2)
    T = rand(Weibull(2, 1), n)
    Δ = rand(Bool, n)
    α = [0.1, -0.2]
    β = [0.5, 0.7]

    # Direct construction for GH
    model = GeneralHazardModel(GHMethod(), T, Δ, Weibull(2, 1), X1, X2, α, β)
    @test model isa GeneralHazardModel{GHMethod, Weibull{Float64}}

    # Direct construction for PH (X2, α unused)
    model_ph = GeneralHazardModel(PHMethod(), T, Δ, Weibull(2, 1), X1, zeros(n,0), zeros(0), β)
    @test model_ph isa GeneralHazardModel{PHMethod, Weibull{Float64}}

    # Direct construction for AFT (X2, α unused)
    model_aft = GeneralHazardModel(AFTMethod(), T, Δ, Weibull(2, 1), X1, zeros(n,0), zeros(0), β)
    @test model_aft isa GeneralHazardModel{AFTMethod, Weibull{Float64}}

    # Direct construction for AH (X1, β unused)
    model_ah = GeneralHazardModel(AHMethod(), T, Δ, Weibull(2, 1), zeros(n,0), X2, α, zeros(0))
    @test model_ah isa GeneralHazardModel{AHMethod, Weibull{Float64}}

    # Simulation
    simdat = simGH(n, model)
    @test length(simdat) == n
    @test all(simdat .> 0)
end

@testitem "GeneralHazardModel fit interface matches direct construction (simple case)" begin
    using SurvivalModels, Distributions, DataFrames, StatsModels, Random
    using SurvivalModels: GeneralHazardModel, GHMethod, PHMethod, AFTMethod, AHMethod, simGH

    n = 100
    Random.seed!(42)
    X1 = randn(n, 1)
    X2 = randn(n, 1)
    T = rand(Weibull(2, 1), n)
    Δ = trues(n)

    α = [0.0]
    β = [0.0]

    # Direct construction
    model = GeneralHazardModel(GHMethod(), T, Δ, Weibull(2, 1), X1, X2, α, β)
    @test model isa GeneralHazardModel{GHMethod, Weibull{Float64}}

    # Fit interface (should not error, but will not recover true params for random data)
    df = DataFrame(time=T, status=Δ, x1=X1[:,1], x2=X2[:,1])
    fitted = fit(GeneralHazard{Weibull}, @formula(Surv(time, status) ~ x1), @formula(Surv(time, status) ~ x2), df)
    @test fitted isa GeneralHazardModel{GHMethod, Weibull{Float64}}
    @test length(fitted.α) == 1
    @test length(fitted.β) == 1
end

@testitem "GeneralHazardModel fit interface for PH/AFT/AH" begin
    using SurvivalModels, Distributions, DataFrames, StatsModels, Random
    using SurvivalModels: GeneralHazardModel, GHMethod, PHMethod, AFTMethod, AHMethod, simGH

    n = 100
    Random.seed!(42)
    X = randn(n, 2)
    T = rand(Weibull(2, 1), n)
    Δ = trues(n)
    df = DataFrame(time=T, status=Δ, x1=X[:,1], x2=X[:,2])

    # PH
    model_ph = fit(ProportionalHazard{Weibull}, @formula(Surv(time, status) ~ x1 + x2), df)
    @test model_ph isa GeneralHazardModel{PHMethod, Weibull{Float64}}
    @test length(model_ph.β) == 2

    # AFT
    model_aft = fit(AcceleratedFaillureTime{Weibull}, @formula(Surv(time, status) ~ x1 + x2), df)
    @test model_aft isa GeneralHazardModel{AFTMethod, Weibull{Float64}}
    @test length(model_aft.β) == 2

    # AH
    model_ah = fit(AcceleratedHazard{Weibull}, @formula(Surv(time, status) ~ x1 + x2), df)
    @test model_ah isa GeneralHazardModel{AHMethod, Weibull{Float64}}
    @test length(model_ah.α) == 2
end