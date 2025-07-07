
@testitem "Check Cox" begin


      # Required packages
      using Distributions, Random, Optim
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
      using Distributions, Random, Optim, RDatasets
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