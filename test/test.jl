
@testitem "Check Cox" begin


      # Required packages
      using Distributions, Random, Optim

      # Sample size
      n = 1000

      # Simulated design matrices
      Random.seed!(123)
      dist = Normal()
      des = hcat(rand(dist, n), rand(dist, n))
      des_t = rand(dist, n)

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

      function simPH(seed, n, des, theta, beta)
            #= Uniform variates =#
            Random.seed!(seed)
            distu = Uniform(0, 1)
            u = rand(distu, n)
        
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
      simdat = simPH(1234, n, des, theta0, beta0)

      # status variable
      status = collect(Bool,(simdat .< cens))

      # Inducing censoring
      simdat = min.(simdat, cens)


      # Model fit

      OPTCox = CoxModel(
            fill(0.0, size(des)[2]), 
            simdat,
            status,  
            des,  
            NelderMead(),
            1000
      )

      betahat = OPTCox.par
      @test betahat[1] ≈ -0.4874388584969206
      @test betahat[2] ≈ 0.7626546774084827

      # 95% Confidence intervals under the reparameterisation
      # CI = ConfInt(FUN = OPTCox[2], MLE = OPTCox[1].minimizer, level = 0.95)

      # CI = DataFrame(CI, :auto)
      
      # rename!( CI, ["Lower", "Upper"] )

      # println(CI)


end



@testitem "Check Cox on real data" begin


      # Required packages
      using Distributions, Random, Optim, RDatasets

      ovarian = dataset("survival","ovarian")

      OPTCox = fit(CoxModel, @formula(Surv(FUTime, FUStat) ~ Age + ECOG_PS), ovarian, NelderMead(), 1000)

      betahat = OPTCox.par
      @test betahat[1] ≈ 0.16149 atol=1e-3
      @test betahat[2] ≈ 0.0187 atol=1e-3

end