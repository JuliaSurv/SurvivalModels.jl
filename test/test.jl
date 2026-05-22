@testitem "Check Cox" begin

      # Required packages
      using Distributions, Random
      using StableRNGs
      using SurvivalModels: getβ, CoxNM, CoxOptim, CoxHessian, CoxDefault, CoxApprox
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
      for M in (CoxNM, CoxOptim, CoxHessian, CoxDefault, CoxApprox)
            β = getβ(M(simdat, status, des))
            @test β[1] ≈ -0.4926892848193542 atol=1e-2
            @test β[2] ≈ 0.6790626074990427 atol=1e-2
      end
end



@testitem "Check Cox on real data" begin

      # Required packages
      using Distributions, Random, RDatasets
      using SurvivalModels: getβ, CoxNM, CoxOptim, CoxHessian, CoxDefault, CoxApprox

      ovarian = dataset("survival","ovarian")

      for M in (CoxNM, CoxOptim, CoxHessian, CoxDefault, CoxApprox)
            β = fit(M, @formula(Surv(FUTime, FUStat) ~ Age + ECOG_PS), ovarian).β
            @test β[1] ≈ 0.16149 atol=1e-3
            @test β[2] ≈ 0.0187 atol=1e-3
      end

      colon = dataset("survival", "colon")
      colon.Time = Float64.(colon.Time)
      colon.Status = Bool.(colon.Status)
      model_colon = fit(Cox, @formula(Surv(Time, Status) ~ Age + Rx), colon)

      for M in (CoxNM, CoxOptim, CoxHessian, CoxDefault, CoxApprox)
            β = fit(M, @formula(Surv(Time, Status) ~ Age + Rx), colon).β
            @test β[1] ≈ -0.00205614 atol=1e-3
            @test β[2] ≈ -0.0200488	 atol=1e-3
            @test β[3] ≈ -0.439289   atol=1e-3
      end
end

@testitem "Verify baseline hazard " begin

    # This test is directly drawn from this web article : https://missingdatasolutions.rbind.io/2022/12/cox-baseline-hazard/

    using DataFrames
    using SurvivalModels: baseline_hazard, predict, CoxNM, CoxOptim, CoxHessian, CoxDefault, CoxApprox
    time = [1.0, 3.0, 5.0, 6.0, 2.0, 7.0, 9.0, 11.0] 
    status = [true, false, true, true, true, false, true, true] 
    sex = Symbol.([1, 1, 1, 1, 0, 0, 0, 0])
    age = [57, 52, 48, 42, 39, 31, 26, 22]

    df = DataFrame(time = time, status = status, sex = sex, age = age)

    f = @formula(Surv(time, status) ~ age + sex)
    models = (
        fit(CoxNM, f, df), 
        fit(CoxOptim, f, df), 
        fit(CoxHessian, f, df), 
        fit(CoxDefault, f, df), 
        fit(CoxApprox, f, df),
    )

    for model in models
        @test baseline_hazard(model, centered = false) ≈ [3.442456e-13, 5.942770e-12, 5.942770e-12, 1.096574e-10, 1.897298e-09, 1.897298e-09, 6.646862e-08, 9.459174e-07] rtol=1e-2
        @test baseline_hazard(model, centered = true) ≈ [2.780808e-02, 4.800556e-01, 4.800556e-01, 8.858100e+00, 1.532633e+02, 1.532633e+02, 5.369320e+03, 7.641099e+04] rtol=1e-2
        @test predict(model, :lp) ≈ [3.5189684, 0.3498842, -2.1853832, -5.9882842, -0.3961355, -5.4666703, -8.6357545, -11.1710219] rtol=1e-2
        @test predict(model, :risk) ≈ [3.374960e+01, 1.418903e+00, 1.124346e-01, 2.507963e-03, 6.729155e-01, 4.225278e-03, 1.776395e-04, 1.407625e-05] rtol=1e-2
        
        @test predict(model, :terms) ≈ [11.0125677 -7.493599
                                        7.8434834 -7.493599
                                        5.3082161 -7.493599
                                        1.5053150 -7.493599
                                        -0.3961355  0.000000
                                        -5.4666703  0.000000
                                        -8.6357545  0.000000
                                        -11.1710219  0.000000] rtol=1e-2

        # The no-arg form evaluates each subject at its own observed time Tᵢ — matches
        # R's `predict(coxph, type="expected")` / `type="survival"` convention.
        @test predict(model, :expected) ≈ [0.9385114, 0.6811525, 0.9959573, 0.3843788, 0.3230369, 0.6475801, 0.9538031, 1.0755799] rtol=1e-2
        @test predict(model, :survival) ≈ [0.3912098, 0.5060335, 0.3693697, 0.6808734, 0.7239472, 0.5233106, 0.3852730, 0.3410999] rtol=1e-2

    end
end


@testitem "Cox predict at arbitrary times" begin
    using DataFrames
    using SurvivalModels: baseline_hazard, predict, CoxNM, CoxOptim, CoxHessian, CoxDefault, CoxApprox

    # Same fixture as the baseline-hazard testitem above
    time   = [1.0, 3.0, 5.0, 6.0, 2.0, 7.0, 9.0, 11.0]
    status = [true, false, true, true, true, false, true, true]
    sex    = Symbol.([1, 1, 1, 1, 0, 0, 0, 0])
    age    = [57, 52, 48, 42, 39, 31, 26, 22]
    df     = DataFrame(time = time, status = status, sex = sex, age = age)
    f      = @formula(Surv(time, status) ~ age + sex)
    n      = length(time)

    for M in (CoxNM, CoxOptim, CoxHessian, CoxDefault, CoxApprox)
        model = fit(M, f, df)

        # Scalar t: length-n vector
        @test size(predict(model, :expected, 5.0))  == (n,)
        @test size(predict(model, :survival, 5.0))  == (n,)

        # Vector ts: n × length(ts) matrix
        ts = [1.0, 5.0, 10.0]
        @test size(predict(model, :expected, ts))   == (n, length(ts))
        @test size(predict(model, :survival, ts))   == (n, length(ts))

        # Self-consistency: the no-arg form equals predict(:expected, Tᵢ) at each subject
        no_arg = predict(model, :expected)
        for i in 1:n
            @test no_arg[i] ≈ predict(model, :expected, time[i])[i] rtol=1e-10
        end

        # Survival probabilities are in [0, 1] and monotone non-increasing in t
        S = predict(model, :survival, [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        @test all(0 .≤ S .≤ 1 .+ 1e-12)
        @test all(diff(S, dims = 2) .≤ 1e-12)

        # S(t < min(T)) ≈ 1; S(t = 0) ≈ 1
        @test predict(model, :survival, 0.0)    ≈ ones(n) atol=1e-10
        @test predict(model, :survival, 1e-6)   ≈ ones(n) atol=1e-10

        # Beyond max(T) plateaus at S(max(T))
        @test predict(model, :survival, 1e6)    ≈ predict(model, :survival, maximum(time)) rtol=1e-10

        # Misuse on a non-time-indexed prediction type errors loudly
        @test_throws ErrorException predict(model, :lp,    5.0)
        @test_throws ErrorException predict(model, :risk,  5.0)
        @test_throws ErrorException predict(model, :terms, 5.0)
    end
end


@testitem "Cox predict on newdata" begin
    using DataFrames
    using SurvivalModels: predict, CoxNM, CoxOptim, CoxHessian, CoxDefault, CoxApprox

    time   = [1.0, 3.0, 5.0, 6.0, 2.0, 7.0, 9.0, 11.0]
    status = [true, false, true, true, true, false, true, true]
    sex    = Symbol.([1, 1, 1, 1, 0, 0, 0, 0])
    age    = [57, 52, 48, 42, 39, 31, 26, 22]
    df     = DataFrame(time = time, status = status, sex = sex, age = age)
    f      = @formula(Surv(time, status) ~ age + sex)
    n      = length(time)

    for M in (CoxNM, CoxOptim, CoxHessian, CoxDefault, CoxApprox)
        model = fit(M, f, df)

        # Self-consistency: passing the training df via the newdata path equals the
        # no-newdata path that uses the model's internal X.
        @test predict(model, :lp,    df) ≈ predict(model, :lp)         rtol = 1e-10
        @test predict(model, :risk,  df) ≈ predict(model, :risk)       rtol = 1e-10
        @test predict(model, :terms, df) ≈ predict(model, :terms)      rtol = 1e-10
        @test predict(model, :expected, df, 1.0) ≈ predict(model, :expected, 1.0) rtol = 1e-10
        @test predict(model, :survival, df, [0.5, 1.0]) ≈ predict(model, :survival, [0.5, 1.0]) rtol = 1e-10

        # Held-out slice shapes
        held = df[1:3, :]
        @test size(predict(model, :lp,    held)) == (3,)
        @test size(predict(model, :risk,  held)) == (3,)
        @test size(predict(model, :terms, held)) == (3, 2)
        @test size(predict(model, :expected, held, 5.0))           == (3,)
        @test size(predict(model, :survival, held, [1.0, 5.0]))    == (3, 2)

        # Survival on newdata is in [0,1] and non-increasing in t (per subject)
        S = predict(model, :survival, held, [0.5, 1.0, 5.0, 11.0])
        @test all(0 .≤ S .≤ 1 .+ 1e-12)
        @test all(diff(S, dims = 2) .≤ 1e-12)

        # Newdata predict requires time for :expected and :survival
        @test_throws ErrorException predict(model, :expected, held)
        @test_throws ErrorException predict(model, :survival, held)

        # Newdata predict rejects time for :lp / :risk / :terms
        @test_throws ErrorException predict(model, :lp,    held, 1.0)
        @test_throws ErrorException predict(model, :risk,  held, 1.0)
        @test_throws ErrorException predict(model, :terms, held, 1.0)
    end
end


@testitem "Cox predict matches R survival package on ovarian" begin
    # Cross-check Julia's `predict(model, type, ...)` against R's
    # `predict(coxph, ...)` and `summary(survfit(coxph, newdata=...), times=...)`
    # on the ovarian fixture. R values were generated by:
    #
    #   library(survival); data(ovarian)
    #   fit <- coxph(Surv(futime, fustat) ~ age + ecog.ps, data = ovarian, ties = "breslow")
    #   options(digits = 12)
    #   idx <- c(1, 5, 10); nd <- ovarian[idx, ]
    #   predict(fit, newdata = nd, type = "lp")
    #   predict(fit, newdata = nd, type = "risk")
    #   summary(survfit(fit, newdata = nd), times = c(100, 300, 600), extend = TRUE)$surv
    using DataFrames, RDatasets
    using SurvivalModels: predict

    ovarian = dataset("survival", "ovarian")
    model = fit(Cox, @formula(Surv(FUTime, FUStat) ~ Age + ECOG_PS), ovarian)
    idx = [1, 5, 10]
    nd  = ovarian[idx, :]

    # type = "lp" (centered linear predictor) — both no-newdata and newdata paths
    R_lp = [2.60222487953, -0.949477658484, -0.149408293638]
    @test predict(model, :lp)[idx] ≈ R_lp        rtol = 1e-6
    @test predict(model, :lp, nd)  ≈ R_lp        rtol = 1e-6

    # type = "risk" = exp(lp)
    R_risk = [13.4937265783, 0.386943087116, 0.861217413514]
    @test predict(model, :risk)[idx] ≈ R_risk    rtol = 1e-6
    @test predict(model, :risk, nd)  ≈ R_risk    rtol = 1e-6

    # Survival at user times via `predict(model, :survival, newdata, ts)`.
    # R's `summary(survfit, times=ts)$surv` is a `length(ts) × n` matrix; Julia
    # returns `n × length(ts)`, so the reference matrix below is R's transposed.
    R_surv = [0.849678059028  0.395775461891  0.000944668070061  ;  # subject 1
              0.995339673559  0.973770320606  0.81896206151      ;  # subject 5
              0.989657147402  0.942557299342  0.641138135476     ]  # subject 10

    ts = [100.0, 300.0, 600.0]
    @test predict(model, :survival, nd, ts)         ≈ R_surv         rtol = 1e-5
    # Training-data path on the same three subjects, same times
    @test predict(model, :survival, ts)[idx, :]     ≈ R_surv         rtol = 1e-5

    # Scalar-t forms match the corresponding column of the matrix.
    @test predict(model, :survival, nd, 100.0)      ≈ R_surv[:, 1]   rtol = 1e-5
    @test predict(model, :survival, nd, 300.0)      ≈ R_surv[:, 2]   rtol = 1e-5
    @test predict(model, :survival, nd, 600.0)      ≈ R_surv[:, 3]   atol = 1e-6
end


@testitem "Cox newdata predict errors without stored formula" begin
    using DataFrames
    using SurvivalModels: predict, CoxDefault, Cox

    time   = [1.0, 3.0, 5.0, 6.0, 2.0, 7.0, 9.0, 11.0]
    status = [true, false, true, true, true, false, true, true]
    sex    = Symbol.([1, 1, 1, 1, 0, 0, 0, 0])
    age    = [57, 52, 48, 42, 39, 31, 26, 22]
    df     = DataFrame(time = time, status = status, sex = sex, age = age)

    # Construct a Cox without a formula via the public constructor (backwards-compatible path)
    # — simulate a direct-constructor user who didn't pass `formula = ...`.
    X = hcat(age, [s == Symbol(1) ? 1.0 : 0.0 for s in sex])
    worker = CoxDefault(time, Bool.(status), X)
    bare = Cox(worker, [:age, :sex], [:continuous, :categorical])

    @test_throws ErrorException predict(bare, :lp, df)
end


@testitem "Cox handles multi-level categorical predictors" begin
    # Regression test for the BoundsError reported in #52: `predict_lp` and
    # `predict_terms` index `pred_types` column-wise, but fit previously built
    # it term-wise — so any k-level categorical with k ≥ 3 broke every predict
    # path that goes through the centring loop.
    using DataFrames, Random
    using SurvivalModels: predict, nvar

    rng = MersenneTwister(123)
    n   = 50
    df  = DataFrame(
        time   = rand(rng, n) .* 10 .+ 0.1,
        status = rand(rng, Bool, n),
        age    = randn(rng, n),
        rx     = Symbol.(rand(rng, [:a, :b, :c], n)),   # 3-level categorical
    )
    model = fit(Cox, @formula(Surv(time, status) ~ age + rx), df)

    # The invariant: pred_types is per-design-column, same length as pred_names
    # and `nvar(C.M)`. With reference encoding on a 3-level factor we expect
    # `[:continuous, :categorical, :categorical]`.
    @test length(model.pred_types) == length(model.pred_names)
    @test length(model.pred_types) == nvar(model.M)
    @test model.pred_types == [:continuous, :categorical, :categorical]

    # Every prediction mode runs without BoundsError and returns the expected
    # shape. The actual numerical values are exercised by other testitems on
    # simpler fixtures.
    @test size(predict(model, :lp))                 == (n,)
    @test size(predict(model, :risk))               == (n,)
    @test size(predict(model, :terms))              == (n, nvar(model.M))
    @test size(predict(model, :expected))           == (n,)
    @test size(predict(model, :survival))           == (n,)
    @test size(predict(model, :expected, 5.0))      == (n,)
    @test size(predict(model, :survival, [1.0, 5.0])) == (n, 2)

    # Newdata path also exercises predict_lp through _build_X_for_newdata.
    @test size(predict(model, :lp, df)) == (n,)
end


@testitem "Cox predict on multi-level categorical matches R (colon)" begin
    # R cross-check for the fix in #53. The `colon` fixture's `rx` column is a
    # 3-level factor — previously the design-matrix-column / formula-term length
    # mismatch crashed every predict path that goes through `predict_lp`'s
    # centring loop. With #53 in place, the predict outputs are well-defined and
    # this testitem pins them against R's `coxph` reference.
    #
    # R values from:
    #   library(survival); data(colon)
    #   fit <- coxph(Surv(time, status) ~ age + rx, data = colon, ties = "breslow")
    #   options(digits = 12)
    #   idx <- c(1, 50, 200, 800); nd <- colon[idx, ]
    #   coef(fit); predict(fit, newdata = nd, type = "lp"); predict(fit, newdata = nd, type = "risk")
    using DataFrames, RDatasets
    using SurvivalModels: predict

    colon = dataset("survival", "colon")
    colon.Time = Float64.(colon.Time)
    colon.Status = Bool.(colon.Status)
    model = fit(Cox, @formula(Surv(Time, Status) ~ Age + Rx), colon)

    R_coef = [-0.00205614120475, -0.0200487755444, -0.43928902633]
    @test model.β ≈ R_coef rtol = 1e-4

    idx = [1, 50, 200, 800]
    nd  = colon[idx, :]

    R_lp   = [-0.404839254692, -0.46035506722,  0.00360765356699, -0.468579632039]
    R_risk = [ 0.667084032869,  0.631059537167, 1.00361416898,     0.625890632233]

    @test predict(model, :lp)[idx]   ≈ R_lp   rtol = 1e-4
    @test predict(model, :lp, nd)    ≈ R_lp   rtol = 1e-4
    @test predict(model, :risk)[idx] ≈ R_risk rtol = 1e-4
    @test predict(model, :risk, nd)  ≈ R_risk rtol = 1e-4
end

@testitem "Verify harrells_c" begin
    using SurvivalModels: harrells_c

    # Perfect concordance: higher risk = earlier event
    T = [1.0, 2.0, 3.0, 4.0]
    Δ = [true, true, true, true]
    @test harrells_c(T, Δ, [4.0, 3.0, 2.0, 1.0]) == 1.0

    # Reversed: every comparable pair is discordant
    @test harrells_c(T, Δ, [1.0, 2.0, 3.0, 4.0]) == 0.0

    # All-tied risks: every pair is tied → C = 0.5 by the tied-half-credit convention
    @test harrells_c(T, Δ, [1.0, 1.0, 1.0, 1.0]) == 0.5

    # Censored subjects don't form permissible pairs with later observations:
    # T = [1, 2, 3], Δ = [false, true, true]. Only pair (subject-2, subject-3) is permissible.
    T_c = [1.0, 2.0, 3.0]
    Δ_c = [false, true, true]
    @test harrells_c(T_c, Δ_c, [0.0, 2.0, 1.0]) == 1.0       # subject 2 has higher risk → concordant
    @test harrells_c(T_c, Δ_c, [0.0, 1.0, 2.0]) == 0.0       # subject 2 has lower risk → discordant
    @test harrells_c(T_c, Δ_c, [0.0, 1.0, 1.0]) == 0.5       # tied risks on the single permissible pair
    @test harrells_c(T_c, Δ_c, [9.0, 2.0, 1.0]) == 1.0       # subject 1's risk irrelevant — no permissible pair involves it

    # Cross-check against R's `survival::concordance` on the ovarian fixture
    using RDatasets
    ovarian = dataset("survival", "ovarian")
    model = fit(Cox, @formula(Surv(FUTime, FUStat) ~ Age + ECOG_PS), ovarian)
    @test harrells_c(model) ≈ 0.7844 rtol = 1e-3
end


@testitem "Verify brier_score" begin
    using DataFrames
    using SurvivalModels: brier_score, integrated_brier_score, predict

    # No-censoring degenerate case: all subjects had events. Ĝ(t) ≡ 1, so the IPCW
    # formula reduces to the standard squared error against the at-risk indicator.
    T = [1.0, 2.0, 3.0, 4.0]
    Δ = [true, true, true, true]
    Ŝ = [0.9, 0.6, 0.3, 0.1]
    # At t* = 2.5: subjects 1, 2 had events ≤ t* (Y=0), subjects 3, 4 still at risk (Y=1)
    expected = ((0.0 - 0.9)^2 + (0.0 - 0.6)^2 + (1.0 - 0.3)^2 + (1.0 - 0.1)^2) / 4
    @test brier_score(T, Δ, Ŝ, 2.5) ≈ expected rtol = 1e-10

    # BS(t* = 0): every subject is still at risk (T > 0), Ŝ ≈ 1, so weights ≈ (1-1)² = 0
    # Use a Cox fit so the predicted Ŝ at t=0 is exactly 1.
    using RDatasets
    ovarian = dataset("survival", "ovarian")
    model = fit(Cox, @formula(Surv(FUTime, FUStat) ~ Age + ECOG_PS), ovarian)
    @test brier_score(model, 0.0) ≈ 0.0 atol = 1e-12

    # Newdata path on the training df matches the no-newdata path
    @test brier_score(model, 500.0)               ≈ brier_score(model, ovarian, 500.0)     rtol = 1e-10
    @test brier_score(model, [200.0, 500.0])      ≈ brier_score(model, ovarian, [200.0, 500.0]) rtol = 1e-10

    # Vector-ts API returns the same values as a loop of scalar-t calls
    ts = [100.0, 500.0, 800.0]
    @test brier_score(model, ts) ≈ [brier_score(model, t) for t in ts] rtol = 1e-10

    # Brier in [0, 1] for typical horizons on a well-behaved fixture
    bs_grid = brier_score(model, [100.0, 500.0, 1000.0])
    @test all(0 .≤ bs_grid .≤ 1)

    # Integrated Brier: trapezoid convergence — refining the grid should not move IBS much
    ibs_100 = integrated_brier_score(model; t_max = 1200.0, n_grid = 100)
    ibs_200 = integrated_brier_score(model; t_max = 1200.0, n_grid = 200)
    @test ibs_100 ≈ ibs_200 rtol = 1e-2
    @test 0 ≤ ibs_100 ≤ 1

    # IBS on training df via newdata path matches no-newdata
    @test integrated_brier_score(model; t_max = 1200.0) ≈
          integrated_brier_score(model, ovarian; t_max = 1200.0) rtol = 1e-10
end


@testitem "brier_score errors without stored formula" begin
    using DataFrames
    using SurvivalModels: brier_score, CoxDefault, Cox

    time   = [1.0, 3.0, 5.0, 6.0, 2.0, 7.0, 9.0, 11.0]
    status = [true, false, true, true, true, false, true, true]
    sex    = Symbol.([1, 1, 1, 1, 0, 0, 0, 0])
    age    = [57, 52, 48, 42, 39, 31, 26, 22]
    df     = DataFrame(time = time, status = status, sex = sex, age = age)

    X = hcat(age, [s == Symbol(1) ? 1.0 : 0.0 for s in sex])
    worker = CoxDefault(time, Bool.(status), X)
    bare = Cox(worker, [:age, :sex], [:continuous, :categorical])

    @test_throws ErrorException brier_score(bare, df, 5.0)
end


@testitem "brier_score for GeneralHazardModel" begin
    using DataFrames, Distributions, Random
    using SurvivalModels: brier_score, integrated_brier_score, predict_survival,
                          ProportionalHazard, AcceleratedFaillureTime,
                          AcceleratedHazard, GeneralHazard
    using StableRNGs

    rng = StableRNG(2025)
    n   = 30
    df  = DataFrame(
        time   = rand(rng, Exponential(2.0), n),
        status = rand(rng, Bool, n),
        x1     = randn(rng, n),
        x2     = randn(rng, n),
    )

    # All four method types route through the same overloads. PH stands in for the
    # group; the GH path is exercised separately below for the two-formula fit.
    single_formula_types = (
        ProportionalHazard{Weibull},
        AcceleratedFaillureTime{Weibull},
        AcceleratedHazard{Weibull},
    )
    for T in single_formula_types
        m = fit(T, @formula(Surv(time, status) ~ x1 + x2), df)

        # Self-consistency: training-data convenience equals the low-level form.
        @test brier_score(m, 1.0) ≈ brier_score(m.T, m.Δ, predict_survival(m, 1.0), 1.0)

        # Vector-ts matches a loop of scalar-t calls.
        ts = [0.5, 1.0, 2.0]
        @test brier_score(m, ts) ≈ [brier_score(m, t) for t in ts] rtol = 1e-12

        # Newdata-on-training equals no-newdata.
        @test brier_score(m, 1.0)              ≈ brier_score(m, df, 1.0)              rtol = 1e-10
        @test brier_score(m, [0.5, 1.0, 2.0])  ≈ brier_score(m, df, [0.5, 1.0, 2.0])  rtol = 1e-10

        # Brier ∈ [0, 1] on a typical horizon.
        @test all(0 .≤ brier_score(m, [0.5, 1.0, 2.0]) .≤ 1)

        # Integrated Brier convergence and newdata-equals-training.
        ibs_100 = integrated_brier_score(m; t_max = 4.0, n_grid = 100)
        ibs_200 = integrated_brier_score(m; t_max = 4.0, n_grid = 200)
        @test ibs_100 ≈ ibs_200 rtol = 1e-2
        @test ibs_100 ≈ integrated_brier_score(m, df; t_max = 4.0) rtol = 1e-10
    end

    # Two-formula GH fit
    m_gh = fit(GeneralHazard{Weibull},
               @formula(Surv(time, status) ~ x1 + x2),
               @formula(Surv(time, status) ~ x1),
               df)
    @test brier_score(m_gh, 1.0)             ≈ brier_score(m_gh, df, 1.0)             rtol = 1e-10
    @test brier_score(m_gh, [0.5, 1.0, 2.0]) ≈ brier_score(m_gh, df, [0.5, 1.0, 2.0]) rtol = 1e-10
end


@testitem "GHM brier_score errors without stored formula" begin
    using DataFrames, Distributions, Random
    using SurvivalModels: brier_score, ProportionalHazard
    using StableRNGs

    rng = StableRNG(99)
    n   = 20
    T   = rand(rng, Exponential(2.0), n)
    Δ   = rand(rng, Bool, n)
    X1  = randn(rng, n, 2)
    X2  = randn(rng, n, 2)
    df  = DataFrame(time = T, status = Δ, x1 = X1[:, 1], x2 = X1[:, 2])

    # Direct positional constructor — no formulas captured.
    bare = ProportionalHazard(T, Δ, Weibull(1.0, 2.0), X1, X2, zeros(2), zeros(2))

    @test_throws ErrorException brier_score(bare, df, 1.0)
    @test_throws ErrorException brier_score(bare, df, [0.5, 1.0])
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


@testitem "GeneralHazardModel predict shapes and consistency" begin
    using Distributions, Random
    using SurvivalModels: predict, ProportionalHazard, AcceleratedFaillureTime,
                          AcceleratedHazard, GeneralHazard, predict_expected, predict_survival
    using StableRNGs

    rng = StableRNG(456)
    n   = 25
    T   = rand(rng, Exponential(2.0), n)
    Δ   = rand(rng, Bool, n)
    X1  = randn(rng, n, 2)
    X2  = randn(rng, n, 2)

    constructors = (ProportionalHazard, AcceleratedFaillureTime, AcceleratedHazard, GeneralHazard)
    for ctor in constructors
        m = ctor(T, Δ, Weibull(1.0, 2.0), X1, X2)

        # Shapes
        @test size(predict(m, :survival))                       == (n,)
        @test size(predict(m, :expected))                       == (n,)
        @test size(predict(m, :survival, 1.0))                  == (n,)
        @test size(predict(m, :expected, 1.0))                  == (n,)
        @test size(predict(m, :survival, [0.5, 1.0, 2.0]))      == (n, 3)
        @test size(predict(m, :expected, [0.5, 1.0, 2.0]))      == (n, 3)

        # Self-consistency: S = exp(-H) at both scalar and vector t
        @test predict(m, :survival, 1.0)            ≈ exp.(-predict(m, :expected, 1.0))            rtol = 1e-12
        @test predict(m, :survival, [0.5, 1.0, 2.0]) ≈ exp.(-predict(m, :expected, [0.5, 1.0, 2.0])) rtol = 1e-12

        # No-arg form evaluates at each subject's own observed Tᵢ
        no_arg_S = predict(m, :survival)
        for i in 1:n
            @test no_arg_S[i] ≈ predict(m, :survival, T[i])[i] rtol = 1e-12
        end

        # S(t) ∈ [0, 1] and non-increasing in t
        S_grid = predict(m, :survival, [0.1, 0.5, 1.0, 2.0, 5.0, 20.0])
        @test all(0 .≤ S_grid .≤ 1 .+ 1e-12)
        @test all(diff(S_grid, dims = 2) .≤ 1e-12)

        # S(0) = 1 exactly (for distributions with support starting at 0+)
        @test predict(m, :survival, 0.0) ≈ ones(n) atol = 1e-12

        # Misuse: predict on an unsupported type
        @test_throws ErrorException predict(m, :lp, 1.0)
        @test_throws ErrorException predict(m, :risk)
    end
end


@testitem "ProportionalHazard with β = 0 reduces to baseline survival" begin
    using Distributions, Random
    using SurvivalModels: predict, ProportionalHazard, PHMethod, GeneralHazardModel
    using StableRNGs

    rng = StableRNG(789)
    n   = 8
    T   = rand(rng, Exponential(2.0), n)
    Δ   = rand(rng, Bool, n)
    X1  = randn(rng, n, 2)
    X2  = randn(rng, n, 2)
    baseline = Weibull(1.0, 2.0)

    # Direct construction with α = β = 0 (no covariate effect).
    m = ProportionalHazard(T, Δ, baseline, X1, X2, zeros(2), zeros(2))

    # S(t | x) should reduce to ccdf(baseline, t) regardless of x.
    for t in (0.5, 1.0, 2.5, 5.0)
        expected = ccdf(baseline, t)
        @test all(predict(m, :survival, t) .≈ expected) skip = false
    end

    # And the matrix form should be constant across subjects at each t.
    S_mat = predict(m, :survival, [0.5, 1.0, 2.5, 5.0])
    for j in 1:size(S_mat, 2)
        @test all(S_mat[:, j] .≈ S_mat[1, j])
    end
end


@testitem "GeneralHazardModel predict on newdata" begin
    using DataFrames, Distributions, Random
    using SurvivalModels: predict, ProportionalHazard, AcceleratedFaillureTime,
                          AcceleratedHazard, GeneralHazard, GeneralHazardModel, PHMethod
    using StableRNGs

    rng = StableRNG(2024)
    n   = 30
    df  = DataFrame(
        time   = rand(rng, Exponential(2.0), n),
        status = rand(rng, Bool, n),
        x1     = randn(rng, n),
        x2     = randn(rng, n),
    )

    # Single-formula fits exercise PH/AFT/AH (all four are routed through the same
    # internal path with both formulas set to the same applied formula).
    single_formula_types = (
        ProportionalHazard{Weibull},
        AcceleratedFaillureTime{Weibull},
        AcceleratedHazard{Weibull},
    )
    for T in single_formula_types
        m = fit(T, @formula(Surv(time, status) ~ x1 + x2), df)

        # Self-consistency: newdata-on-training matches the no-newdata path.
        @test predict(m, :survival, df, 1.0)             ≈ predict(m, :survival, 1.0)             rtol = 1e-10
        @test predict(m, :expected, df, 1.0)             ≈ predict(m, :expected, 1.0)             rtol = 1e-10
        @test predict(m, :survival, df, [0.5, 1.0, 2.0]) ≈ predict(m, :survival, [0.5, 1.0, 2.0]) rtol = 1e-10

        # Held-out slice shapes.
        held = df[1:5, :]
        @test size(predict(m, :survival, held, 1.0))             == (5,)
        @test size(predict(m, :expected, held, 1.0))             == (5,)
        @test size(predict(m, :survival, held, [0.5, 1.0, 2.0])) == (5, 3)
        @test size(predict(m, :expected, held, [0.5, 1.0, 2.0])) == (5, 3)

        # `S = exp(-H)` consistency on newdata.
        @test predict(m, :survival, held, 1.0) ≈ exp.(-predict(m, :expected, held, 1.0)) rtol = 1e-12

        # `S(t)` ∈ [0,1] and non-increasing in t on newdata.
        S = predict(m, :survival, held, [0.5, 1.0, 2.0, 5.0])
        @test all(0 .≤ S .≤ 1 .+ 1e-12)
        @test all(diff(S, dims = 2) .≤ 1e-12)

        # Newdata predict requires an explicit time argument.
        @test_throws ErrorException predict(m, :survival, held)
        @test_throws ErrorException predict(m, :expected, held)

        # Unsupported types error.
        @test_throws ErrorException predict(m, :lp, held, 1.0)
    end

    # Two-formula GH fit captures both formulas distinctly.
    m_gh = fit(GeneralHazard{Weibull},
               @formula(Surv(time, status) ~ x1 + x2),
               @formula(Surv(time, status) ~ x1),
               df)
    @test predict(m_gh, :survival, df, 1.0) ≈ predict(m_gh, :survival, 1.0) rtol = 1e-10
    @test size(predict(m_gh, :expected, df[1:5, :], [1.0, 2.0])) == (5, 2)

    # Direct construction without formulas should error on newdata predict.
    n2  = 20
    T2  = rand(rng, Exponential(2.0), n2)
    Δ2  = rand(rng, Bool, n2)
    X12 = randn(rng, n2, 2)
    X22 = randn(rng, n2, 2)
    bare = ProportionalHazard(T2, Δ2, Weibull(1.0, 2.0), X12, X22, zeros(2), zeros(2))
    @test_throws ErrorException predict(bare, :survival, df, 1.0)
end


@testitem "GHM Weibull PH/AFT matches R flexsurv on ovarian" begin
    # Cross-check `ProportionalHazard{Weibull}` and `AcceleratedFaillureTime{Weibull}`
    # against R's `flexsurv::flexsurvreg(..., dist = "weibullPH" | "weibull")` on the
    # ovarian fixture.
    #
    # Reference values produced with flexsurv 2.3.2 and `options(digits = 14)`:
    #
    #   m_ph  <- flexsurvreg(Surv(futime, fustat) ~ age + ecog.ps,
    #                        data = ovarian, dist = "weibullPH")
    #   m_aft <- flexsurvreg(Surv(futime, fustat) ~ age + ecog.ps,
    #                        data = ovarian, dist = "weibull")
    #   newd  <- data.frame(age = c(60, 50), ecog.ps = c(1, 2))
    #   summary(m_ph,  newdata = newd, t = c(100, 300, 600), type = "survival", tidy = TRUE)
    #   summary(m_aft, newdata = newd, t = c(100, 300, 600), type = "survival", tidy = TRUE)
    #
    # FUTime is scaled by 1/1000 so that LBFGS converges from its zeros-init (which
    # places the baseline at `Weibull(1, 1)`; on the unscaled t ~ 60..1227 it sits in
    # a flat region of the loglik and fails to make progress). β is scale-invariant.

    using DataFrames, Distributions, RDatasets
    using SurvivalModels: ProportionalHazard, AcceleratedFaillureTime, predict_survival

    ovarian = dataset("survival", "ovarian")
    ovarian.FUTime = Float64.(ovarian.FUTime) ./ 1000
    ovarian.FUStat = Bool.(ovarian.FUStat)

    m_ph  = fit(ProportionalHazard{Weibull},
                @formula(Surv(FUTime, FUStat) ~ Age + ECOG_PS), ovarian)
    m_aft = fit(AcceleratedFaillureTime{Weibull},
                @formula(Surv(FUTime, FUStat) ~ Age + ECOG_PS), ovarian)

    # PH coefficients are log hazard ratios — identical convention in both packages.
    @test m_ph.β[1] ≈  0.16083342676210 rtol = 1e-4
    @test m_ph.β[2] ≈ -0.16538646310541 rtol = 1e-4

    # AFT coefficients: SurvivalModels.jl uses `c1 = exp(x'β)` (positive β → time
    # accelerates → shorter survival); flexsurv's AFT uses `log T = x'β + σW`
    # (positive β → longer survival). Hence Julia's β = -flexsurv's β_aft.
    @test m_aft.β[1] ≈ -(-0.097026692796186) rtol = 1e-4
    @test m_aft.β[2] ≈ -( 0.099773423332264) rtol = 1e-4

    # Predicted survival is parameterization-invariant — pin flexsurv's values.
    newd = DataFrame(Age = [60.0, 50.0], ECOG_PS = [1.0, 2.0])
    ts   = [100.0, 300.0, 600.0] ./ 1000   # match scaled fit
    S_ph_ref = [0.96180227004260 0.78613097470795 0.46804856049608
                0.99341257476790 0.95998720313543 0.87912015135050]

    @test predict_survival(m_ph,  newd, ts) ≈ S_ph_ref rtol = 1e-5
    @test predict_survival(m_aft, newd, ts) ≈ S_ph_ref rtol = 1e-5

    # Sanity: same data, same baseline family, so PH and AFT survival predictions
    # coincide regardless of which parameterization optimization found.
    @test predict_survival(m_ph, newd, ts) ≈ predict_survival(m_aft, newd, ts) rtol = 1e-6
end
