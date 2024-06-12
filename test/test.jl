# Required packages
using HazReg
using Distributions
using Random
using DataFrames
using LogExpFunctions
using Optim


# Sample size
n = 10000

# Simulated design matrices
Random.seed!(123)
dist = Normal()
des = hcat(rand(dist, n), rand(dist, n))
des_t = rand(dist, n)


#----------------------------
# PGW-GH simulation
#----------------------------

# True parameters
theta0 = [0.1,2.0,5.0]
alpha0 = 0.5
beta0 = [-0.5,0.75]

# censoring
cens = 10

# Data simulation
simdat = simGH(seed = 1234, n = n, des = des, des_t = des_t,
      theta = theta0, alpha = alpha0, beta = beta0, 
      hstr = "GH", dist = "PGW")

# status variable
status = collect(Bool,(simdat .< cens))

# Inducing censoring
simdat = min.(simdat, cens)


# Model fit

OPTCox = CoxMPLE(init = fill(0.0, size(des)[2]), times = simdat,
            status = status,  
            des = des,  
            method = "NM", maxit = 1000)
# 95% Confidence intervals under the reparameterisation
CI = ConfInt(FUN = OPTCox[2], MLE = OPTCox[1].minimizer, level = 0.95)

CI = DataFrame(CI, :auto)
 
rename!( CI, ["Lower", "Upper"] )

println(CI)