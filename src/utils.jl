
# Rounded formatting for coefficient/parameter values in `show` methods.
_coef_fmt(x) = string(round(x; sigdigits = 5))

# Confidence level as a compact percentage string, e.g. 0.95 -> "95", 0.9 -> "90".
_ci_levstr(level) = (v = 100 * level; isinteger(v) ? string(Int(v)) : string(v))

# Wald coefficient table shared by the hazard-based regression models: estimate,
# standard error, `z`, p-value, `exp(coef)`, and the confidence interval for
# `exp(coef)` at `level`. `exp(coef)` is a hazard ratio for hazard-level effects
# and a time/acceleration ratio for time-scale effects; the row names carry that
# distinction.
function _hr_coeftable(b, se, rownms; level::Real = 0.95)
    z = b ./ se
    p = 2 .* ccdf.(Normal(), abs.(z))
    zc = quantile(Normal(), (1 + level) / 2)
    ls = _ci_levstr(level)
    return CoefTable(
        [b, se, z, p, exp.(b), exp.(b .- zc .* se), exp.(b .+ zc .* se)],
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "exp(coef)", "Lower $ls%", "Upper $ls%"],
        rownms, 4, 3,
    )
end

"""
    harrells_c(times, statuses, risk_scores)
    harrels_c(C::Cox)

Compute Harrell's concordance index (C-index) for survival models.

# Arguments
- `times`: Vector of observed times (Float64).
- `statuses`: Vector of event indicators (Bool; true if event, false if censored).
- `risk_scores`: Vector of predicted risk scores (higher means higher risk).

# Returns
- The C-index, a value between 0 and 1 indicating the proportion of all usable patient pairs in which predictions and outcomes are concordant.

# Details
The C-index measures the discriminative ability of a survival model: it is the probability that, for a randomly chosen pair of comparable subjects, the subject with the higher predicted risk actually experiences the event before the other. Tied risk scores count as half-concordant.

You can also call `harrels_c(C::Cox)` to compute the C-index for a fitted Cox model.

"""
function harrells_c(times, statuses, risk_scores)
	permissible_pairs = 0
	concordant_pairs = 0
	tied_risk_pairs = 0

	n = length(times)
	for i in 1:n
		for j in (i+1):n
			# Determine which patient has the shorter follow-up time
			if times[i] < times[j]
				p1_idx, p2_idx = i, j
			elseif times[j] < times[i]
				p1_idx, p2_idx = j, i
			else # If times are equal, order by status (event first)
				if statuses[i] && !statuses[j]
					p1_idx, p2_idx = i, j
				elseif !statuses[i] && statuses[j]
					p1_idx, p2_idx = j, i
				else # both have same time and status
					continue 
				end
			end

			# A pair is only "permissible" for comparison if the patient with the
			# shorter follow-up time actually had an event. If they were censored,
			# we don't know when their event would have happened, so we can't compare.
			if statuses[p1_idx]
				permissible_pairs += 1

				# Check for concordance
				if risk_scores[p1_idx] > risk_scores[p2_idx]
					concordant_pairs += 1
				elseif risk_scores[p1_idx] == risk_scores[p2_idx]
					tied_risk_pairs += 1
				end
				# If risk_scores[p1_idx] < risk_scores[p2_idx], it's discordant.
			end
		end
	end

	if permissible_pairs == 0
		return 0.0 # Or NaN, depending on desired behavior for no permissible pairs
	end

	return (concordant_pairs + 0.5 * tied_risk_pairs) / permissible_pairs
end