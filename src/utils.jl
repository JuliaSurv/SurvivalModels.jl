
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