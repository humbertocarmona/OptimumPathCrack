#=
Analysis of the parameter study: which parameters control robustness?

Reads `summary.csv` written by `parameter_study.jl` and, for each response
(R_mean = mean removals per pair to disconnection, and the cost-capped versions),
reports:
  - noise: share of the variance that comes from network realization (replicates of
    the same setting), which caps how much any model can explain;
  - standardized regression coefficients (OLS on z-scored parameters): linear effect
    of one standard deviation of each parameter, in units of the response's standard
    deviation, with standard errors;
  - η² (correlation ratio over 10 quantile bins): share of the response variance
    explained by each parameter alone, including non-linear effects (a crude
    first-order sensitivity index);
  - Spearman rank correlation: monotone effect;
  - the same regression on the *measured* network structure (`net_*` columns) instead
    of the generator settings.

Usage (from the package root):
	julia --project=scripts scripts/analyze_study.jl [study_dir]
Writes `analysis_<response>.csv` files into the study directory.
=#

using CSV, DataFrames, LinearAlgebra, Statistics, Printf

const DIR = isempty(ARGS) ? "results/parameter_study" : ARGS[1]

const DESIGN_PARAMS = [:oneway_fraction, :alternation, :mean_degree, :arterial_fraction,
	:arterial_speedup, :road_cv, :segment_cv]
const MEASURED = [:net_mean_degree, :net_oneway_fraction, :net_dead_end_fraction,
	:net_scc_fraction, :net_weight_cv, :alternation, :arterial_fraction, :arterial_speedup]

zscore(x) = (x .- mean(x)) ./ std(x)
ranks(x) = invperm(sortperm(x))
spearman(x, y) = cor(ranks(x), ranks(y))

"""Correlation ratio: variance of the bin means of y over quantile bins of x / var(y)."""
function eta2(x, y; nbins = 10)
	n = length(x)
	nbins = min(nbins, n ÷ 3)
	nbins < 2 && return NaN
	bin = ceil.(Int, ranks(x) ./ n .* nbins)
	ȳ = mean(y)
	between = sum(sum(bin .== b) * (mean(y[bin .== b]) - ȳ)^2 for b in unique(bin))
	return between / sum(abs2, y .- ȳ)
end

"""OLS of z-scored y on z-scored X: coefficients, standard errors, R²."""
function standardized_ols(df, xs, y)
	X = hcat(ones(nrow(df)), (zscore(df[!, x]) for x in xs)...)
	Y = zscore(df[!, y])
	if rank(X) < size(X, 2)    # fewer distinct settings than coefficients
		n = length(xs)
		return fill(NaN, n), fill(NaN, n), NaN
	end
	β = X \ Y
	resid = Y - X * β
	dof = nrow(df) - size(X, 2)
	σ2 = dof > 0 ? sum(abs2, resid) / dof : NaN
	se = sqrt.(max.(diag(σ2 * inv(X' * X)), 0))
	R2 = 1 - sum(abs2, resid) / sum(abs2, Y .- mean(Y))
	return β[2:end], se[2:end], R2
end

"""Share of variance due to replicates of the same setting (realization noise)."""
function noise_share(df, y)
	within = combine(groupby(df, :setting), y => (v -> length(v) > 1 ? var(v) : NaN) => :v)
	w = filter(!isnan, within.v)
	return isempty(w) ? NaN : mean(w) / var(df[!, y])
end

function report(df, y)
	println("\n", "="^78, "\nResponse: $y   (mean $(@sprintf("%.2f", mean(df[!, y]))), ",
		"sd $(@sprintf("%.2f", std(df[!, y]))), $(nrow(df)) runs)\n", "="^78)
	ns = noise_share(df, y)
	@printf("realization noise: %.1f%% of the variance (best possible R² ≈ %.2f)\n",
		100ns, 1 - ns)

	β, se, R2 = standardized_ols(df, DESIGN_PARAMS, y)
	table = DataFrame(parameter = DESIGN_PARAMS, beta = β, se = se,
		eta2 = [eta2(df[!, x], df[!, y]) for x in DESIGN_PARAMS],
		spearman = [spearman(df[!, x], df[!, y]) for x in DESIGN_PARAMS])
	sort!(table, :eta2; rev = true)
	isnan(R2) && println("\n(linear model not identifiable: fewer distinct settings than coefficients)")
	@printf("\nGenerator settings (linear model R² = %.2f), ranked by η²:\n", R2)
	@printf("  %-20s %8s %7s %7s %9s\n", "parameter", "beta", "±se", "η²", "spearman")
	for r in eachrow(table)
		@printf("  %-20s %8.3f %7.3f %7.3f %9.3f\n", r.parameter, r.beta, r.se, r.eta2, r.spearman)
	end
	println("  (eta² sums above 1 or far above R² signal interactions / non-linear effects)")

	β2, se2, R22 = standardized_ols(df, MEASURED, y)
	@printf("\nMeasured structure (linear model R² = %.2f):\n", R22)
	for (x, b, s) in sort(collect(zip(MEASURED, β2, se2)); by = t -> -abs(t[2]))
		@printf("  %-22s %8.3f ± %.3f\n", x, b, s)
	end

	CSV.write(joinpath(DIR, "analysis_$(y).csv"), table)
	return table
end

function main()
	df = CSV.read(joinpath(DIR, "summary.csv"), DataFrame)
	println("Study in $DIR: $(nrow(df)) runs, $(length(unique(df.setting))) settings")
	if nrow(df) < 3 * length(DESIGN_PARAMS)
		@warn "only $(nrow(df)) runs for $(length(DESIGN_PARAMS)) parameters: estimates are very rough"
	end
	responses = filter(n -> startswith(n, "R_mean"), names(df))
	for y in responses
		report(df, Symbol(y))
	end
end

main()
