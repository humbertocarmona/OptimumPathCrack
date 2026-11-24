#=
Analysis of the square-network study (`square_study.jl`).

For each cell (β, H, p_oneway, L) it averages over realizations, then for each
(β, H, p_oneway) fits power laws in the system size L:
	⟨n_removed⟩ ∝ L^α        (removals to disconnect left from right)
	⟨path_length⟩ ∝ L^d_opt  (length of the first optimal path; 1 = straight,
							   ≈ 1.22 for the strong-disorder optimal path in 2D)
and the same α for each cost cap. Prints α and d_opt as β × H tables for each p_oneway
and writes `cells.csv` (per-cell means) and `exponents.csv`.

Realizations whose origin could not reach the destination (stop_reason == no_path, which
happens with one-way links) are counted and left out of the means.

Usage: julia --project=scripts scripts/analyze_square_study.jl [study_dir]
=#

using CSV, DataFrames, Statistics, Printf

const DIR = isempty(ARGS) ? "results/square_study" : ARGS[1]

sem(x) = length(x) > 1 ? std(x) / sqrt(length(x)) : NaN

"""Least-squares slope of log y on log x, with its standard error."""
function loglog_slope(x, y)
	ok = (x .> 0) .& (y .> 0)
	count(ok) < 2 && return NaN, NaN
	lx, ly = log.(x[ok]), log.(y[ok])
	b = cov(lx, ly) / var(lx)
	n = length(lx)
	if n <= 2
		return b, NaN
	end
	resid = ly .- (mean(ly) - b * mean(lx)) .- b .* lx
	return b, sqrt(sum(abs2, resid) / (n - 2) / sum(abs2, lx .- mean(lx)))
end

function print_table(ex, col, title)
	betas = sort(unique(ex.beta))
	hursts = sort(unique(ex.hurst))
	for p in sort(unique(ex.p_oneway))
		println("\n$title, p_oneway = $p   (rows: β, columns: H)")
		@printf("%8s", "β \\ H")
		for H in hursts
			@printf("%8.2f", H)
		end
		println()
		for β in betas
			@printf("%8.2f", β)
			for H in hursts
				r = filter(r -> r.beta == β && r.hurst == H && r.p_oneway == p, ex)
				@printf("%8.3f", nrow(r) == 1 ? r[1, col] : NaN)
			end
			println()
		end
	end
end

function main()
	df = CSV.read(joinpath(DIR, "summary.csv"), DataFrame)
	n_nopath = count(==("no_path"), string.(df.stop_reason))
	println("Square study in $DIR: $(nrow(df)) realizations; $n_nopath without an initial path ",
		"(excluded)")
	df = filter(r -> string(r.stop_reason) != "no_path", df)

	caps = filter(n -> startswith(n, "n_removed_cap"), names(df))
	keys = [:beta, :hurst, :p_oneway, :L]
	cells = combine(groupby(df, keys), nrow => :n,
		:n_removed => mean => :n_removed, :n_removed => sem => :n_removed_sem,
		:path_length => mean => :path_length,
		:removed_weight_rel => mean => :removed_weight_rel,
		(Symbol(c) => mean => Symbol(c) for c in caps)...)
	sort!(cells, keys)
	CSV.write(joinpath(DIR, "cells.csv"), cells)

	ex = combine(groupby(cells, [:beta, :hurst, :p_oneway])) do c
		α, αse = loglog_slope(c.L, c.n_removed)
		d, dse = loglog_slope(c.L, c.path_length)
		nt = (alpha = α, alpha_se = αse, d_opt = d, d_opt_se = dse, n_sizes = nrow(c))
		for cap in caps
			a, _ = loglog_slope(c.L, c[!, cap])
			nt = merge(nt, NamedTuple{(Symbol("alpha_", replace(cap, "n_removed_" => "")),)}((a,)))
		end
		nt
	end
	sort!(ex, [:p_oneway, :beta, :hurst])
	CSV.write(joinpath(DIR, "exponents.csv"), ex)

	print_table(ex, :alpha, "α: ⟨n_removed⟩ ∝ L^α")
	print_table(ex, :d_opt, "d_opt: ⟨first optimal path length⟩ ∝ L^d_opt")
	for cap in caps
		print_table(ex, Symbol("alpha_", replace(cap, "n_removed_" => "")),
			"α with $(replace(cap, "n_removed_" => "")) (removals before cost > ratio × first)")
	end
	println("\nPer-cell means in $(joinpath(DIR, "cells.csv")); exponents in ",
		joinpath(DIR, "exponents.csv"))
end

main()
