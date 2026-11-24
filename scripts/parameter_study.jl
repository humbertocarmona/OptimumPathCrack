#=
Parameter study: which city-lattice parameters control robustness to optimal-path
cracking?

For each parameter setting (a Latin hypercube over the ranges in PARAMS) and each of
N_SEEDS network realizations, the script
  1. builds a city lattice (`buildCityLattice`),
  2. cracks N_OD_PAIRS origin-destination pairs at each distance in DISTANCES, until
     the pair is disconnected (`crackStatistics`, multi-threaded),
  3. computes the robustness R = ∫ ⟨n_removed⟩ dL, both to disconnection and with a cost
     cap (removals before the shortest path costs more than `ratio` × the first one),
  4. appends one row to `summary.csv` with the parameters, the measured network
     structure (`networkSummary`) and the responses.

Cost caps need no extra runs: path costs never decrease as edges are removed, and the
random tie-breaking draws are the same up to the stop, so a capped run removes exactly
the edges of the uncapped run whose `path_cost ≤ ratio × first_cost`.

The study can be stopped and restarted: runs already in `summary.csv` are skipped.

Usage (from the package root):
	julia -t auto --project=scripts scripts/parameter_study.jl [--quick] [output_dir]

`--quick` runs a small version (few settings, small lattice) to check the pipeline.
Then analyse with `scripts/analyze_study.jl`.
=#

using OptimumPathCrack
using CSV, DataFrames, Random, TOML

# ------------------------------------------------------------------------- settings
const QUICK = "--quick" in ARGS
const POSITIONAL = filter(a -> !startswith(a, "--"), ARGS)
const OUTDIR = isempty(POSITIONAL) ?
			   (QUICK ? "results/parameter_study_quick" : "results/parameter_study") : POSITIONAL[1]

# Parameters varied, with their ranges (Latin hypercube). Uniform within the range, or
# uniform in log for ranges marked :log.
const PARAMS = [
	:oneway_fraction => (0.0, 1.0),    # share of local roads that are one-way
	:alternation => (0.0, 1.0),        # P(next one-way road runs opposite); 0.5 = random
	:mean_degree => (2.8, 4.0),        # street neighbours per intersection (Boston: 2.89)
	:arterial_fraction => (0.0, 0.3),  # share of roads that are arterials
	:arterial_speedup => (1.0, 3.0),   # arterial block time = local / speedup
	:road_cv => (0.0, 0.8),            # weight variation between roads
	:beta => (0.5, 20.0, :log),        # disorder of local blocks (Boston speeds: ≈ 1.7)
	:hurst => (-1.0, 0.8),             # spatial correlation of disorder (-1: none)
]

# Fixed lattice settings
const FIXED = (arterials_twoway = true, arterial_segment_cv = 0.02, pbc = true,
	block = 100.0, weight_mean = 1.0)

const N_SETTINGS = QUICK ? 6 : 200            # parameter settings
const N_SEEDS = QUICK ? 2 : 3                 # network realizations per setting
const NX = QUICK ? 64 : 256                   # lattice side (NX * block meters)
const DISTANCES = QUICK ? [500.0, 1000.0, 2000.0] : [500.0, 1000.0, 2000.0, 4000.0, 8000.0]
const N_OD_PAIRS = QUICK ? 20 : 50            # OD pairs per distance and network
# Cracking stops at disconnection or when the path costs more than MAX_COST_RATIO × the
# first path. Inf = always to disconnection (slow for strong disorder: hundreds of
# removals per pair at beta ≳ 5). R_mean is the mean removals with this stopping rule.
const MAX_COST_RATIO = Inf
const COST_RATIOS = filter(<(MAX_COST_RATIO), [3.0, 2.0, 1.5, 1.2])  # caps evaluated after the run
const SAVE_EDGES = false                      # keep per-edge tables (large at this size)
const DESIGN_SEED = 2026

@assert maximum(DISTANCES) < NX * FIXED.block / 2 "distances must stay below half the torus size"

# --------------------------------------------------------------------- design
"""Latin hypercube: n points, each parameter stratified into n equal bins."""
function latin_hypercube(n, params; rng)
	design = DataFrame(setting = 1:n)
	for (name, range) in params
		lo, hi = range[1], range[2]
		u = (randperm(rng, n) .- rand(rng, n)) ./ n
		design[!, name] = length(range) == 3 && range[3] == :log ?
						  exp.(log(lo) .+ (log(hi) - log(lo)) .* u) : lo .+ (hi - lo) .* u
	end
	return design
end

"""Number of removals before the path cost exceeds `ratio` × the first cost, per pair."""
function capped_removals(edges_df, od_df, ratio)
	first_cost = Dict((r.ℓ, r.od_id) => r.first_cost for r in eachrow(od_df))
	counts = Dict{Tuple{Float64, Int}, Int}()
	for r in eachrow(edges_df)
		r.path_cost <= ratio * first_cost[(r.ℓ, r.od_id)] || continue
		key = (r.ℓ, r.od_id)
		counts[key] = get(counts, key, 0) + 1
	end
	return [get(counts, (r.ℓ, r.od_id), 0) for r in eachrow(od_df)]
end

ratio_tag(ratio) = replace(string(ratio), "." => "p")

function run_one(setting::DataFrameRow, rep::Int)
	run_id = "s$(lpad(setting.setting, 4, '0'))_r$rep"
	seed = 1000 * setting.setting + rep
	kwargs = (; (name => setting[name] for (name, _) in PARAMS)..., FIXED...)

	g, coords, dm, wm, _ = buildCityLattice(NX, NX; kwargs..., seed = seed)
	net = networkSummary(g, wm)
	edges_df, od_df = crackStatistics(g, coords, dm, wm, DISTANCES;
		n_od_pairs = N_OD_PAIRS, seed = seed, maxCostRatio = MAX_COST_RATIO)

	# columns in a fixed order, so rows can be appended to the CSV
	row = Pair{Symbol, Any}[:run_id => run_id, :setting => setting.setting, :rep => rep,
		:seed => seed, :n_pairs => nrow(od_df)]
	append!(row, [name => setting[name] for (name, _) in PARAMS])
	append!(row, [Symbol("net_", k) => v for (k, v) in pairs(net)])

	span = maximum(DISTANCES) - minimum(DISTANCES)
	R, per = integratedRemovals(od_df)
	push!(row, :R_mean => R / span)                 # average removals per pair
	for ℓ in DISTANCES
		k = findfirst(==(ℓ), per.ℓ)
		push!(row, Symbol("removed_L", Int(ℓ)) => k === nothing ? NaN : per.mean[k])
	end
	for ratio in COST_RATIOS
		col = Symbol("n_removed_cap", ratio_tag(ratio))
		od_df[!, col] = capped_removals(edges_df, od_df, ratio)
		Rc, _ = integratedRemovals(od_df; column = col)
		push!(row, Symbol("R_mean_cap", ratio_tag(ratio)) => Rc / span)
	end
	# mean weight of removed edges relative to the network mean
	push!(row, :removed_weight_rel => isempty(edges_df) ? NaN :
									   sum(edges_df.weight) / nrow(edges_df) / net.weight_mean)

	if SAVE_EDGES
		writeCrackStatistics(joinpath(OUTDIR, "runs", run_id), edges_df, od_df;
			extra = Dict(string(k) => v for (k, v) in pairs(kwargs)))
	end
	return row
end

# --------------------------------------------------------------------------- main
function main()
	mkpath(OUTDIR)
	design = latin_hypercube(N_SETTINGS, PARAMS; rng = Xoshiro(DESIGN_SEED))
	CSV.write(joinpath(OUTDIR, "design.csv"), design)
	open(joinpath(OUTDIR, "study.toml"), "w") do io
		TOML.print(io, Dict("n_settings" => N_SETTINGS, "n_seeds" => N_SEEDS, "nx" => NX,
			"distances" => DISTANCES, "n_od_pairs" => N_OD_PAIRS,
			"cost_ratios" => COST_RATIOS, "max_cost_ratio" => MAX_COST_RATIO,
			"design_seed" => DESIGN_SEED,
			"ranges" => Dict(string(k) => string.(collect(v)) for (k, v) in PARAMS),
			"fixed" => Dict(string(k) => v for (k, v) in pairs(FIXED)),
			"threads" => Threads.nthreads()))
	end

	summary_file = joinpath(OUTDIR, "summary.csv")
	done = isfile(summary_file) ? Set(CSV.read(summary_file, DataFrame).run_id) : Set{String}()
	todo = [(s, r) for s in eachrow(design) for r in 1:N_SEEDS
			if "s$(lpad(s.setting, 4, '0'))_r$r" ∉ done]
	@info "parameter study: $(length(todo)) runs to do, $(length(done)) already done, " *
		  "$(Threads.nthreads()) threads, output in $OUTDIR"

	t0 = time()
	for (k, (setting, rep)) in enumerate(todo)
		row = run_one(setting, rep)
		df = DataFrame([name => [v] for (name, v) in row])
		CSV.write(summary_file, df; append = isfile(summary_file))
		elapsed = time() - t0
		eta = elapsed / k * (length(todo) - k)
		@info "run $(df.run_id[1]) ($k/$(length(todo))): R_mean = $(round(df.R_mean[1], digits = 2)), " *
			  "elapsed $(round(elapsed / 60, digits = 1)) min, ETA $(round(eta / 60, digits = 1)) min"
	end
	@info "done: $summary_file"
end

main()
