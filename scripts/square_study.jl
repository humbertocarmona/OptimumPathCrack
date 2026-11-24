#=
Square-network study: how disorder strength β, spatial correlation H and the share of
one-way links p control optimal-path cracking across an L × L square lattice.

Each realization is an L × L `buildSquareNetwork` with origin / destination super-nodes
attached to the left and right columns. It is cracked from origin to destination
(left side to right side) until they disconnect. The script runs a full grid over
BETAS × HURSTS × P_ONEWAY × SIZES with N_REPS independent networks per cell, and
writes one row per realization to `summary.csv`:
  n_removed, stop_reason, first / last path cost, length of the first optimal path
  (lattice edges), removals under cost caps (from the path-cost sequence), and the mean
  weight of removed edges relative to the mean weight.

Realizations run in parallel (Threads.@threads). The study can be stopped and
restarted: cells already in `summary.csv` are skipped.

Usage (from the package root):
	julia -t auto --project=scripts scripts/square_study.jl [--quick] [output_dir]
Then: julia --project=scripts scripts/analyze_square_study.jl [output_dir]
=#

using OptimumPathCrack
import OptimumPathCrack: CrackNetwork, _crack
using CSV, DataFrames, Random, Statistics, TOML

const QUICK = "--quick" in ARGS
const POSITIONAL = filter(a -> !startswith(a, "--"), ARGS)
const OUTDIR = isempty(POSITIONAL) ?
			   (QUICK ? "results/square_study_quick" : "results/square_study") : POSITIONAL[1]

# ------------------------------------------------------------------------- settings
const BETAS = QUICK ? [1.0, 10.0] : [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
const HURSTS = QUICK ? [-1.0, 0.6] : [-1.0, -0.5, 0.0, 0.3, 0.6, 0.8]  # -1: uncorrelated
const P_ONEWAY = QUICK ? [0.0, 0.5] : [0.0, 0.25, 0.5]   # probability a link is one-way
const SIZES = QUICK ? [16, 32] : [16, 32, 64, 128]       # L (lattice is L × L)
const N_REPS = QUICK ? 8 : 50                            # networks per cell
const DIRECTION_RHO = 0.0                                # correlation between directions
const COST_RATIOS = [3.0, 2.0, 1.5, 1.2]                 # caps evaluated after the run

ratio_tag(r) = replace(string(r), "." => "p")
cell_id(β, H, p, L) = "b$(β)_h$(H)_p$(p)_L$(L)"

function crack_one(β, H, p, L, rep, seed)
	g, _, W, _ = buildSquareNetwork(L, L; p = p, dist = β, seed = seed, od = true,
		pbc = false, hurst = H == -1 ? nothing : H, direction_rho = DIRECTION_RHO)
	org, dst = L * L + 1, L * L + 2
	net = CrackNetwork(g, W)
	res = _crack(net, org, dst; rng = Random.Xoshiro(UInt64[seed % UInt64, 3]))
	costs = res.path_costs
	first_cost = isempty(costs) ? NaN : costs[1]
	# first optimal path: edges recorded on path 1, minus the two super-node links
	path_length = count(==(1), values(res.path_first)) - 2
	μ = mean(DisorderDist(β))
	row = Pair{Symbol, Any}[:beta => β, :hurst => H, :p_oneway => p, :L => L, :rep => rep,
		:seed => seed, :n_removed => length(res.removed), :stop_reason => res.stop_reason,
		:first_cost => first_cost, :last_cost => isempty(costs) ? NaN : costs[end],
		:path_length => isempty(costs) ? 0 : path_length]
	for r in COST_RATIOS
		push!(row, Symbol("n_removed_cap", ratio_tag(r)) =>
			count(c -> c <= r * first_cost, costs[1:min(end, length(res.removed))]))
	end
	push!(row, :removed_weight_rel => isempty(res.removed) ? NaN :
									  sum(net.weights[res.removed]) / length(res.removed) / μ)
	return row
end

function main()
	mkpath(OUTDIR)
	open(joinpath(OUTDIR, "study.toml"), "w") do io
		TOML.print(io, Dict("betas" => BETAS, "hursts" => HURSTS, "p_oneway" => P_ONEWAY,
			"sizes" => SIZES, "n_reps" => N_REPS, "direction_rho" => DIRECTION_RHO,
			"cost_ratios" => COST_RATIOS, "threads" => Threads.nthreads()))
	end
	summary_file = joinpath(OUTDIR, "summary.csv")
	done = Set{String}()
	if isfile(summary_file)
		s = CSV.read(summary_file, DataFrame)
		for r in eachrow(unique(s[!, [:beta, :hurst, :p_oneway, :L]]))
			push!(done, cell_id(r.beta, r.hurst, r.p_oneway, r.L))
		end
	end
	# small systems first, so complete scaling curves appear early
	cells = [(β, H, p, L) for L in SIZES for β in BETAS for H in HURSTS for p in P_ONEWAY
			 if cell_id(β, H, p, L) ∉ done]
	@info "square study: $(length(cells)) cells to do ($(length(done)) done), " *
		  "$N_REPS networks each, $(Threads.nthreads()) threads, output in $OUTDIR"

	t0 = time()
	for (k, (β, H, p, L)) in enumerate(cells)
		ci = findfirst(==(β), BETAS) * 1_000_000 + findfirst(==(H), HURSTS) * 10_000 +
			 findfirst(==(p), P_ONEWAY) * 100 + findfirst(==(L), SIZES)
		rows = Vector{Any}(undef, N_REPS)
		Threads.@threads :greedy for rep in 1:N_REPS
			rows[rep] = crack_one(β, H, p, L, rep, ci * 1000 + rep)
		end
		df = DataFrame([name => [r[i][2] for r in rows] for (i, (name, _)) in enumerate(rows[1])])
		CSV.write(summary_file, df; append = isfile(summary_file))
		elapsed = time() - t0
		@info "cell $(cell_id(β, H, p, L)) ($k/$(length(cells))): mean removals " *
			  "$(round(sum(df.n_removed) / nrow(df), digits = 1)), " *
			  "elapsed $(round(elapsed / 60, digits = 1)) min"
	end
	@info "done: $summary_file"
end

main()
