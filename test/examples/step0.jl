#=
Example: step 0 of the city-lattice study (victor2026), the uniform two-way grid.

A periodic n × n lattice of 100 m blocks with every road two-way, no fast roads, no dead
ends and no road-to-road variation; block travel times have mean 1 and only a weak
uncorrelated disorder of strength β, which breaks the ties of the uniform grid. For OD
pairs at distance L (in blocks) it measures

  η   = efficiency, (L / v₀) / T₁ with v₀ = 1 block per unit time (`routeEfficiency`);
  N_ε = removals made while the fastest route takes at most (1 + ε) T₁ (`removalsWithin`),
        from one crack stopped at (1 + max ε) T₁;

and summarises a lattice by R_ε, the mean of N_ε over L.

With the default arguments this reproduces the runs of the analysis repository
(OPCAnalysis/scripts/lattice_steps.jl --tag=step0 --vary=beta=0.01,0.03,0.1,0.3
--seeds=1:4 --pairs=200): the same lattices, OD pairs and random streams. That takes
about 20 minutes per (β, seed) on 8 threads; use smaller `n`, `Ls` and `pairs` to try it:

	julia -t auto --project=. -e 'include("test/examples/step0.jl"); step0(n = 64, Ls = [5, 10], pairs = 20)'

Returns a DataFrame with one row per OD pair.
=#

using OptimumPathCrack
using Graphs, Random, DataFrames

mean_(x) = sum(x) / length(x)

"""Keyword arguments of `buildCityLattice` for step 0 (the uniform two-way grid)."""
const STEP0 = (oneway_fraction = 0.0, alternation = 0.5, arterial_fraction = 0.0,
	arterial_spacing = 0, loop_fraction = 1.0, arterial_speedup = 2.0, arterials_twoway = true,
	road_cv = 0.0, hurst = -1.0, direction_rho = 0.0, arterial_segment_cv = 0.02,
	mean_degree = 4.0, weight_mean = 1.0)

function step0(; n = 256, Ls = [5, 10, 20, 30, 40, 50, 60], betas = [0.01, 0.03, 0.1, 0.3],
	seeds = 1:4, pairs = 200, eps = [0.1, 0.2, 0.3, 0.5, 1.0], verbose = true)
	out = DataFrame()
	for β in betas, seed in seeds
		g, _, _, wm, _ = buildCityLattice(n, n; STEP0..., beta = β, pbc = true, block = 100.0,
			seed = seed)
		net = CrackNetwork(g, wm)
		ok = falses(nv(g))
		ok[argmax(length, strongly_connected_components(g))] .= true
		rng = Random.Xoshiro(UInt64[seed, 7])
		trips = [(L, j, od...) for L in Ls for (j, od) in enumerate(torusODs(n, L, pairs, ok, rng))]
		rows = Vector{NamedTuple}(undef, length(trips))
		Threads.@threads :dynamic for t in eachindex(trips)
			L, j, o, d, dx, dy = trips[t]
			res = OptimumPathCrack._crack(net, o, d; maxCostRatio = 1 + maximum(eps),
				record_paths = false, rng = Random.Xoshiro(UInt64[seed, 11, t]))
			T1 = res.path_costs[1]
			ne = removalsWithin(res.path_costs, length(res.removed), eps)
			rows[t] = merge((beta = β, seed = seed, L = L, pair = j, dist = hypot(dx, dy),
					T1 = T1, eta = routeEfficiency(T1, hypot(dx, dy))),
				NamedTuple{Tuple(Symbol("ne_e", replace(string(e), "." => "p")) for e in eps)}(Tuple(ne)))
		end
		df = DataFrame(rows)
		append!(out, df)
		if verbose
			per_L = combine(groupby(df, :L), :ne_e0p3 => mean_ => :m)
			println("β = $β, seed $seed: η ", round(mean_(df.eta), digits = 3),
				", R_0.3 ", round(mean_(per_L.m), digits = 1))
		end
	end
	return out
end

if abspath(PROGRAM_FILE) == @__FILE__
	step0()
end
