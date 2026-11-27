"""
	crackStatistics(g, coords, distance_matrix, weight_matrix, ℓ;
					n_od_pairs = 100, seed = 1, δ = 0.01, cellWidth = 100.0,
					maxCostRatio = Inf, nMaxRem = 0, cell_list = nothing,
					route_by = :weight, removal_matrix = nothing)

Runs the optimum-path-crack experiment for OD distance `ℓ` (meters), or for each
distance in a vector `ℓ`. For each OD pair from `create_ODs`, the network is cracked
with `crackOptimalPaths`, starting each time from an untouched `g`. Only pairs in the
same strongly connected component are used, so every pair starts with a path.
`route_by = :time` reads `weight_matrix` as the pace 1 / speed: it routes by travel
time (`distance_matrix` × pace) and removes the slowest edge; path costs are then travel
times, while the `weight` column of `edges_df` stays the pace. `removal_matrix` sets any
other removal criterion. See `crackOptimalPaths`.

OD pairs are cracked in parallel on `Threads.nthreads()` threads (start Julia with
`-t auto` or `JULIA_NUM_THREADS`). Results do not depend on the number of threads.

# Returns
`(edges_df, od_df)`:
- `edges_df`: one row per removed edge. Columns `ℓ, od_id, origin, destination,
  order, src, dst, length, weight, path_cost` (`path_cost` is the cost of the path
  the edge was removed from).
- `od_df`: one row per OD pair. Columns `ℓ, od_id, origin, destination, od_distance,
  origin_outdegree, destination_indegree, n_removed, first_cost, last_cost,
  cost_ratio, stop_reason`. `last_cost` is the cost of the last shortest path found
  before the pair disconnected (or the path that exceeded `maxCostRatio`, when
  `stop_reason == :cost_ratio`).

With a vector `ℓ`, the `k`-th distance uses seed `seed + k - 1`, so each distance gets
its own OD sample. Ties (equal-cost paths, equal-weight heaviest edges) are broken at
random, with a generator seeded from `(seed, k, od_id)` for each pair, so results are
reproducible and independent of the number of threads.

The run settings (distances, seeds, tolerances, network size, package and Julia
versions, date) are attached to both tables as metadata (`DataFrames.metadata`), and
`writeCrackStatistics` saves them alongside the tables.
"""
function crackStatistics(g::SimpleDiGraph,
	coords::Vector{Tuple{Float64, Float64}},
	distance_matrix::SparseMatrixCSC{Float64, Int64},
	weight_matrix::SparseMatrixCSC{Float64, Int64},
	ℓs::AbstractVector{<:Real};
	n_od_pairs::Int64 = 100,
	seed::Int64 = 1,
	δ::Float64 = 0.01,
	cellWidth::Float64 = 100.0,
	maxCostRatio::Float64 = Inf,
	nMaxRem::Int64 = 0,
	cell_list::Union{Nothing, AbstractDict} = nothing,
	route_by::Symbol = :weight,
	removal_matrix::Union{Nothing, SparseMatrixCSC{Float64, Int64}} = nothing)

	if cell_list === nothing
		cell_list = cellList_lat_lon(coords; cellWidth = cellWidth)
	end

	# One flat list of (ℓ index, od_id, origin, destination) tasks over all distances
	tasks = NTuple{4, Int}[]
	for (k, ℓ) in enumerate(ℓs)
		ODs = create_ODs(Float64(ℓ), cell_list; seed = seed + k - 1,
			n_od_pairs = n_od_pairs, δ = δ, graph = g)
		for (od_id, (o, d)) in enumerate(ODs)
			push!(tasks, (k, od_id, o, d))
		end
	end

	# Crack in parallel on one shared, read-only network; each task writes only its
	# own slot.
	net, rw = _crack_setup(CrackNetwork(g, weight_matrix), removal_matrix, route_by,
		distance_matrix)
	results = Vector{Any}(undef, length(tasks))
	Threads.@threads :greedy for t in eachindex(tasks)
		k, od_id, o, d = tasks[t]
		rng = Random.Xoshiro(UInt64[seed % UInt64, k, od_id])
		results[t] = _crack(net, o, d; nMaxRem = nMaxRem, maxCostRatio = maxCostRatio,
			record_paths = false, rng = rng, removal_weights = rw)
	end

	# Assemble the tables serially, in task order (deterministic)
	edges_df = DataFrame(ℓ = Float64[], od_id = Int[], origin = Int[], destination = Int[],
		order = Int[], src = Int[], dst = Int[], length = Float64[], weight = Float64[],
		path_cost = Float64[])
	od_df = DataFrame(ℓ = Float64[], od_id = Int[], origin = Int[], destination = Int[],
		od_distance = Float64[], origin_outdegree = Int[], destination_indegree = Int[],
		n_removed = Int[], first_cost = Float64[], last_cost = Float64[],
		cost_ratio = Float64[], stop_reason = Symbol[])

	for (t, (k, od_id, o, d)) in enumerate(tasks)
		ℓ = Float64(ℓs[k])
		res = results[t]
		costs = res.path_costs
		for (k, e) in enumerate(res.removed)
			i, j = net.srcs[e], net.targets[e]
			push!(edges_df, (ℓ, od_id, o, d, k, i, j,
				distance_matrix[i, j], weight_matrix[i, j], costs[k]))
		end
		first_cost = isempty(costs) ? NaN : costs[1]
		last_cost = isempty(costs) ? NaN : costs[end]
		od_distance = euclidean_distance(LLA(coords[o]..., 0.0), LLA(coords[d]..., 0.0))
		push!(od_df, (ℓ, od_id, o, d, od_distance, outdegree(g, o), indegree(g, d),
			length(res.removed), first_cost, last_cost, last_cost / first_cost, res.stop_reason))
	end

	params = Dict{String, Any}(
		"distances" => Float64.(collect(ℓs)),
		"n_od_pairs" => n_od_pairs,
		"seed" => seed,
		"delta" => δ,
		"cellWidth" => Float64(get(cell_list, "cellWidth", cellWidth)),
		"maxCostRatio" => maxCostRatio, "route_by" => string(route_by),
		"nMaxRem" => nMaxRem,
		"tie_breaking" => "uniform random (equal-cost paths, then equal-weight edges)",
		"n_nodes" => nv(g),
		"n_edges" => ne(g),
		"package_version" => string(pkgversion(@__MODULE__)),
		"julia_version" => string(VERSION),
		"created" => string(Dates.now()),
	)
	for df in (edges_df, od_df), (key, val) in params
		metadata!(df, key, val; style = :note)
	end

	return edges_df, od_df
end

function crackStatistics(g::SimpleDiGraph,
	coords::Vector{Tuple{Float64, Float64}},
	distance_matrix::SparseMatrixCSC{Float64, Int64},
	weight_matrix::SparseMatrixCSC{Float64, Int64},
	ℓ::Real; kwargs...)
	crackStatistics(g, coords, distance_matrix, weight_matrix, [ℓ]; kwargs...)
end

"""
	writeCrackStatistics(prefix, edges_df, od_df; extra = Dict())

Saves the output of `crackStatistics` as `prefix_edges.csv`, `prefix_ods.csv` and
`prefix_params.toml`. The TOML file holds the run settings stored in the tables'
metadata, plus anything in `extra` (for example the network file names), under
`[extra]`. Returns the three file names.

```julia
edges_df, od_df = crackStatistics(g, coords, dm, wm, [1000.0, 3000.0]; n_od_pairs = 500)
writeCrackStatistics("results/boston_4h", edges_df, od_df;
					 extra = Dict("edges_file" => "boston-edges-4h.csv"))
```
"""
function writeCrackStatistics(prefix::AbstractString, edges_df::DataFrame, od_df::DataFrame;
	extra::AbstractDict = Dict{String, Any}())
	edges_file = "$(prefix)_edges.csv"
	ods_file = "$(prefix)_ods.csv"
	params_file = "$(prefix)_params.toml"
	dir = dirname(prefix)
	isempty(dir) || mkpath(dir)

	CSV.write(edges_file, edges_df)
	CSV.write(ods_file, od_df)

	params = Dict{String, Any}(k => metadata(od_df, k) for k in metadatakeys(od_df))
	isempty(extra) || (params["extra"] = Dict{String, Any}(string(k) => v for (k, v) in extra))
	open(params_file, "w") do io
		TOML.print(io, params; sorted = true)
	end
	@info "saved $edges_file, $ods_file, $params_file"
	return edges_file, ods_file, params_file
end

"""
	readCrackStatistics(prefix) -> (edges_df, od_df, params)

Reads files written by `writeCrackStatistics`. `stop_reason` is converted back to
`Symbol`, and the settings from `prefix_params.toml` are returned as a `Dict` and also
attached to both tables as metadata.
"""
function readCrackStatistics(prefix::AbstractString)
	edges_df = CSV.read("$(prefix)_edges.csv", DataFrame)
	od_df = CSV.read("$(prefix)_ods.csv", DataFrame)
	od_df.stop_reason = Symbol.(od_df.stop_reason)
	params = TOML.parsefile("$(prefix)_params.toml")
	for df in (edges_df, od_df), (key, val) in params
		key == "extra" && continue
		metadata!(df, key, val; style = :note)
	end
	return edges_df, od_df, params
end
