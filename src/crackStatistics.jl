"""
	crackStatistics(g, coords, distance_matrix, weight_matrix, ℓ;
					n_od_pairs = 100, seed = 1, δ = 0.01, cellWidth = 100.0,
					maxCostRatio = Inf, nMaxRem = 0, cell_list = nothing)

Runs the optimum-path-crack experiment for OD distance `ℓ` (meters), or for each
distance in a vector `ℓ`. For each OD pair from `create_ODs`, the network is cracked
with `crackOptimalPaths`, starting each time from an untouched `g`. Only pairs in the
same strongly connected component are used, so every pair starts with a path.

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
its own OD sample.
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
	cell_list::Union{Nothing, AbstractDict} = nothing)

	if cell_list === nothing
		cell_list = cellList_lat_lon(coords; cellWidth = cellWidth)
	end

	# One flat list of (ℓ, od_id, origin, destination) tasks over all distances
	tasks = Tuple{Float64, Int, Int, Int}[]
	for (k, ℓ) in enumerate(ℓs)
		ODs = create_ODs(Float64(ℓ), cell_list; seed = seed + k - 1,
			n_od_pairs = n_od_pairs, δ = δ, graph = g)
		for (od_id, (o, d)) in enumerate(ODs)
			push!(tasks, (Float64(ℓ), od_id, o, d))
		end
	end

	# Crack in parallel on one shared, read-only network; each task writes only its
	# own slot.
	net = CrackNetwork(g, weight_matrix)
	results = Vector{Any}(undef, length(tasks))
	Threads.@threads :greedy for t in eachindex(tasks)
		_, _, o, d = tasks[t]
		results[t] = _crack(net, o, d; nMaxRem = nMaxRem, maxCostRatio = maxCostRatio,
			record_paths = false)
	end

	# Assemble the tables serially, in task order (deterministic)
	edges_df = DataFrame(ℓ = Float64[], od_id = Int[], origin = Int[], destination = Int[],
		order = Int[], src = Int[], dst = Int[], length = Float64[], weight = Float64[],
		path_cost = Float64[])
	od_df = DataFrame(ℓ = Float64[], od_id = Int[], origin = Int[], destination = Int[],
		od_distance = Float64[], origin_outdegree = Int[], destination_indegree = Int[],
		n_removed = Int[], first_cost = Float64[], last_cost = Float64[],
		cost_ratio = Float64[], stop_reason = Symbol[])

	for (t, (ℓ, od_id, o, d)) in enumerate(tasks)
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

	return edges_df, od_df
end

function crackStatistics(g::SimpleDiGraph,
	coords::Vector{Tuple{Float64, Float64}},
	distance_matrix::SparseMatrixCSC{Float64, Int64},
	weight_matrix::SparseMatrixCSC{Float64, Int64},
	ℓ::Real; kwargs...)
	crackStatistics(g, coords, distance_matrix, weight_matrix, [ℓ]; kwargs...)
end
