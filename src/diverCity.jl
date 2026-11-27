#=
DiverCity (Cornacchia, Pappalardo, Nanni, Pedreschi & González, arXiv:2510.02582):
potential route diversification of a trip,

	D(o, d) = S(NSR) · |NSR|,

where NSR is the set of near-shortest routes and S = 1 - J their spatial spread (J = mean
weighted Jaccard similarity over pairs of routes, edges weighted by length).

Alternatives come from path penalization: find the optimal path, multiply the weight of
each of its edges by (1 + p), repeat (cumulatively) until k distinct routes are found.
The near-shortest routes are those whose cost under the original weights is at most
(1 + ε) times that of the optimal path. Defaults are the paper's: k = 10, p = 0.1,
ε = 0.3. The paper weights edges by travel time (length / speed); pass the routing
weights through the `CrackNetwork` (or `weights`) and the lengths through `lengths`.
=#

"""
	edgeValues(net::CrackNetwork, M::SparseMatrixCSC) -> Vector{Float64}

Values `M[src, dst]` for the edges of `net`, in edge-id order (e.g. lengths from a
distance matrix).
"""
edgeValues(net::CrackNetwork, M::SparseMatrixCSC) =
	[M[net.srcs[e], net.targets[e]] for e in 1:ne(net)]

"""
	diverCity(net::CrackNetwork, org, dst; k = 10, p = 0.1, epsilon = 0.3,
			  lengths = ones(ne(net)), weights = net.weights, max_iter = 100k,
			  rng = Xoshiro(1)) -> NamedTuple

DiverCity of the trip `org → dst` by path penalization (see the file header).

Routes are found with `weights` (positive; defaults to the network's) and compared by
their cost under the same weights. Ties between equal-cost paths are broken uniformly at
random, as in the crack. A route found again after a penalization is not counted twice;
the search stops after `k` distinct routes or `max_iter` iterations.

Returns `(D, n_nsr, spread, n_routes, costs, routes)`: `D = spread · n_nsr`; `n_nsr` the
number of near-shortest routes; `spread = 1 - J` (0 when `n_nsr ≤ 1`); `n_routes` the
number of distinct routes found; `costs` their original costs, in the order found; and
`routes` their edge ids. `D = NaN` if there is no path.
"""
function diverCity(net::CrackNetwork, org::Int, dst::Int;
	k::Int = 10, p::Float64 = 0.1, epsilon::Float64 = 0.3,
	lengths::AbstractVector{<:Real} = ones(ne(net)),
	weights::AbstractVector{<:Real} = net.weights,
	max_iter::Int = 100k, rng::AbstractRNG = Random.Xoshiro(1))

	k >= 1 && p > 0 && epsilon >= 0 || throw(ArgumentError("need k ≥ 1, p > 0, epsilon ≥ 0"))
	length(lengths) == ne(net) && length(weights) == ne(net) ||
		throw(ArgumentError("lengths and weights need one value per edge"))
	routes = Vector{Int}[]
	costs = Float64[]
	none = (D = NaN, n_nsr = 0, spread = NaN, n_routes = 0, costs = costs, routes = routes)
	org == dst && return none

	w0 = Vector{Float64}(weights)
	pnet = CrackNetwork(net.offsets, net.srcs, net.targets, copy(w0), net.in_offsets,
		net.in_edges)
	ws = DijkstraWorkspace(nv(net))
	removed = falses(ne(net))
	path = Int[]
	seen = Set{Vector{Int}}()
	for _ in 1:max_iter
		isinf(_dijkstra_to!(ws, pnet, removed, org, dst)) && return none
		_sample_path!(path, ws, pnet, removed, org, dst, rng)
		key = sort(path)
		if !(key in seen)
			push!(seen, key)
			push!(routes, copy(path))
			push!(costs, sum(w0[e] for e in path))
		end
		for e in path
			pnet.weights[e] *= 1 + p
		end
		length(routes) == k && break
	end

	best = minimum(costs)
	nsr = [r for (r, c) in zip(routes, costs) if c <= (1 + epsilon) * best * (1 + TIE_RTOL)]
	n = length(nsr)
	spread = n <= 1 ? 0.0 : 1 - _mean_jaccard(nsr, lengths)
	return (D = spread * n, n_nsr = n, spread = spread, n_routes = length(routes),
		costs = costs, routes = routes)
end

# Mean weighted Jaccard similarity over all pairs of routes (edges weighted by `len`).
function _mean_jaccard(routes::Vector{Vector{Int}}, len::AbstractVector{<:Real})
	sets = [Set(r) for r in routes]
	tot = [sum(len[e] for e in s) for s in sets]
	J, m = 0.0, 0
	for a in 1:length(sets), b in (a+1):length(sets)
		inter = sum((len[e] for e in sets[a] if e in sets[b]); init = 0.0)
		J += inter / (tot[a] + tot[b] - inter)
		m += 1
	end
	return J / m
end

"""
	diverCityStatistics(net::CrackNetwork, ods; seed = 1, kwargs...) -> DataFrame

`diverCity` for each `(org, dst)` in `ods`, in parallel over threads. Pair `i` uses the
random stream `Xoshiro(seed, i)`, so results do not depend on the number of threads.
Keyword arguments are passed to `diverCity`. Columns: `org, dst, D, n_nsr, spread,
n_routes, first_cost`.
"""
function diverCityStatistics(net::CrackNetwork, ods::AbstractVector{<:Tuple{Int, Int}};
	seed::Int = 1, kwargs...)
	n = length(ods)
	out = Vector{NamedTuple}(undef, n)
	Threads.@threads :dynamic for i in 1:n
		o, d = ods[i]
		r = diverCity(net, o, d; rng = Random.Xoshiro(UInt64[seed % UInt64, i % UInt64]),
			kwargs...)
		out[i] = (org = o, dst = d, D = r.D, n_nsr = r.n_nsr, spread = r.spread,
			n_routes = r.n_routes, first_cost = isempty(r.costs) ? NaN : r.costs[1])
	end
	return DataFrame(out)
end
