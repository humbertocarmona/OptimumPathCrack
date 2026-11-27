#=
Optimum path crack with an early-exit Dijkstra.

The network is stored once as flat (CSR) out- and in-edge arrays (`CrackNetwork`),
which are read-only and can be shared between threads. Each crack keeps its own
`removed` mask over edge ids instead of copying the graph, and its own
`DijkstraWorkspace`. The search stops as soon as the destination is settled, and only
the nodes it touched are reset (generation stamps), so each search costs time in
proportion to the region it explores, not the whole network.

Ties are broken at random:
- among equal-cost shortest paths, one is drawn uniformly over all of them (paths are
  counted on the shortest-path DAG, then sampled backwards from the destination);
- among equal-weight heaviest edges on the chosen path, one is drawn uniformly.
Path costs are compared with a relative tolerance `TIE_RTOL`, since costs are sums of
floating-point weights. Weights must be positive.
=#

const TIE_RTOL = 1e-10

"""
	CrackNetwork(g::SimpleDiGraph, weight_matrix)

Flat, read-only representation of `g` for cracking. Edge `e` goes from `srcs[e]` to
`targets[e]` with weight `weights[e]`; the out-edges of node `u` are
`offsets[u]:offsets[u+1]-1`, in the order of `outneighbors(g, u)`, and the in-edges of
`v` are `in_edges[in_offsets[v]:in_offsets[v+1]-1]`.
"""
struct CrackNetwork
	offsets::Vector{Int}
	srcs::Vector{Int}
	targets::Vector{Int}
	weights::Vector{Float64}
	in_offsets::Vector{Int}
	in_edges::Vector{Int}
end

function CrackNetwork(g::SimpleDiGraph, weight_matrix::SparseMatrixCSC{Float64, Int64})
	N, E = nv(g), ne(g)
	offsets = Vector{Int}(undef, N + 1)
	srcs = Vector{Int}(undef, E)
	targets = Vector{Int}(undef, E)
	weights = Vector{Float64}(undef, E)
	e = 1
	for u in 1:N
		offsets[u] = e
		for v in outneighbors(g, u)
			srcs[e], targets[e], weights[e] = u, v, weight_matrix[u, v]
			e += 1
		end
	end
	offsets[N+1] = e
	bad = findall(w -> !(w > 0), weights)
	if !isempty(bad)
		e = bad[1]
		throw(ArgumentError("edge weights must be positive: $(length(bad)) edge(s) are not, " *
							"e.g. ($(srcs[e]), $(targets[e])) has weight $(weights[e])"))
	end

	# in-edges, grouped by target
	in_offsets = zeros(Int, N + 1)
	for v in targets
		in_offsets[v+1] += 1
	end
	in_offsets[1] = 1
	cumsum!(in_offsets, in_offsets)
	in_edges = Vector{Int}(undef, E)
	fill_pos = copy(in_offsets)
	for e in 1:E
		v = targets[e]
		in_edges[fill_pos[v]] = e
		fill_pos[v] += 1
	end
	return CrackNetwork(offsets, srcs, targets, weights, in_offsets, in_edges)
end

Graphs.nv(net::CrackNetwork) = length(net.offsets) - 1
Graphs.ne(net::CrackNetwork) = length(net.targets)

# Per-crack scratch space; not shared between threads.
mutable struct DijkstraWorkspace
	dist::Vector{Float64}
	stamp::Vector{Int}           # node reached in this search iff stamp[v] == gen
	settled::Vector{Int}         # node settled in this search iff settled[v] == gen
	rank::Vector{Int}            # settle order in this search (valid if settled)
	gen::Int
	heap::Vector{Tuple{Float64, Int}}
	# shortest-path DAG (tie-breaking)
	in_dag::Vector{Int}          # node in the DAG iff in_dag[v] == gen
	sigma::Vector{Float64}       # number of shortest paths org -> v
	dag::Vector{Int}
end

DijkstraWorkspace(N::Int) = DijkstraWorkspace(zeros(N), zeros(Int, N), zeros(Int, N),
	zeros(Int, N), 0,
	Tuple{Float64, Int}[], zeros(Int, N), zeros(N), Int[])

# Binary min-heap on (distance, node).
function _heap_push!(h::Vector{Tuple{Float64, Int}}, x::Tuple{Float64, Int})
	push!(h, x)
	i = length(h)
	@inbounds while i > 1
		p = i >> 1
		h[p] <= x && break
		h[i] = h[p]
		i = p
	end
	@inbounds h[i] = x
	return h
end

function _heap_pop!(h::Vector{Tuple{Float64, Int}})
	@inbounds top = h[1]
	x = pop!(h)
	n = length(h)
	n == 0 && return top
	i = 1
	@inbounds while true
		c = 2i
		c > n && break
		if c < n && h[c+1] < h[c]
			c += 1
		end
		x <= h[c] && break
		h[i] = h[c]
		i = c
	end
	@inbounds h[i] = x
	return top
end

"""
	_dijkstra_to!(ws, net, removed, org, dst) -> Float64

Shortest-path cost from `org` to `dst` ignoring edges with `removed[e]`, or `Inf`.
Stops when `dst` is settled. Every node closer than `dst` is then settled, which is
all `_sample_path!` needs.
"""
function _dijkstra_to!(ws::DijkstraWorkspace, net::CrackNetwork, removed::BitVector,
	org::Int, dst::Int)
	gen = (ws.gen += 1)
	dist, stamp, settled, rank, heap = ws.dist, ws.stamp, ws.settled, ws.rank, ws.heap
	offsets, targets, weights = net.offsets, net.targets, net.weights
	empty!(heap)
	n_settled = 0

	@inbounds begin
		stamp[org] = gen
		dist[org] = 0.0
		_heap_push!(heap, (0.0, org))
		while !isempty(heap)
			d, u = _heap_pop!(heap)
			(d > dist[u] || settled[u] == gen) && continue   # stale entry
			settled[u] = gen
			rank[u] = (n_settled += 1)
			u == dst && return d
			for e in offsets[u]:(offsets[u+1]-1)
				removed[e] && continue
				v = targets[e]
				nd = d + weights[e]
				if stamp[v] != gen || nd < dist[v]
					stamp[v] = gen
					dist[v] = nd
					_heap_push!(heap, (nd, v))
				end
			end
		end
	end
	return Inf
end

# Edge e = (u -> v) lies on a shortest path to v (within tolerance). u must also have
# been settled before v: with weights far below the float resolution of the path cost
# (very strong disorder), dist[u] + w can equal dist[v] around a cycle, and the
# tolerance alone would let the backward walk loop forever. Settle order is a
# topological order of the true shortest-path DAG, so no genuine predecessor is lost.
@inline function _tight(ws::DijkstraWorkspace, net::CrackNetwork, removed::BitVector,
	e::Int, v::Int)
	@inbounds begin
		removed[e] && return false
		u = net.srcs[e]
		ws.settled[u] == ws.gen || return false
		ws.rank[u] < ws.rank[v] || return false
		return ws.dist[u] + net.weights[e] <= ws.dist[v] * (1 + TIE_RTOL)
	end
end

"""
	_sample_path!(path, ws, net, removed, org, dst, rng)

After `_dijkstra_to!` reached `dst`, fills `path` with the edge ids of one shortest
path, from `dst` back to `org`, drawn uniformly among all shortest paths.
"""
function _sample_path!(path::Vector{Int}, ws::DijkstraWorkspace, net::CrackNetwork,
	removed::BitVector, org::Int, dst::Int, rng::AbstractRNG)
	gen = ws.gen
	in_dag, sigma, dag, dist = ws.in_dag, ws.sigma, ws.dag, ws.dist
	in_offsets, in_edges, srcs = net.in_offsets, net.in_edges, net.srcs
	empty!(path)

	# 1. Collect the nodes that reach dst through tight edges (the shortest-path DAG).
	#    Without ties this is just the path itself.
	empty!(dag)
	push!(dag, dst)
	in_dag[dst] = gen
	k = 1
	ties = false
	@inbounds while k <= length(dag)
		v = dag[k]
		k += 1
		n_tight = 0
		for p in in_offsets[v]:(in_offsets[v+1]-1)
			e = in_edges[p]
			_tight(ws, net, removed, e, v) || continue
			n_tight += 1
			u = srcs[e]
			if in_dag[u] != gen
				in_dag[u] = gen
				push!(dag, u)
			end
		end
		v != org && n_tight > 1 && (ties = true)
	end

	# 2. Count shortest paths from org to each DAG node, in settle (topological) order.
	if ties
		sort!(dag; by = v -> ws.rank[v])
		@inbounds for v in dag
			if v == org
				sigma[v] = 1.0
				continue
			end
			s = 0.0
			for p in in_offsets[v]:(in_offsets[v+1]-1)
				e = in_edges[p]
				_tight(ws, net, removed, e, v) && (s += sigma[srcs[e]])
			end
			sigma[v] = s
		end
	end

	# 3. Walk back from dst, choosing each predecessor with probability σ[u]/σ[v].
	v = dst
	@inbounds while v != org
		chosen = 0
		if ties
			r = rand(rng) * sigma[v]
			for p in in_offsets[v]:(in_offsets[v+1]-1)
				e = in_edges[p]
				_tight(ws, net, removed, e, v) || continue
				chosen = e
				r -= sigma[srcs[e]]
				r < 0 && break
			end
		else
			for p in in_offsets[v]:(in_offsets[v+1]-1)
				e = in_edges[p]
				if _tight(ws, net, removed, e, v)
					chosen = e
					break
				end
			end
		end
		push!(path, chosen)
		v = srcs[chosen]
	end
	return path
end

"""
	_crack(net, org, dst; nMaxRem = 0, maxCostRatio = Inf, record_paths = true,
		   rng = Xoshiro(1), removal_weights = nothing)

Core crack loop on a `CrackNetwork`. Paths are optimal for `net.weights`; the edge removed
from each path is the one with the largest `removal_weights[e]` (by default the same
weights). Returns a named tuple
`(removed, path_first, path_costs, stop_reason)`, where `removed` lists the removed edge
ids in order, and `path_first[e] = k` if edge `e` first appeared on the `k`-th path
(empty unless `record_paths`). See `crackOptimalPaths` for the other fields.
"""
function _crack(net::CrackNetwork, org::Int, dst::Int;
	nMaxRem::Int = 0, maxCostRatio::Float64 = Inf, record_paths::Bool = true,
	rng::AbstractRNG = Random.Xoshiro(1),
	removal_weights::Union{Nothing, AbstractVector{<:Real}} = nothing)

	nMaxRem == 0 && (nMaxRem = ne(net))
	rw = removal_weights === nothing ? net.weights : removal_weights
	length(rw) == ne(net) || throw(ArgumentError("removal_weights needs one value per edge"))
	removed_list = Int[]
	path_first = Dict{Int, Int}()
	path_costs = Float64[]
	result(reason) = (removed = removed_list, path_first = path_first,
		path_costs = path_costs, stop_reason = reason)

	org == dst && return result(:same_node)

	ws = DijkstraWorkspace(nv(net))
	removed = falses(ne(net))
	path = Int[]
	cost = _dijkstra_to!(ws, net, removed, org, dst)
	isinf(cost) && return result(:no_path)
	first_cost = cost
	push!(path_costs, cost)

	while true
		length(removed_list) >= nMaxRem && return result(:max_removals)

		_sample_path!(path, ws, net, removed, org, dst, rng)

		# Heaviest edge on the path; uniform among equal weights (reservoir sampling)
		k = length(removed_list) + 1
		emax, wmax, n_max = 0, -Inf, 0
		for e in path
			record_paths && get!(path_first, e, k)
			w = rw[e]
			if w > wmax
				emax, wmax, n_max = e, w, 1
			elseif w == wmax
				n_max += 1
				rand(rng, 1:n_max) == 1 && (emax = e)
			end
		end

		removed[emax] = true
		push!(removed_list, emax)

		cost = _dijkstra_to!(ws, net, removed, org, dst)
		isinf(cost) && return result(:disconnected)
		push!(path_costs, cost)
		cost > maxCostRatio * first_cost && return result(:cost_ratio)
	end
end

# Network to route on and weights to remove by. `route_by = :weight` routes and removes
# by `weight_matrix` (or removes by `removal_matrix`). `route_by = :time` reads
# `weight_matrix` as the pace 1/speed (time per unit length): it routes by the travel
# time length × pace and removes the edge of largest pace (the slowest).
function _crack_setup(net::CrackNetwork, removal_matrix, route_by::Symbol, distance_matrix)
	route_by in (:weight, :time) ||
		throw(ArgumentError("route_by must be :weight or :time, got :$route_by"))
	if route_by == :time
		removal_matrix === nothing ||
			throw(ArgumentError("route_by = :time removes the slowest edge; do not also " *
								"give removal_matrix"))
		distance_matrix === nothing &&
			throw(ArgumentError("route_by = :time needs the distance matrix"))
		len = edgeValues(net, distance_matrix)
		all(>(0), len) || throw(ArgumentError("route_by = :time needs positive edge lengths"))
		tnet = CrackNetwork(net.offsets, net.srcs, net.targets, len .* net.weights,
			net.in_offsets, net.in_edges)
		return tnet, net.weights
	end
	return net, removal_matrix === nothing ? nothing : edgeValues(net, removal_matrix)
end

"""
	crackOptimalPaths(g::SimpleDiGraph, org::Int64, dst::Int64,
					  weight_matrix::SparseMatrixCSC{Float64, Int64};
					  nMaxRem::Int64 = 0, maxCostRatio::Float64 = Inf,
					  rng::AbstractRNG = Xoshiro(1))

Iteratively finds the shortest path from `org` to `dst` and removes its
highest-weight edge, on a copy of `g`.

# Arguments
- `g::SimpleDiGraph`: The input directed graph (not modified).
- `org::Int64`: The origin node.
- `dst::Int64`: The destination node.
- `weight_matrix::SparseMatrixCSC{Float64, Int64}`: Edge weights (must be positive).
- `nMaxRem::Int64` (optional): Maximum number of edges to remove. `0` means `ne(g)`.
- `maxCostRatio::Float64` (optional): Stop when the cost of the current shortest
  path exceeds `maxCostRatio` times the cost of the first shortest path. The path
  that triggers the stop is not cracked. Defaults to `Inf` (no limit).
- `route_by` (optional): `:weight` (default) finds the shortest path by `weight_matrix`
  and removes its heaviest edge. `:time` reads `weight_matrix` as the pace, 1 / speed
  (time per unit length), as in Sousa et al.: it finds the fastest path by travel time,
  `distance_matrix[i, j] × weight_matrix[i, j]` = length / speed, and removes its
  slowest edge (largest pace). Neither step then depends on how roads are cut into
  segments (splitting a road keeps its travel time and its speed), whereas the sum of
  1 / speed grows with the number of segments. Needs `distance_matrix`. Path costs are
  then travel times.
- `distance_matrix` (optional): edge lengths, needed for `route_by = :time`.
- `removal_matrix` (optional): any other removal criterion: the edge removed from each
  shortest path is the one with the largest `removal_matrix[i, j]`. Paths stay optimal
  for `weight_matrix`.
- `rng::AbstractRNG` (optional): Random generator for tie-breaking. When several
  shortest paths have the same cost, one is drawn uniformly among them; when several
  edges on the path share the largest weight, one is drawn uniformly. The default is
  a fresh `Xoshiro(1)`, so repeated calls give the same result.

# Returns
- `gr::SimpleDiGraph`: The cracked copy of `g`.
- `removed_edges::SparseMatrixCSC{Float64, Int64}`: `removed_edges[i,j] = k` if edge
  `(i,j)` was the `k`-th edge removed.
- `path_edges::SparseMatrixCSC{Float64, Int64}`: `path_edges[i,j] = k` if edge `(i,j)`
  first appeared on the `k`-th cracked path.
- `info::NamedTuple`: `(path_costs, stop_reason)`. `path_costs[k]` is the cost of the
  `k`-th shortest path found (it includes the path that triggered a `:cost_ratio`
  stop). `stop_reason` is one of `:disconnected` (no path left), `:cost_ratio`,
  `:max_removals`, `:no_path` (no path from the start) or `:same_node`.

For many OD pairs on the same network use `crackStatistics`, which builds the
`CrackNetwork` once and runs the pairs in parallel.
"""
function crackOptimalPaths(g::SimpleDiGraph, org::Int64, dst::Int64,
	weight_matrix::SparseMatrixCSC{Float64, Int64};
	nMaxRem::Int64 = 0,
	maxCostRatio::Float64 = Inf,
	route_by::Symbol = :weight,
	distance_matrix::Union{Nothing, SparseMatrixCSC{Float64, Int64}} = nothing,
	removal_matrix::Union{Nothing, SparseMatrixCSC{Float64, Int64}} = nothing,
	rng::AbstractRNG = Random.Xoshiro(1))

	net, rw = _crack_setup(CrackNetwork(g, weight_matrix), removal_matrix, route_by,
		distance_matrix)
	res = _crack(net, org, dst; nMaxRem = nMaxRem, maxCostRatio = maxCostRatio, rng = rng,
		removal_weights = rw)

	N = nv(g)
	gr = copy(g)
	for e in res.removed
		rem_edge!(gr, net.srcs[e], net.targets[e])
	end
	removed_edges = sparse(net.srcs[res.removed], net.targets[res.removed],
		Float64.(1:length(res.removed)), N, N)
	pe = collect(keys(res.path_first))
	path_edges = sparse(net.srcs[pe], net.targets[pe],
		Float64[res.path_first[e] for e in pe], N, N)

	return gr, removed_edges, path_edges,
	(path_costs = res.path_costs, stop_reason = res.stop_reason)
end
