#=
Optimum path crack with an early-exit Dijkstra.

The network is stored once as flat (CSR) out-edge arrays (`CrackNetwork`), which are
read-only and can be shared between threads. Each crack keeps its own `removed` mask
over edge ids instead of copying the graph, and its own `DijkstraWorkspace`. The
search stops as soon as the destination is settled, and only the nodes it touched
are reset (generation stamps), so each search costs time in proportion to the
region it explores, not the whole network.
=#

"""
	CrackNetwork(g::SimpleDiGraph, weight_matrix)

Flat, read-only representation of `g` for cracking. Edge `e` goes from `srcs[e]` to
`targets[e]` with weight `weights[e]`; the out-edges of node `u` are
`offsets[u]:offsets[u+1]-1`, in the order of `outneighbors(g, u)`.
"""
struct CrackNetwork
	offsets::Vector{Int}
	srcs::Vector{Int}
	targets::Vector{Int}
	weights::Vector{Float64}
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
	return CrackNetwork(offsets, srcs, targets, weights)
end

Graphs.nv(net::CrackNetwork) = length(net.offsets) - 1
Graphs.ne(net::CrackNetwork) = length(net.targets)

# Per-crack scratch space; not shared between threads.
mutable struct DijkstraWorkspace
	dist::Vector{Float64}
	parent_edge::Vector{Int}
	stamp::Vector{Int}           # node is valid in this search iff stamp[v] == gen
	gen::Int
	heap::Vector{Tuple{Float64, Int}}
end

DijkstraWorkspace(N::Int) = DijkstraWorkspace(zeros(N), zeros(Int, N), zeros(Int, N), 0,
	Tuple{Float64, Int}[])

# Binary min-heap on (distance, node); ties are broken by node index.
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
Stops when `dst` is settled; the path can then be read from `ws.parent_edge`.
"""
function _dijkstra_to!(ws::DijkstraWorkspace, net::CrackNetwork, removed::BitVector,
	org::Int, dst::Int)
	gen = (ws.gen += 1)
	dist, parent_edge, stamp, heap = ws.dist, ws.parent_edge, ws.stamp, ws.heap
	offsets, targets, weights = net.offsets, net.targets, net.weights
	empty!(heap)

	@inbounds begin
		stamp[org] = gen
		dist[org] = 0.0
		parent_edge[org] = 0
		_heap_push!(heap, (0.0, org))
		while !isempty(heap)
			d, u = _heap_pop!(heap)
			d > dist[u] && continue      # stale entry
			u == dst && return d
			for e in offsets[u]:(offsets[u+1]-1)
				removed[e] && continue
				v = targets[e]
				nd = d + weights[e]
				if stamp[v] != gen || nd < dist[v]
					stamp[v] = gen
					dist[v] = nd
					parent_edge[v] = e
					_heap_push!(heap, (nd, v))
				end
			end
		end
	end
	return Inf
end

"""
	_crack(net, org, dst; nMaxRem = 0, maxCostRatio = Inf, record_paths = true)

Core crack loop on a `CrackNetwork`. Returns a named tuple
`(removed, path_first, path_costs, stop_reason)`, where `removed` lists the removed edge
ids in order, and `path_first[e] = k` if edge `e` first appeared on the `k`-th path
(empty unless `record_paths`). See `crackOptimalPaths` for the other fields.
"""
function _crack(net::CrackNetwork, org::Int, dst::Int;
	nMaxRem::Int = 0, maxCostRatio::Float64 = Inf, record_paths::Bool = true)

	nMaxRem == 0 && (nMaxRem = ne(net))
	removed_list = Int[]
	path_first = Dict{Int, Int}()
	path_costs = Float64[]
	result(reason) = (removed = removed_list, path_first = path_first,
		path_costs = path_costs, stop_reason = reason)

	org == dst && return result(:same_node)

	ws = DijkstraWorkspace(nv(net))
	removed = falses(ne(net))
	cost = _dijkstra_to!(ws, net, removed, org, dst)
	isinf(cost) && return result(:no_path)
	first_cost = cost
	push!(path_costs, cost)

	while true
		length(removed_list) >= nMaxRem && return result(:max_removals)

		# Walk the path back from dst; keep the heaviest edge (ties: nearest to dst)
		k = length(removed_list) + 1
		emax, wmax = 0, -Inf
		v = dst
		while v != org
			e = ws.parent_edge[v]
			record_paths && get!(path_first, e, k)
			if net.weights[e] > wmax
				emax, wmax = e, net.weights[e]
			end
			v = net.srcs[e]
		end

		removed[emax] = true
		push!(removed_list, emax)

		cost = _dijkstra_to!(ws, net, removed, org, dst)
		isinf(cost) && return result(:disconnected)
		push!(path_costs, cost)
		cost > maxCostRatio * first_cost && return result(:cost_ratio)
	end
end

"""
	crackOptimalPaths(g::SimpleDiGraph, org::Int64, dst::Int64,
					  weight_matrix::SparseMatrixCSC{Float64, Int64};
					  nMaxRem::Int64 = 0, maxCostRatio::Float64 = Inf)

Iteratively finds the shortest path from `org` to `dst` and removes its
highest-weight edge, on a copy of `g`.

# Arguments
- `g::SimpleDiGraph`: The input directed graph (not modified).
- `org::Int64`: The origin node.
- `dst::Int64`: The destination node.
- `weight_matrix::SparseMatrixCSC{Float64, Int64}`: Edge weights.
- `nMaxRem::Int64` (optional): Maximum number of edges to remove. `0` means `ne(g)`.
- `maxCostRatio::Float64` (optional): Stop when the cost of the current shortest
  path exceeds `maxCostRatio` times the cost of the first shortest path. The path
  that triggers the stop is not cracked. Defaults to `Inf` (no limit).

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

For many OD pairs on the same network, build a `CrackNetwork` once and use
`crackStatistics`, which avoids rebuilding it for every pair.
"""
function crackOptimalPaths(g::SimpleDiGraph, org::Int64, dst::Int64,
	weight_matrix::SparseMatrixCSC{Float64, Int64};
	nMaxRem::Int64 = 0,
	maxCostRatio::Float64 = Inf)

	net = CrackNetwork(g, weight_matrix)
	res = _crack(net, org, dst; nMaxRem = nMaxRem, maxCostRatio = maxCostRatio)

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
