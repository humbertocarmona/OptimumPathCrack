#=
City-like square lattice built from whole roads.

Each row and each column of the lattice is a road. A road is either two-way (one edge
per direction per block, with independent weights) or one-way (all its blocks point
the same way). Because every node lies on exactly one row and one column, one-way
roads give each node one edge in and one edge out, so flow balance (in = out) holds
automatically. With periodic boundaries every one-way road closes into a loop, so the
lattice is strongly connected for any choice of directions (before block removal).

Block weight = road mean × block factor:
- road mean: `weight_mean`, varied between roads (`road_cv`) and divided by
  `arterial_speedup` on arterials;
- block factor on local roads: `DisorderDist(β)` normalized to mean 1, arranged in
  space by a correlated surface with Hurst exponent `hurst` (rank mapping, as in
  `buildSquareNetwork`); on arterials: a small lognormal variation
  (`arterial_segment_cv`), so arterials stay smooth.
=#

# Uniform random spanning tree of a graph given as adjacency lists of (neighbour, edge id)
# (Wilson's algorithm); returns a mask over edge ids.
function _wilson_tree(rng::AbstractRNG, adj::Vector{Vector{Tuple{Int, Int}}}, n_edges::Int)
	n = length(adj)
	in_tree = falses(n)
	next_edge = zeros(Int, n)       # edge taken on the last exit from each node
	next_node = zeros(Int, n)
	tree = falses(n_edges)
	in_tree[rand(rng, 1:n)] = true
	for start in randperm(rng, n)
		u = start
		while !in_tree[u]           # loop-erased random walk (erasure by overwriting)
			v, e = adj[u][rand(rng, 1:length(adj[u]))]
			next_node[u], next_edge[u] = v, e
			u = v
		end
		u = start
		while !in_tree[u]
			in_tree[u] = true
			tree[next_edge[u]] = true
			u = next_node[u]
		end
	end
	return tree
end

# Lognormal factor with mean 1 and coefficient of variation `cv` (1.0 when cv == 0).
function _lognormal_factor(rng::AbstractRNG, cv::Float64)
	cv == 0 && return 1.0
	σ2 = log1p(cv^2)
	return exp(-σ2 / 2 + sqrt(σ2) * randn(rng))
end

"""
	buildCityLattice(nx, ny; kwargs...) -> (g, coords, distance_matrix, weight_matrix, roads)

Square lattice of `nx × ny` intersections whose rows and columns are roads, either
one-way or two-way, with a hierarchy of local streets and faster arterials.

# Keyword arguments
Road structure:
- `oneway_fraction = 0.5`: fraction of the non-arterial roads that are one-way
  (rounded to a whole number of roads per orientation).
- `alternation = 0.5`: for one-way roads taken in spatial order (rows bottom to top,
  columns left to right), probability that a road runs opposite to the previous
  one-way road of the same orientation. `1` alternates perfectly (Manhattan-like),
  `0.5` is independent, `0` makes all one-way rows (columns) point the same way.
- `arterial_fraction = 0.1`: fraction of roads of each orientation that are arterials
  (ignored when `arterial_spacing > 0`).
- `arterial_spacing = 0`: `0` places whole-line arterials at random (`arterial_fraction`).
  `s > 0` puts the arterials on every `s`-th row and column (indices `1, 1+s, …`); with
  `pbc` the lattice sides must be multiples of `s` with at least 3 coarse lines each.
  Only the chosen coarse links are fast (see `loop_fraction`); rows and columns of the
  coarse grid are flagged `arterial` in `roads` and follow `arterials_twoway`.
- `loop_fraction = 1.0`: with `arterial_spacing > 0`, the fast links are a uniform random
  spanning tree of the coarse grid plus this fraction of the other coarse links
  (`0` = tree, `1` = full grid). The fast length is then about `(1 + loop_fraction)/s`
  of all blocks.
- `arterials_twoway = true`: arterials are always two-way. If `false`, they follow
  `oneway_fraction` like other roads.
- `mean_degree = 4.0`: target mean number of street neighbours per intersection
  (undirected degree), between about 2.5 and 4. Values below 4 are reached by removing
  randomly chosen local blocks (both directions); arterials are never cut. On an open
  lattice (`pbc = false`) the maximum is below 4 because of the boundary.

Weights:
- `weight_mean = 1.0`: mean travel time of a local block.
- `road_cv = 0.3`: coefficient of variation of the mean weight between roads.
- `arterial_speedup = 2.0`: arterial blocks take `1/arterial_speedup` of the time of a
  local block on average.
- `beta = 1.0`: disorder strength of local blocks: the block factor is
  `DisorderDist(beta)` divided by its mean, so `log` of the factor is uniform over a
  range `beta` wide (`beta = 1` is close to a lognormal with CV 0.3; Boston's speed
  disorder corresponds to about 1.7).
- `hurst = -1.0`: Hurst exponent of the surface that arranges local block factors in
  space (`fbmSurface`, sampled at block midpoints). `-1` is uncorrelated; larger values
  give longer-range correlations (fractional Brownian surface for `0 < hurst < 1`).
  The distribution of factors is the same for every `hurst`.
- `direction_rho = 0.0`: correlation between the surfaces used for the two directions
  of a block (`0` independent, `1` the same surface).
- `arterial_segment_cv = 0.02`: variation of block weights along an arterial.

Geometry:
- `pbc = true`: periodic boundaries in both directions (a torus). OD distances from
  `create_ODs` are planar, which matches the torus only for `L < block·min(nx, ny)/2`.
- `block = 100.0`: block length in meters.
- `origin = (-3.7327, -38.5270)`: (lat, lon) of node 1.
- `seed = 1`: structure (arterials, one-way roads, directions, removed blocks) and
  weights use separate random streams, so changing only weight parameters (`beta`,
  `hurst`, `road_cv`, ...) keeps the same street layout.

# Returns
- `g::SimpleDiGraph`, `coords::Vector{Tuple{Float64,Float64}}` (lat, lon),
  `distance_matrix` (block lengths, also across the periodic boundary),
  `weight_matrix`.
- `roads::DataFrame`: one row per road with `orientation` (`:row`/`:column`), `index`,
  `arterial`, `oneway`, `direction` (`+1` towards increasing i or j, `-1` opposite,
  `0` two-way) and `mean_weight` (of the road's fast blocks for an arterial road; its
  local blocks take `mean_weight × arterial_speedup`). The generator parameters are
  attached as metadata, plus `"arterial_nodes"` (sorted nodes touching a fast block) and
  `"arterial_block_share"` (fast share of the kept blocks).
Node `(i, j)` has index `i + nx*(j-1)`.
"""
function buildCityLattice(nx::Int, ny::Int;
	oneway_fraction::Float64 = 0.5,
	alternation::Float64 = 0.5,
	arterial_fraction::Float64 = 0.1,
	arterial_spacing::Int = 0,
	loop_fraction::Float64 = 1.0,
	arterials_twoway::Bool = true,
	arterial_speedup::Float64 = 2.0,
	weight_mean::Float64 = 1.0,
	road_cv::Float64 = 0.3,
	beta::Float64 = 1.0,
	hurst::Float64 = -1.0,
	direction_rho::Float64 = 0.0,
	arterial_segment_cv::Float64 = 0.02,
	mean_degree::Float64 = 4.0,
	pbc::Bool = true,
	block::Float64 = 100.0,
	origin::Tuple{Float64, Float64} = (-3.7327, -38.5270),
	seed::Int = 1)

	@assert 0 <= oneway_fraction <= 1 && 0 <= alternation <= 1 && 0 <= arterial_fraction <= 1
	@assert arterial_speedup > 0 && weight_mean > 0
	0 <= direction_rho <= 1 || throw(ArgumentError("direction_rho must be in [0, 1]"))
	0 <= loop_fraction <= 1 || throw(ArgumentError("loop_fraction must be in [0, 1]"))
	s_art = arterial_spacing
	s_art >= 0 || throw(ArgumentError("arterial_spacing must be ≥ 0"))
	if s_art > 0
		ok = pbc ? (nx % s_art == 0 && ny % s_art == 0 && nx ÷ s_art >= 3 && ny ÷ s_art >= 3) :
			 (cld(nx, s_art) >= 2 && cld(ny, s_art) >= 2)
		ok || throw(ArgumentError("arterial_spacing = $s_art does not fit a $nx × $ny " *
								  "lattice (pbc: sides multiple of s, ≥ 3 coarse lines)"))
	end
	disorder = DisorderDist(beta)            # validates 0 < beta ≤ 700
	rng = Random.Xoshiro(UInt64[seed % UInt64, 1])    # structure
	wrng = Random.Xoshiro(UInt64[seed % UInt64, 2])   # weights
	N = nx * ny

	# --- roads ---------------------------------------------------------------------
	roads = DataFrame(orientation = Symbol[], index = Int[], arterial = Bool[],
		oneway = Bool[], direction = Int[], mean_weight = Float64[])
	for (orientation, n_roads) in ((:row, ny), (:column, nx))
		arterial = falses(n_roads)
		if s_art > 0
			arterial[1:s_art:n_roads] .= true
		else
			arterial[randperm(rng, n_roads)[1:round(Int, arterial_fraction * n_roads)]] .= true
		end

		eligible = arterials_twoway ? findall(!, arterial) : collect(1:n_roads)
		oneway = falses(n_roads)
		n_oneway = round(Int, oneway_fraction * length(eligible))
		oneway[eligible[randperm(rng, length(eligible))[1:n_oneway]]] .= true

		direction = zeros(Int, n_roads)
		previous = 0
		for r in 1:n_roads
			oneway[r] || continue
			direction[r] = previous == 0 ? rand(rng, (-1, 1)) :
						   (rand(rng) < alternation ? -previous : previous)
			previous = direction[r]
		end

		for r in 1:n_roads
			m = weight_mean * _lognormal_factor(wrng, road_cv)
			arterial[r] && (m /= arterial_speedup)
			push!(roads, (orientation, r, arterial[r], oneway[r], direction[r], m))
		end
	end

	# --- blocks (undirected segments) ------------------------------------------------
	# (a, b, road row in `roads`, midpoint x, midpoint y), with a -> b the +1 direction.
	# Node (i, j) sits at (2i-1, 2j-1) of a midpoint grid twice as fine as the lattice.
	ij2node(i, j) = i + nx * (j - 1)
	segments = NTuple{5, Int}[]
	for j in 1:ny, i in 1:nx                      # rows: road j
		(i < nx || (pbc && nx > 2)) || continue
		push!(segments, (ij2node(i, j), ij2node(mod1(i + 1, nx), j), j, 2i, 2j - 1))
	end
	for i in 1:nx, j in 1:ny                      # columns: road ny + i
		(j < ny || (pbc && ny > 2)) || continue
		push!(segments, (ij2node(i, j), ij2node(i, mod1(j + 1, ny)), ny + i, 2i - 1, 2j))
	end

	# fast blocks: whole arterial roads, or the chosen links of the coarse grid
	fast = [roads.arterial[sg[3]] for sg in segments]
	if s_art > 0
		fast .= _coarse_links(nx, ny, s_art, pbc, loop_fraction,
			Random.Xoshiro(UInt64[seed % UInt64, 3]), segments)
	end

	# Remove local blocks to reach the target mean degree (2 * segments / N)
	n_keep = round(Int, mean_degree * N / 2)
	local_segments = findall(!, fast)
	n_remove = length(segments) - n_keep
	if n_remove < 0
		throw(ArgumentError("mean_degree = $mean_degree is above the lattice maximum " *
							"$(round(2length(segments) / N, digits = 3))"))
	elseif n_remove > length(local_segments)
		throw(ArgumentError("mean_degree = $mean_degree needs removing more blocks than " *
							"there are local blocks; lower arterial_fraction"))
	end
	keep = trues(length(segments))
	keep[local_segments[randperm(rng, length(local_segments))[1:n_remove]]] .= false

	# --- directed edges --------------------------------------------------------------
	# (src, dst, road, midpoint x, midpoint y, forward?)
	dedges = Tuple{Int, Int, Int, Int, Int, Bool}[]
	edge_fast = Bool[]
	for (k, (a, b, road, mx, my)) in enumerate(segments)
		keep[k] || continue
		dir = roads.direction[road]
		dir >= 0 && (push!(dedges, (a, b, road, mx, my, true)); push!(edge_fast, fast[k]))
		dir <= 0 && (push!(dedges, (b, a, road, mx, my, false)); push!(edge_fast, fast[k]))
	end

	# --- weights ---------------------------------------------------------------------
	E = length(dedges)
	factor = Vector{Float64}(undef, E)
	is_local = .!edge_fast
	for k in findall(!, is_local)
		factor[k] = _lognormal_factor(wrng, arterial_segment_cv)
	end
	local_idx = findall(is_local)
	if !isempty(local_idx)
		h_fwd = fbmSurface(2nx, 2ny, hurst; rng = wrng)
		h_bwd = direction_rho == 1 ? h_fwd :
				direction_rho .* h_fwd .+
				sqrt(1 - direction_rho^2) .* fbmSurface(2nx, 2ny, hurst; rng = wrng)
		values = [(dedges[k][6] ? h_fwd : h_bwd)[dedges[k][4], dedges[k][5]] for k in local_idx]
		EL = length(local_idx)
		μ = mean(disorder)
		for (rank, i) in enumerate(sortperm(values))
			factor[local_idx[i]] = quantile(disorder, (rank - 0.5) / EL) / μ
		end
	end

	# matrices built in one call (inserting entries one by one is quadratic in E)
	g = SimpleDiGraph(N)
	Is = [e[1] for e in dedges]
	Js = [e[2] for e in dedges]
	for (s, d) in zip(Is, Js)
		add_edge!(g, s, d)
	end
	distance_matrix = sparse(Is, Js, fill(block, E), N, N)
	# an arterial road's mean_weight is that of its fast blocks; its local blocks are slower
	block_mean(k) = (r = dedges[k][3];
		roads.mean_weight[r] * (roads.arterial[r] && !edge_fast[k] ? arterial_speedup : 1.0))
	weight_matrix = sparse(Is, Js, [block_mean(k) * factor[k] for k in 1:E], N, N)

	# --- coordinates -----------------------------------------------------------------
	lat0, lon0 = origin
	dlat = block / 111_320.0
	dlon = block / (111_320.0 * cosd(lat0))
	coords = [(lat0 + (j - 1) * dlat, lon0 + (i - 1) * dlon) for j in 1:ny for i in 1:nx]

	params = Dict{String, Any}("nx" => nx, "ny" => ny, "oneway_fraction" => oneway_fraction,
		"alternation" => alternation, "arterial_fraction" => arterial_fraction,
		"arterial_spacing" => arterial_spacing, "loop_fraction" => loop_fraction,
		"arterials_twoway" => arterials_twoway, "arterial_speedup" => arterial_speedup,
		"weight_mean" => weight_mean, "road_cv" => road_cv, "beta" => beta,
		"hurst" => hurst, "direction_rho" => direction_rho,
		"arterial_segment_cv" => arterial_segment_cv, "mean_degree" => mean_degree,
		"pbc" => pbc, "block" => block, "seed" => seed)
	for (k, v) in params
		metadata!(roads, k, v; style = :note)
	end
	art_nodes = Set{Int}()
	for (a, b, _, _, _) in segments[fast .& keep]
		push!(art_nodes, a, b)
	end
	metadata!(roads, "arterial_nodes", sort!(collect(art_nodes)); style = :note)
	metadata!(roads, "arterial_block_share", count(fast .& keep) / max(count(keep), 1);
		style = :note)

	return g, coords, distance_matrix, weight_matrix, roads
end

# Fast mask over `segments` for the coarse-grid layout: coarse nodes are the crossings of
# rows and columns 1, 1+s, …; a coarse link is the stretch of s blocks between two
# neighbouring coarse nodes on a coarse line (a shorter trailing stretch on an open
# lattice is not a link and stays local). Fast links: a uniform random spanning tree of
# the coarse grid plus round(loop_fraction × other links) of the others, chosen in a fixed
# random order so that layouts are nested in loop_fraction.
function _coarse_links(nx, ny, s, pbc, loop_fraction, rng, segments)
	cx, cy = pbc ? (nx ÷ s, ny ÷ s) : (cld(nx, s), cld(ny, s))
	cnode(a, b) = a + cx * (b - 1)
	links = Tuple{Int, Int}[]             # coarse links (coarse node u, coarse node v)
	link_of = Dict{Tuple{Int, Int}, Int}()  # (road, first block index along it) -> link
	for b in 1:cy, a in 1:cx               # along coarse row b (road index 1 + (b-1)s)
		(a < cx || pbc) || continue
		(pbc || a * s + 1 <= nx) || continue
		push!(links, (cnode(a, b), cnode(mod1(a + 1, cx), b)))
		link_of[(1 + (b - 1) * s, 1 + (a - 1) * s)] = length(links)
	end
	for a in 1:cx, b in 1:cy               # along coarse column a (road ny + 1 + (a-1)s)
		(b < cy || pbc) || continue
		(pbc || b * s + 1 <= ny) || continue
		push!(links, (cnode(a, b), cnode(a, mod1(b + 1, cy))))
		link_of[(ny + 1 + (a - 1) * s, 1 + (b - 1) * s)] = length(links)
	end
	adj = [Tuple{Int, Int}[] for _ in 1:(cx * cy)]
	for (e, (u, v)) in enumerate(links)
		push!(adj[u], (v, e)); push!(adj[v], (u, e))
	end
	chosen = _wilson_tree(rng, adj, length(links))
	others = findall(!, chosen)
	order = others[randperm(rng, length(others))]
	chosen[order[1:round(Int, loop_fraction * length(others))]] .= true

	fast = falses(length(segments))
	for (k, (_, _, road, mx, my)) in enumerate(segments)
		is_row = road <= ny
		line = is_row ? road : road - ny
		(line - 1) % s == 0 || continue
		pos = is_row ? mx ÷ 2 : my ÷ 2          # block index along the road (1-based)
		start = 1 + ((pos - 1) ÷ s) * s
		l = get(link_of, (road, start), 0)
		l > 0 && chosen[l] && (fast[k] = true)
	end
	return fast
end
