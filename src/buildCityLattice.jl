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
- `arterial_fraction = 0.1`: fraction of roads of each orientation that are arterials.
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
  `0` two-way) and `mean_weight`. The generator parameters are attached as metadata.
Node `(i, j)` has index `i + nx*(j-1)`.
"""
function buildCityLattice(nx::Int, ny::Int;
	oneway_fraction::Float64 = 0.5,
	alternation::Float64 = 0.5,
	arterial_fraction::Float64 = 0.1,
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
	disorder = DisorderDist(beta)            # validates 0 < beta ≤ 700
	rng = Random.Xoshiro(UInt64[seed % UInt64, 1])    # structure
	wrng = Random.Xoshiro(UInt64[seed % UInt64, 2])   # weights
	N = nx * ny

	# --- roads ---------------------------------------------------------------------
	roads = DataFrame(orientation = Symbol[], index = Int[], arterial = Bool[],
		oneway = Bool[], direction = Int[], mean_weight = Float64[])
	for (orientation, n_roads) in ((:row, ny), (:column, nx))
		arterial = falses(n_roads)
		arterial[randperm(rng, n_roads)[1:round(Int, arterial_fraction * n_roads)]] .= true

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

	# Remove local blocks to reach the target mean degree (2 * segments / N)
	n_keep = round(Int, mean_degree * N / 2)
	local_segments = findall(s -> !roads.arterial[s[3]], segments)
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
	for (k, (a, b, road, mx, my)) in enumerate(segments)
		keep[k] || continue
		dir = roads.direction[road]
		dir >= 0 && push!(dedges, (a, b, road, mx, my, true))
		dir <= 0 && push!(dedges, (b, a, road, mx, my, false))
	end

	# --- weights ---------------------------------------------------------------------
	E = length(dedges)
	factor = Vector{Float64}(undef, E)
	is_local = [!roads.arterial[road] for (_, _, road, _, _, _) in dedges]
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
	weight_matrix = sparse(Is, Js, [roads.mean_weight[dedges[k][3]] * factor[k] for k in 1:E], N, N)

	# --- coordinates -----------------------------------------------------------------
	lat0, lon0 = origin
	dlat = block / 111_320.0
	dlon = block / (111_320.0 * cosd(lat0))
	coords = [(lat0 + (j - 1) * dlat, lon0 + (i - 1) * dlon) for j in 1:ny for i in 1:nx]

	params = Dict{String, Any}("nx" => nx, "ny" => ny, "oneway_fraction" => oneway_fraction,
		"alternation" => alternation, "arterial_fraction" => arterial_fraction,
		"arterials_twoway" => arterials_twoway, "arterial_speedup" => arterial_speedup,
		"weight_mean" => weight_mean, "road_cv" => road_cv, "beta" => beta,
		"hurst" => hurst, "direction_rho" => direction_rho,
		"arterial_segment_cv" => arterial_segment_cv, "mean_degree" => mean_degree,
		"pbc" => pbc, "block" => block, "seed" => seed)
	for (k, v) in params
		metadata!(roads, k, v; style = :note)
	end

	return g, coords, distance_matrix, weight_matrix, roads
end
