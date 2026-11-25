function ij2n(i, j, nx)
	return i + nx * (j - 1)
end

function n2ij(n, nx)
	j = (n - 1) ÷ nx
	i = n - nx * j
	return i, j + 1
end

function buildSquareNetwork_velho(nx::Int, ny::Int; p::Float64 = 0.5,
	dist, seed::Int64 = 123,
	od::Bool = true,
	pbc::Bool = false)
	"""
		nx, ny: network size
		p: probability of bidirect edges
		dist: distribution to generate weights - DisorderDist is defined here,
			  but it can be anyone defined in Distributions

		add one abobe the top row and one below to the bottom row

		add edges
		31---32---33---34---35---36      6
		 |    |    |    |    |    |
		25---26---27---28---29---30      5
		 |    |    |    |    |    |
		19---20---21---22---23---24      4
		 |    |    |    |    |    |
		13---14---15---16---17---18      3
		 |    |    |    |    |    |
		 7----8----9---10---11---12      2
		 |    |    |    |    |    |
		 1----2----3----4----5----6      1
										||
	   I=1    2    3    4    5    6      J
	"""

	if isa(dist, Float64)
		dist = DisorderDist(dist)
	end

	Random.seed!(seed)
	nvertices = nx * ny
	nnodes = nvertices
	if od
		nnodes = nvertices + 2 # last 2 are origin-destination
	end
	g = SimpleDiGraph(nnodes)
	weightmx = spzeros(nnodes, nnodes)

	en = 0
	center = [-3.7327, -38.5270]
	dx = 9.043317e-4
	dy = 9.002123e-4
	coords = Tuple{Float64, Float64}[]
	# interior
	for i in 1:nx
		for j in 1:ny
			pos = center + [(i - 1) * dx, (j - 1) * dy]
			push!(coords, Tuple(pos))
			s = ij2n(i, j, nx)

			#add horizontal
			if i < nx
				d = ij2n(i + 1, j, nx)
				unidirected = p > rand()    # bidirected edge?
				if unidirected
					right = rand() > 0.5 # choose direction
					if right
						ϵ = rand(dist)
						add_edge!(g, s, d)
						weightmx[s, d] = ϵ
						en += 1
					else
						ϵ = rand(dist)
						add_edge!(g, d, s)
						weightmx[d, s] = ϵ
						en += 1
					end
				else # add bi-directed
					ϵ = rand(dist)
					add_edge!(g, s, d)
					weightmx[s, d] = ϵ

					ϵ = rand(dist)
					add_edge!(g, d, s)
					weightmx[d, s] = ϵ
					en += 1
				end
			elseif pbc
				d = ij2n(1, j, nx)
				unidirected = p > rand()    # bidirected edge?
				if unidirected
					right = rand() > 0.5 # choose direction
					if right
						ϵ = rand(dist)
						add_edge!(g, s, d)
						weightmx[s, d] = ϵ
						en += 1
					else
						ϵ = rand(dist)
						add_edge!(g, d, s)
						weightmx[d, s] = ϵ
						en += 1
					end
				else # add bi-directed
					ϵ = rand(dist)
					add_edge!(g, s, d)
					weightmx[s, d] = ϵ

					ϵ = rand(dist)
					add_edge!(g, d, s)
					weightmx[d, s] = ϵ
					en += 1
				end
			end

			#add vertical
			if j < ny
				d = ij2n(i, j + 1, nx)
				unidirected = p > rand()    # bidirected edge?
				if unidirected
					up = rand() > 0.5 # choose direction
					if up
						ϵ = rand(dist)
						add_edge!(g, s, d)
						weightmx[s, d] = ϵ
						en += 1
					else
						ϵ = rand(dist)
						add_edge!(g, d, s)
						weightmx[d, s] = ϵ
					end
				else # add bi-directed - each direction has a different weight
					ϵ = rand(dist)
					add_edge!(g, s, d)
					weightmx[s, d] = ϵ

					ϵ = rand(dist)
					add_edge!(g, d, s)
					weightmx[d, s] = ϵ
				end
			end
		end
	end

	if od # connects origin and destination
		org = nvertices + 1
		dst = nvertices + 2
		for j ∈ 1:ny  #left column
			d = ij2n(1, j, nx)
			add_edge!(g, org, d)
			weightmx[org, d] = 0.0
		end

		for j ∈ 1:ny #right column
			s = ij2n(nx, j, nx)
			add_edge!(g, s, dst)
			weightmx[s, dst] = 0.0
		end
		push!(coords, Tuple(center + [0.5 * (nx - 1.0) * dx, -1.0 * dy]))
		push!(coords, Tuple(center + [0.5 * (nx - 1.0) * dx, (ny + 1.0) * dy]))
	end

	return g, coords, weightmx, 0
end

"""
	buildSquareNetwork(nx, ny; p = 0.5, dist, seed = 123, od = true, pbc = false,
					   pbc_y = false, hurst = nothing, direction_rho = 0.0)
		-> (g, coords, weightmx, 0)

Square grid of `nx × ny` nodes with random edge directions and weights.

# Arguments
- `nx::Int`, `ny::Int`: Number of columns and rows.
- `p::Float64 = 0.5`: Probability that a link is one-way (random direction); otherwise
  it is two-way, with an independent weight per direction.
- `dist`: Weight distribution: any univariate distribution, or a number `β` for
  `DisorderDist(β)`. Weights are drawn conditioned on being strictly positive (values
  ≤ 0 are redrawn), so e.g. `Normal(0.5, 0.3)` gives a normal truncated at 0.
- `seed::Int64 = 123`: Seed for a local random generator (the global one is not touched).
- `od::Bool = true`: Adds an origin node (index `nx*ny + 1`) linked to every node of
  the first column and a destination node (`nx*ny + 2`) linked from every node of the
  last column. These links get a weight of `1e-6 ×` the smallest lattice weight: they
  are positive, as cracking requires, but never the heaviest edge of a path and they do
  not change which path is shortest.
- `pbc::Bool = false`: Periodic boundary in the horizontal direction.
- `pbc_y::Bool = false`: Periodic boundary in the vertical direction. With both `pbc`
  and `pbc_y` (and `od = false`) the lattice is a torus, for point-to-point cracking.
- `hurst = nothing`: If a number `H ≥ -1`, weights are spatially correlated. A surface
  from `fbmSurface` with Hurst exponent `H` is sampled at each link's midpoint (on a grid
  twice as fine as the lattice), and the values are mapped to weights by rank:
  `w = quantile(dist, (rank - 1/2)/E)` over the `E` lattice edges. The weights then have
  exactly the distribution `dist` (truncated at 0 if needed); only their arrangement
  changes. `H = -1` is uncorrelated; larger `H` means longer-range correlations
  (fractional Brownian surface for `0 < H < 1`). `nothing` draws independent weights.
- `direction_rho = 0.0`: With `hurst`, the correlation between the surfaces used for
  the two directions of a link (`+x`/`+y` vs `-x`/`-y`): `0` independent, `1` the same
  surface (both directions of a two-way link get adjacent ranks, so nearly equal weights).

Link directions and weights use separate random streams derived from `seed`, so the same
`seed` gives the same topology with or without `hurst`.

# Returns
`(g, coords, weightmx, 0)`: directed graph, (lat, lon) coordinates, weight matrix, and
a placeholder. Node `(i, j)` has index `i + nx*(j-1)`.
"""
function buildSquareNetwork(
	nx::Int,
	ny::Int;
	p::Float64 = 0.5,
	dist,
	seed::Int64 = 123,
	od::Bool = true,
	pbc::Bool = false,
	pbc_y::Bool = false,
	hurst::Union{Nothing, Real} = nothing,
	direction_rho::Real = 0.0,
)
	dist isa Real && (dist = DisorderDist(dist))
	0 <= direction_rho <= 1 || throw(ArgumentError("direction_rho must be in [0, 1]"))
	rng = Random.Xoshiro(UInt64[seed % UInt64, 1])    # topology
	wrng = Random.Xoshiro(UInt64[seed % UInt64, 2])   # weights

	nvertices = nx * ny
	nnodes = nvertices + 2 * od
	g = SimpleDiGraph(nnodes)

	center = [-3.7327, -38.5270]  # Base coordinate
	dx = 9.043317e-4  # X-axis spacing
	dy = 9.002123e-4  # Y-axis spacing
	coords = Tuple{Float64, Float64}[]

	# Directed lattice edges: (src, dst, midpoint x, midpoint y, forward?)
	# forward = pointing towards increasing i or j
	lattice_edges = Tuple{Int, Int, Int, Int, Bool}[]
	function add_link!(s, d, mx, my)
		if p > rand(rng)            # one-way, random direction
			if rand(rng) > 0.5
				push!(lattice_edges, (d, s, mx, my, false))
			else
				push!(lattice_edges, (s, d, mx, my, true))
			end
		else                        # two-way
			push!(lattice_edges, (s, d, mx, my, true))
			push!(lattice_edges, (d, s, mx, my, false))
		end
	end

	# node (i, j) sits at (2i-1, 2j-1) of the midpoint grid
	for i in 1:nx
		for j in 1:ny
			push!(coords, Tuple(center + [(i - 1) * dx, (j - 1) * dy]))
			s = ij2n(i, j, nx)
			(i < nx || pbc) && add_link!(s, ij2n(mod1(i + 1, nx), j, nx), 2i, 2j - 1)
			(j < ny || pbc_y) && add_link!(s, ij2n(i, mod1(j + 1, ny), nx), 2i - 1, 2j)
		end
	end

	E = length(lattice_edges)
	if hurst === nothing
		weights = [_positive_rand(wrng, dist) for _ in 1:E]
	else
		h_fwd = fbmSurface(2nx, 2ny, hurst; rng = wrng)
		h_bwd = direction_rho == 1 ? h_fwd :
				direction_rho * h_fwd .+
				sqrt(1 - direction_rho^2) .* fbmSurface(2nx, 2ny, hurst; rng = wrng)
		values = [(fwd ? h_fwd : h_bwd)[mx, my] for (_, _, mx, my, fwd) in lattice_edges]
		wdist = minimum(dist) > 0 ? dist : truncated(dist; lower = 0.0)
		weights = Vector{Float64}(undef, E)
		for (rank, e) in enumerate(sortperm(values))
			weights[e] = quantile(wdist, (rank - 0.5) / E)
		end
		all(>(0), weights) || throw(ArgumentError("quantiles of $dist gave non-positive weights"))
	end
	# weight matrix built in one call (inserting entries one by one is quadratic in E)
	Is = [e[1] for e in lattice_edges]
	Js = [e[2] for e in lattice_edges]
	for (s, d) in zip(Is, Js)
		add_edge!(g, s, d)
	end

	if od
		w_od = 1e-6 * minimum(weights)
		org, dst = nvertices + 1, nvertices + 2
		for j in 1:ny
			add_edge!(g, org, ij2n(1, j, nx))
			push!(Is, org); push!(Js, ij2n(1, j, nx)); push!(weights, w_od)
			add_edge!(g, ij2n(nx, j, nx), dst)
			push!(Is, ij2n(nx, j, nx)); push!(Js, dst); push!(weights, w_od)
		end
		push!(coords, Tuple(center + [0.5 * (nx - 1.0) * dx, -1.0 * dy]))
		push!(coords, Tuple(center + [0.5 * (nx - 1.0) * dx, (ny + 1.0) * dy]))
	end
	weightmx = sparse(Is, Js, weights, nnodes, nnodes)

	return g, coords, weightmx, 0
end
