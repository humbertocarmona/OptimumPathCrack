"""
	create_ODs(ℓ, cell_list; seed = 1, n_od_pairs = 10, δ = 0.01, graph = nothing)

Generates origin-destination (OD) pairs whose Euclidean (geodesic) distance is
within `δ⋅ℓ` of `ℓ`.

# Arguments
- `ℓ::Float64`: Target OD distance in meters.
- `cell_list::AbstractDict`: Output of `cellList_lat_lon` or `cellList_euclidean`.

# Keyword Arguments
- `seed::Int64 = 1`: Seed for a local RNG (the global RNG is not touched).
- `n_od_pairs::Int64 = 10`: Number of OD pairs to generate.
- `δ::Float64 = 0.01`: Allowed relative deviation from `ℓ`.
- `graph::Union{Nothing, AbstractGraph} = nothing`: If given, only pairs in the same
  strongly connected component are accepted, so `D` is reachable from `O` (and back).

# Returns
- `Vector{Tuple{Int64, Int64}}`: OD pairs as node indices. May be shorter than
  `n_od_pairs` if not enough pairs were found.
"""
function create_ODs(
	ℓ::Float64,
	cell_list::AbstractDict;
	seed::Int64 = 1,
	n_od_pairs::Int64 = 10,
	δ::Float64 = 0.01,
	graph::Union{Nothing, AbstractGraph} = nothing)::Vector{Tuple{Int64, Int64}}

	rng = Random.Xoshiro(seed)

	δ = δ * ℓ # absolute tolerance in meters
	# Displacement in cell units. Cells are cellWidth meters wide in both
	# cell-list flavours (in degrees for lat_lon, but ≈ cellWidth meters).
	cellWidth = Float64(get(cell_list, "cellWidth", 100.0))
	ℓ_cells = ℓ / cellWidth

	pos = cell_list["pos"]
	coords = cell_list["coords"]
	cellSize = cell_list["cellSize"] # [dx, dy]
	n_cells = cell_list["nCells"]    # [nx, ny]
	cells = cell_list["cells"]
	next = cell_list["next"]

	n_nodes = size(pos, 1)

	# component label per node; all equal when no graph is given
	component = ones(Int, n_nodes)
	if graph !== nothing
		for (c, members) in enumerate(strongly_connected_components(graph))
			component[members] .= c
		end
	end

	OD_vector = Tuple{Int64, Int64}[]
	OD_set = Set{Tuple{Int64, Int64}}()

	max_cell_tries = 100
	n_orig = 0 # counter to prevent infinite loops
	n = 0      # successful OD pairs
	while n < n_od_pairs && n_orig < 100 * n_nodes
		n_orig += 1
		origin_idx = rand(rng, 1:n_nodes)

		lat, lon = coords[origin_idx]
		origin_coord = LLA(lat, lon, 0.0)

		# continuous cell coordinates of the origin (1-based)
		cell_origin = pos[origin_idx] ./ cellSize .+ 1

		# Search for a non-empty destination cell at distance ℓ
		foundCell = false
		dst_cell = [0, 0]
		n_cell_tries = 0
		while !foundCell && n_cell_tries < max_cell_tries
			ϕ = 2π * rand(rng)
			dst_cell = Int.(floor.(cell_origin .+ ℓ_cells .* [cos(ϕ), sin(ϕ)]))
			foundCell = all(0 < dst_cell[k] <= n_cells[k] for k in 1:2) &&
						(cells[dst_cell[1], dst_cell[2]] > 0)
			n_cell_tries += 1
		end
		foundCell || continue

		# Scan nodes in the destination cell
		dest_idx = cells[dst_cell[1], dst_cell[2]]
		while dest_idx > 0
			if component[dest_idx] == component[origin_idx] &&
			   (origin_idx, dest_idx) ∉ OD_set
				lat, lon = coords[dest_idx]
				dist = euclidean_distance(origin_coord, LLA(lat, lon, 0.0))
				if abs(dist - ℓ) < δ
					n += 1
					push!(OD_vector, (origin_idx, dest_idx))
					push!(OD_set, (origin_idx, dest_idx))
					n ≥ n_od_pairs && break
				end
			end
			dest_idx = next[dest_idx]
		end
	end

	if n < n_od_pairs
		@warn("create_ODs: found only $(n)/$(n_od_pairs) OD pairs for ℓ = $(ℓ)")
	end
	return OD_vector
end
