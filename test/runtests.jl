using OptimumPathCrack
using Test, CSV, DataFrames, Graphs, SparseArrays
using Random, Distributions
using GeoDataFrames   # loads the writeGPKGFile extension

function test_buildNetworkFromTables()
	# Create temporary CSV files for testing
	edges_file = "data/edges.csv"
	nodes_file = "data/nodes.csv"

	# Run the function with test data
	g, coords, distance_matrix, weight_matrix, edges_index_dict =
		buildNetworkFromTables(edges_file, nodes_file)
	display(weight_matrix)

	# Perform assertions
	@test nv(g) == 28        # Ensure nodes are added
	@test ne(g) == 20        # Ensure edges are added
	@test length(coords) == nv(g)  # Ensure coordinates match node count
	@test size(distance_matrix) == (nv(g), nv(g))  # Ensure distance matrix size is correct
	@test size(weight_matrix) == (nv(g), nv(g))    # Ensure weight matrix size is correct
	@test length(edges_index_dict) == ne(g)  # Ensure edges index dictionary is correct

	println("All tests passed!")
end

function test_buildSquareNetwork()
	nx, ny = 5, 5  # Define grid size
	dist = Normal(0.5, 0.1)  # Define weight distribution
	seed = 42
	od, pbc = true, false

	# Run the function
	g, coords, weightmx, placeholder =
		buildSquareNetwork(nx, ny; dist = dist, seed = seed, od = od, pbc = pbc)

	# Perform assertions
	@test nv(g) == nx * ny + 2 * od  # Check total nodes including origin/destination
	@test ne(g) > 0  # Ensure edges are added
	@test length(coords) == nv(g)  # Ensure coordinates match node count
	@test size(weightmx) == (nv(g), nv(g))  # Check weight matrix size
	@test placeholder == 0  # Ensure placeholder return value is correct

	println("All tests passed for buildSquareNetwork!")
end

function test_cellList()
	# Define test coordinates
	coords = [(-3.7000, -38.6000),
		(-3.7001, -38.6000),
		(-3.7020, -38.6000),
		(-3.7021, -38.6000),
		(-3.7040, -38.6000),
		(-3.7041, -38.6000),
		(-3.7060, -38.6000),
		(-3.7061, -38.6000),
		(-3.7000, -38.6010),
		(-3.7000, -38.6030),
		(-3.7000, -38.6040)]
	cellWidth = 100.0  # Cell width in meters

	# Run the function
	result = cellList_euclidean(coords; cellWidth = cellWidth)

	display(result["cells"])
	display(result["next"])
	display(result["pos"])

	# Extract values from the result dictionary
	dx, dy = result["cellSize"]
	nx, ny = result["nCells"]
	cells, next = result["cells"], result["next"]

	# Perform assertions
	@test length(result["coords"]) == length(coords)  # Ensure coords are stored properly
	@test nx > 0 && ny > 0  # Grid dimensions must be positive
	@test size(cells) == (nx, ny)  # Ensure cell matrix matches grid size
	@test size(next) == (length(coords),)  # Ensure next array has the correct size

	println("All tests passed for cellList!")
end

function test_OD_matrix()
	edges_file = "data/boston-edges-4h.csv"
	nodes_file = "data/boston-nodes.csv"
	g, coords, distance_matrix, weight_matrix, edges_index_dict =
		buildNetworkFromTables(edges_file, nodes_file)
	cell_list = cellList_lat_lon(coords; cellWidth = 100.0)
	println(typeof(cell_list))
	n_od_pairs = 1000
	OD = create_ODs(1000.0, cell_list, n_od_pairs = n_od_pairs)

	@test size(OD, 1) == n_od_pairs  # Ensure coords are stored properly
end

function test_crackOptimalPaths()
	edges_file = "data/boston-edges-4h.csv"
	nodes_file = "data/boston-nodes.csv"
	g, coords, distance_matrix, weight_matrix, edges_index_dict =
		buildNetworkFromTables(edges_file, nodes_file)
	cell_list = cellList_lat_lon(coords; cellWidth = 100.0)
	n_od_pairs = 10
	ℓ = 5000.0
	OD = create_ODs(ℓ, cell_list, n_od_pairs = n_od_pairs)
	orig, dest = OD[4]
	gr, removed_edges, path_edges = crackOptimalPaths(g, orig, dest, weight_matrix)

	n_removed = length(findnz(removed_edges)[1])
	@test  maximum(removed_edges) == n_removed
	@test  maximum(path_edges) == n_removed
	writeGPKGFile(g, coords, distance_matrix, weight_matrix, 
	removed_matrix=removed_edges, path_matrix=path_edges, od=(orig, dest))
end

function load_boston()
	buildNetworkFromTables("data/boston-edges-4h.csv", "data/boston-nodes.csv")
end

function test_cost_ratio_and_reachability()
	g, coords, distance_matrix, weight_matrix, _ = load_boston()
	cell_list = cellList_lat_lon(coords; cellWidth = 100.0)
	ℓ = 3000.0
	OD = create_ODs(ℓ, cell_list, n_od_pairs = 20, graph = g)
	@test length(OD) == 20
	@test allunique(OD)
	comp = zeros(Int, nv(g))
	for (c, m) in enumerate(strongly_connected_components(g))
		comp[m] .= c
	end
	@test all(comp[o] == comp[d] for (o, d) in OD)

	# pick a pair that needs several removals
	k = findfirst(od -> maximum(crackOptimalPaths(g, od..., weight_matrix)[2]) >= 3, OD)
	@test k !== nothing
	orig, dest = OD[k]
	_, rem_full, _, res_full = crackOptimalPaths(g, orig, dest, weight_matrix)
	@test res_full.stop_reason == :disconnected
	@test issorted(res_full.path_costs)            # costs never decrease
	@test length(res_full.path_costs) == maximum(rem_full)

	ratio = 1.05
	_, rem_r, _, res_r = crackOptimalPaths(g, orig, dest, weight_matrix; maxCostRatio = ratio)
	n_r = Int(maximum(rem_r; init = 0.0))
	@test n_r <= maximum(rem_full)
	if res_r.stop_reason == :cost_ratio
		@test res_r.path_costs[end] > ratio * res_r.path_costs[1]
		@test all(res_r.path_costs[1:(end-1)] .<= ratio * res_r.path_costs[1])
	end
	# same edges removed, in the same order, up to the stop
	I, J, K = findnz(rem_r)
	@test all(rem_full[I[p], J[p]] == K[p] for p in eachindex(K))

	_, rem_m, _, res_m = crackOptimalPaths(g, orig, dest, weight_matrix; nMaxRem = 2)
	@test res_m.stop_reason == :max_removals && maximum(rem_m) == 2
end

"""
Replays each crack with Graphs.jl's Dijkstra: before the k-th removal, the shortest
path cost must equal `path_costs[k]`, and the k-th removed edge must lie on a
shortest path.
"""
function test_early_exit_dijkstra()
	g, coords, _, weight_matrix, _ = load_boston()
	cell_list = cellList_lat_lon(coords; cellWidth = 100.0)
	for ℓ in (500.0, 3000.0)
		for (o, d) in create_ODs(ℓ, cell_list, n_od_pairs = 5, graph = g, seed = 7)
			_, rem, _, res = crackOptimalPaths(g, o, d, weight_matrix; maxCostRatio = 2.0)
			I, J, K = findnz(rem)
			order = sortperm(K)
			gr = copy(g)
			for (k, p) in enumerate(order)
				i, j = I[p], J[p]
				from_o = dijkstra_shortest_paths(gr, o, weight_matrix).dists
				to_d = dijkstra_shortest_paths(reverse(gr), d, transpose(weight_matrix)).dists
				@test from_o[d] ≈ res.path_costs[k]
				@test from_o[i] + weight_matrix[i, j] + to_d[j] ≈ res.path_costs[k]
				rem_edge!(gr, i, j)
			end
			final = dijkstra_shortest_paths(gr, o, weight_matrix).dists[d]
			if res.stop_reason == :disconnected
				@test isinf(final)
			else
				@test final ≈ res.path_costs[end] && final > 2.0 * res.path_costs[1]
			end
		end
	end
end

function test_crackStatistics()
	g, coords, distance_matrix, weight_matrix, _ = load_boston()
	edges_df, od_df = crackStatistics(g, coords, distance_matrix, weight_matrix,
		[1000.0, 3000.0]; n_od_pairs = 5, maxCostRatio = 1.5)
	@test nrow(od_df) == 10
	@test sort(unique(od_df.ℓ)) == [1000.0, 3000.0]
	@test all(abs.(od_df.od_distance .- od_df.ℓ) .< 0.01 .* od_df.ℓ)
	@test all(od_df.n_removed .> 0)
	@test sum(od_df.n_removed) == nrow(edges_df)
	@test all(r -> weight_matrix[r.src, r.dst] == r.weight &&
				   distance_matrix[r.src, r.dst] == r.length, eachrow(edges_df))
	@test all(s -> s in (:disconnected, :cost_ratio), od_df.stop_reason)
	@test od_df.origin_outdegree == outdegree.(Ref(g), od_df.origin)
	@test od_df.destination_indegree == indegree.(Ref(g), od_df.destination)
	@test all(od_df.cost_ratio[od_df.stop_reason .== :cost_ratio] .> 1.5)
	show(stdout, od_df; allcols = true); println()
end

function small_graph(edge_list)
	n = maximum(max(s, d) for (s, d, _) in edge_list)
	g = SimpleDiGraph(n)
	W = spzeros(n, n)
	for (s, d, w) in edge_list
		add_edge!(g, s, d)
		W[s, d] = w
	end
	return g, W
end

function test_tie_breaking()
	# Three equal-cost (3.0) paths 1 -> 5: via 2-4, via 3-4, and via 6.
	# A uniform draw picks each with probability 1/3 (a random parent at each step
	# would pick the path via 6 half of the time).
	g, W = small_graph([(1, 2, 1.0), (1, 3, 1.0), (2, 4, 1.0), (3, 4, 1.0), (4, 5, 1.0),
		(1, 6, 1.5), (6, 5, 1.5)])
	n = 6000
	via = zeros(Int, 3)
	rem_on_6 = zeros(Int, 2)
	for s in 1:n
		_, rem, path, _ = crackOptimalPaths(g, 1, 5, W; nMaxRem = 1, rng = Xoshiro(s))
		path[1, 2] == 1 && (via[1] += 1)
		path[1, 3] == 1 && (via[2] += 1)
		if path[1, 6] == 1
			via[3] += 1
			rem[1, 6] == 1 && (rem_on_6[1] += 1)
			rem[6, 5] == 1 && (rem_on_6[2] += 1)
		end
	end
	@test sum(via) == n
	@test all(abs.(via ./ n .- 1 / 3) .< 0.03)
	# the two edges of the via-6 path tie at weight 1.5: each removed half the time
	@test abs(rem_on_6[1] / via[3] - 0.5) < 0.05

	# Equal-weight edges on a single path: each removed with probability 1/3
	g, W = small_graph([(1, 2, 2.0), (2, 3, 2.0), (3, 4, 2.0)])
	counts = zeros(Int, 3)
	for s in 1:n
		_, rem, _, _ = crackOptimalPaths(g, 1, 4, W; nMaxRem = 1, rng = Xoshiro(s))
		counts[findfirst(k -> rem[k, k+1] == 1, 1:3)] += 1
	end
	@test all(abs.(counts ./ n .- 1 / 3) .< 0.03)

	# Same rng seed -> same result
	g, coords, _, weight_matrix, _ = load_boston()
	a = crackOptimalPaths(g, 817, 10663, weight_matrix; rng = Xoshiro(5))
	b = crackOptimalPaths(g, 817, 10663, weight_matrix; rng = Xoshiro(5))
	@test a[2] == b[2] && a[4] == b[4]
end

function test_statistics_io()
	g, coords, distance_matrix, weight_matrix, _ = load_boston()
	edges_df, od_df = crackStatistics(g, coords, distance_matrix, weight_matrix,
		[1000.0, 2000.0]; n_od_pairs = 4, seed = 3)
	@test metadata(od_df, "seed") == 3
	@test metadata(edges_df, "distances") == [1000.0, 2000.0]
	@test isinf(metadata(od_df, "maxCostRatio"))
	mktempdir() do dir
		prefix = joinpath(dir, "run", "boston")
		files = writeCrackStatistics(prefix, edges_df, od_df;
			extra = Dict("edges_file" => "boston-edges-4h.csv"))
		@test all(isfile, files)
		e2, o2, params = readCrackStatistics(prefix)
		@test e2 == edges_df
		@test isequal(o2, od_df)
		@test params["seed"] == 3 && params["extra"]["edges_file"] == "boston-edges-4h.csv"
		@test metadata(o2, "n_edges") == ne(g)
	end
end

@testset "tie breaking" begin
	test_tie_breaking()
end

@testset "statistics IO" begin
	test_statistics_io()
end

@testset "cracking Tests" begin
	test_crackOptimalPaths()
end

@testset "cost ratio and reachability" begin
	test_cost_ratio_and_reachability()
end

@testset "early-exit Dijkstra vs Graphs.jl" begin
	test_early_exit_dijkstra()
end

@testset "crackStatistics" begin
	test_crackStatistics()
end

@testset "OD matrix Tests" begin
    test_OD_matrix()
end

@testset "cellList Tests" begin
    test_cellList()
end

@testset "buildSquareNetwork Tests" begin
    test_buildSquareNetwork()
end

@testset "buildNetworkFromTables Tests" begin
    test_buildNetworkFromTables()
end
