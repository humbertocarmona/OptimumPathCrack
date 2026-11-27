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

	# All weights strictly positive, also for a distribution that can go negative
	# (Normal(0.5, 0.3) is negative ~5% of the time) and for the OD links
	g, _, W, _ = buildSquareNetwork(20, 20; dist = Normal(0.5, 0.3), seed = 1)
	w = [W[src(e), dst(e)] for e in edges(g)]
	@test all(>(0), w)
	org, dst_node = 20 * 20 + 1, 20 * 20 + 2
	w_od = W[org, 1]
	@test w_od > 0 && w_od < minimum(W[src(e), dst(e)] for e in edges(g) if src(e) <= 400 && dst(e) <= 400)
	# reproducible, and crackable from the origin to the destination super-node
	@test buildSquareNetwork(20, 20; dist = 5.0, seed = 7)[3] ==
		  buildSquareNetwork(20, 20; dist = 5.0, seed = 7)[3]
	g, _, W, _ = buildSquareNetwork(20, 20; dist = 5.0, seed = 7)
	_, rem, _, res = crackOptimalPaths(g, org, dst_node, W)
	@test res.stop_reason == :disconnected
	@test all(k -> rem[org, k] == 0 && rem[k, dst_node] == 0, 1:nv(g))  # OD links never cracked

	# A clear error for non-positive weights
	W0 = copy(W); W0[org, 1] = 0.0
	@test_throws ArgumentError crackOptimalPaths(g, org, dst_node, W0)

	println("All tests passed for buildSquareNetwork!")
end

function test_DisorderDist()
	β = 5.0
	d = DisorderDist(β)
	@test rand(Xoshiro(3), d) == rand(Xoshiro(3), d)
	x = rand(Xoshiro(1), d, 200_000)
	@test all(w -> exp(-β) <= w <= 1, x)
	@test isapprox(sum(x) / length(x), mean(d); rtol = 0.01)
	@test isapprox(sum(abs2, x .- mean(d)) / length(x), var(d); rtol = 0.03)
	@test isapprox(sum(log, x) / length(x), -β / 2; atol = 0.02)  # log w uniform on [-β, 0]
	@test minimum(d) == exp(-β) && maximum(d) == 1.0
	@test quantile(d, cdf(d, 0.1)) ≈ 0.1
	@test pdf(d, 0.1) ≈ 1 / (β * 0.1) && pdf(d, 1e-5) == 0
	@test length(rand(d, 3)) == 3                      # global-RNG forms still work
	@test rand(Xoshiro(1), DisorderDist(700.0)) > 0    # strongest allowed disorder
	@test_throws ArgumentError DisorderDist(0.0)
	@test_throws ArgumentError DisorderDist(800.0)
	# lattice built from a number β uses DisorderDist(β)
	g, _, W, _ = buildSquareNetwork(10, 10; dist = 3.0, od = false)
	@test all(e -> exp(-3.0) <= W[src(e), dst(e)] <= 1, edges(g))
end

function test_correlated_weights()
	# surface: normalized, H = -1 uncorrelated, Hurst exponent recovered for H > 0
	h = fbmSurface(128, 128, -1.0; rng = Xoshiro(1))
	@test abs(sum(h) / length(h)) < 1e-10 && sum(abs2, h) / length(h) ≈ 1
	nn(h) = sum(h .* circshift(h, (1, 0))) / length(h)
	@test abs(nn(h)) < 0.05
	function Hest(h)
		x = log.(1:8)
		y = log.([sum(abs2, circshift(h, (-r, 0)) .- h) for r in 1:8])
		xm, ym = sum(x) / 8, sum(y) / 8
		return sum((x .- xm) .* (y .- ym)) / sum(abs2, x .- xm) / 2
	end
	for H in (0.5, 0.8)
		est = sum(Hest(fbmSurface(256, 256, H; rng = Xoshiro(s))) for s in 1:4) / 4
		@test abs(est - H) < 0.1
	end
	@test_throws ArgumentError fbmSurface(8, 8, -1.5)

	# network: same topology as uncorrelated, exact weight distribution, more correlation
	# for larger H
	nx = 40
	g0, _, W0, _ = buildSquareNetwork(nx, nx; dist = 5.0, seed = 2)
	corr_next = Float64[]
	for H in (-1.0, 0.0, 0.7)
		g, _, W, _ = buildSquareNetwork(nx, nx; dist = 5.0, seed = 2, hurst = H)
		@test g == g0
		w = sort([W[src(e), dst(e)] for e in edges(g) if src(e) <= nx^2 && dst(e) <= nx^2])
		E = length(w)
		@test w ≈ [quantile(DisorderDist(5.0), (k - 0.5) / E) for k in 1:E]
		a = Float64[]; b = Float64[]
		for j in 1:(nx-1), i in 1:(nx-1)
			s1, s2 = i + nx * (j - 1), i + nx * j          # horizontal links one row apart
			# same direction only (directions use independent surfaces)
			for ((u, v), (x, y)) in (((s1, s1 + 1), (s2, s2 + 1)), ((s1 + 1, s1), (s2 + 1, s2)))
				has_edge(g, u, v) && has_edge(g, x, y) && (push!(a, log(W[u, v])); push!(b, log(W[x, y])))
			end
		end
		push!(corr_next, (sum(a .* b) / length(a) - sum(a) * sum(b) / length(a)^2) /
						 sqrt((sum(abs2, a) / length(a) - (sum(a) / length(a))^2) *
							  (sum(abs2, b) / length(b) - (sum(b) / length(b))^2)))
	end
	@test abs(corr_next[1]) < 0.1 && corr_next[1] < corr_next[2] < corr_next[3]
	@test corr_next[3] > 0.8

	# direction_rho = 1: the two directions of a link get adjacent ranks
	g, _, W, _ = buildSquareNetwork(20, 20; dist = 5.0, p = 0.0, od = false, hurst = 0.5,
		direction_rho = 1.0)
	E = ne(g)
	q(k) = quantile(DisorderDist(5.0), (k - 0.5) / E)
	@test all(e -> abs(log(W[src(e), dst(e)]) - log(W[dst(e), src(e)])) <=
				   log(q(E) / q(1)) / (E - 1) * 1.01, edges(g))

	# positive weights from a distribution that can go negative; cracking works
	g, _, W, _ = buildSquareNetwork(20, 20; dist = Normal(0.5, 0.3), hurst = 0.5)
	@test all(e -> W[src(e), dst(e)] > 0, edges(g))
	_, _, _, res = crackOptimalPaths(g, 401, 402, W)
	@test res.stop_reason == :disconnected
end

@testset "correlated weights" begin
	test_correlated_weights()
end

@testset "DisorderDist" begin
	test_DisorderDist()
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

function test_buildCityLattice()
	nx, ny = 20, 16
	g, coords, dm, wm, roads = buildCityLattice(nx, ny; oneway_fraction = 0.6,
		alternation = 1.0, arterial_fraction = 0.2, seed = 3)
	s = networkSummary(g, wm)
	@test s.mean_degree == 4.0 && s.scc_fraction == 1.0 && s.imbalance == 0.0
	@test length(coords) == nv(g) == nx * ny
	@test all(e -> dm[src(e), dst(e)] == 100.0 && wm[src(e), dst(e)] > 0, edges(g))

	# every block of a one-way road points in the road's direction
	node(i, j) = i + nx * (j - 1)
	for r in eachrow(roads)
		r.oneway || continue
		for k in 1:(r.orientation == :row ? nx : ny)
			a, b = r.orientation == :row ? (node(k, r.index), node(mod1(k + 1, nx), r.index)) :
				   (node(r.index, k), node(r.index, mod1(k + 1, ny)))
			r.direction == 1 ? (@test has_edge(g, a, b) && !has_edge(g, b, a)) :
			(@test has_edge(g, b, a) && !has_edge(g, a, b))
		end
	end
	# arterials two-way; alternation = 1 alternates consecutive one-way roads
	@test all(.!roads.oneway[roads.arterial])
	for o in (:row, :column)
		d = filter(r -> r.orientation == o && r.oneway, roads).direction
		@test all(d[1:(end-1)] .== .-d[2:end])
	end
	@test metadata(roads, "alternation") == 1.0

	# alternation = 0: all one-way roads of an orientation share a direction
	_, _, _, _, roads0 = buildCityLattice(nx, ny; oneway_fraction = 1.0, alternation = 0.0)
	for o in (:row, :column)
		@test length(unique(filter(r -> r.orientation == o && r.oneway, roads0).direction)) == 1
	end

	# target degree is exact; arterials are never cut
	g3, _, _, _, roads3 = buildCityLattice(nx, ny; mean_degree = 3.2, arterial_fraction = 0.25)
	@test networkSummary(g3).mean_degree ≈ 3.2
	for r in eachrow(filter(r -> r.arterial && r.orientation == :row, roads3)), i in 1:nx
		@test has_edge(g3, node(i, r.index), node(mod1(i + 1, nx), r.index))
	end
	@test_throws ArgumentError buildCityLattice(nx, ny; mean_degree = 4.5)

	# beta: with no road variation and no arterials, weights = weight_mean × factor, and the
	# factors are exactly the normalized DisorderDist quantiles, for any hurst
	nb = 40
	gb, _, _, _, _ = buildCityLattice(nb, nb; arterial_fraction = 0.0, road_cv = 0.0, seed = 4)
	for (β, H) in ((3.0, -1.0), (3.0, 0.7), (10.0, 0.3))
		g4, _, _, W4, _ = buildCityLattice(nb, nb; arterial_fraction = 0.0, road_cv = 0.0,
			weight_mean = 2.0, beta = β, hurst = H, seed = 4)
		@test g4 == gb                                   # weights never change the layout
		w = sort([W4[src(e), dst(e)] for e in edges(g4)])
		E = length(w)
		d = DisorderDist(β)
		@test w ≈ [2.0 * quantile(d, (k - 0.5) / E) / mean(d) for k in 1:E]
		@test isapprox(networkSummary(g4, W4).equivalent_beta, β; rtol = 0.01)
	end
	nodeb(i, j) = i + nb * (j - 1)
	# hurst: neighbouring parallel blocks (same direction) more correlated for larger H
	function next_corr(H)
		g5, _, _, W5, _ = buildCityLattice(nb, nb; oneway_fraction = 0.0, arterial_fraction = 0.0,
			road_cv = 0.0, beta = 5.0, hurst = H, seed = 6)
		a = [log(W5[nodeb(i, j), nodeb(i + 1, j)]) for j in 1:(nb-1), i in 1:(nb-1)]
		b = [log(W5[nodeb(i, j + 1), nodeb(i + 1, j + 1)]) for j in 1:(nb-1), i in 1:(nb-1)]
		a, b = vec(a) .- sum(a) / length(a), vec(b) .- sum(b) / length(b)
		return sum(a .* b) / sqrt(sum(abs2, a) * sum(abs2, b))
	end
	c = next_corr.((-1.0, 0.0, 0.7))
	@test abs(c[1]) < 0.1 && c[1] < c[2] < c[3] && c[3] > 0.8
	# coarse-grid layout: tree (q = 0), nested loops, full grid (q = 1)
	function fast_graph(q; pbc = true, n1 = 24, n2 = 16, s = 4, k = 4.0)
		g7, _, _, W7, r7 = buildCityLattice(n1, n2; arterial_spacing = s, loop_fraction = q,
			oneway_fraction = 0.5, arterial_speedup = 4.0, road_cv = 0.0,
			arterial_segment_cv = 0.0, pbc = pbc, mean_degree = k, seed = 5)
		fe = Set(minmax(src(e), dst(e)) for e in edges(g7) if W7[src(e), dst(e)] == 0.25)
		return g7, r7, fe
	end
	g7, r7, f0 = fast_graph(0.0)
	cx, cy = 24 ÷ 4, 16 ÷ 4
	@test length(f0) == (cx * cy - 1) * 4                  # spanning tree of s-block links
	ft = SimpleGraph(nv(g7))
	foreach(e -> add_edge!(ft, e...), f0)
	coarse = [i + 24 * (j - 1) for i in 1:4:24 for j in 1:4:16]
	@test all(c -> has_path(ft, coarse[1], c), coarse)     # connects every coarse node
	@test ne(ft) == count(>(0), degree(ft)) - 1            # and has no loop
	@test sort(metadata(r7, "arterial_nodes")) == findall(>(0), degree(ft))
	_, _, fh = fast_graph(0.5)
	_, r1, f1 = fast_graph(1.0)
	@test f0 ⊆ fh ⊆ f1 && length(f1) == 2 * cx * cy * 4    # nested; q = 1 is the full grid
	@test metadata(r1, "arterial_block_share") ≈ length(f1) / (2 * 24 * 16)
	@test all(.!r1.oneway[r1.arterial]) && count(r1.arterial) == cx + cy
	# local blocks of a coarse line are not fast; open lattice: links of s blocks only
	_, _, fo = fast_graph(1.0; pbc = false, n1 = 23, n2 = 13, k = 3.3)
	@test length(fo) == (((23 - 1) ÷ 4) * 4 * 4 + ((13 - 1) ÷ 4) * 6 * 4)
	@test_throws ArgumentError buildCityLattice(20, 16; arterial_spacing = 3)
	@test_throws ArgumentError buildCityLattice(24, 16; arterial_spacing = 4, loop_fraction = 1.5)
	# default layout unchanged by the new keywords
	@test buildCityLattice(nx, ny; seed = 3)[1] == buildCityLattice(nx, ny; seed = 3,
		loop_fraction = 0.2)[1]

	# arterials stay smooth whatever beta is
	g6, _, _, W6, roads6 = buildCityLattice(nb, nb; arterial_fraction = 0.2, beta = 20.0, seed = 2)
	for r in eachrow(filter(r -> r.arterial && r.orientation == :row, roads6))
		w = [W6[nodeb(i, r.index), nodeb(mod1(i + 1, nb), r.index)] for i in 1:nb]
		@test maximum(w) / minimum(w) < 1.3
	end

	# same seed -> same network; it plugs into crackStatistics
	@test buildCityLattice(nx, ny; seed = 9)[4] == buildCityLattice(nx, ny; seed = 9)[4]
	_, od = crackStatistics(g, coords, dm, wm, [300.0, 600.0]; n_od_pairs = 10)
	R, per = integratedRemovals(od)
	@test nrow(per) == 2 && R ≈ 300 * (per.mean[1] + per.mean[2]) / 2
end

@testset "city lattice" begin
	test_buildCityLattice()
end

@testset "very strong disorder terminates" begin
	# Weights spanning far more than Float64 resolution used to create cycles of
	# "tight" edges, so the backward path walk never ended.
	for β in (50.0, 200.0)
		g, _, W, _ = buildSquareNetwork(16, 16; p = 0.0, dist = β, seed = 7, od = true)
		net = OptimumPathCrack.CrackNetwork(g, W)
		r = OptimumPathCrack._crack(net, 257, 258; rng = Random.Xoshiro(1))
		@test r.stop_reason == :disconnected
		@test 16 <= length(r.removed) <= ne(g)
		# every removed edge was on a simple path: no edge removed twice
		@test allunique(r.removed)
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

@testset "square torus" begin
	# p = 0: every link two-way, so a torus has exactly 4 out-edges per node
	g, _, W, _ = buildSquareNetwork(6, 5; p = 0.0, dist = 2.0, seed = 3, od = false,
		pbc = true, pbc_y = true)
	@test nv(g) == 30 && ne(g) == 4 * 30
	@test all(outdegree(g, v) == 4 for v in 1:30)
	@test has_edge(g, 1 + 6 * 4, 1) && has_edge(g, 1, 1 + 6 * 4)   # (1,5) <-> (1,1)
	@test is_strongly_connected(g)
	# defaults unchanged: open in y
	g2, _, _, _ = buildSquareNetwork(6, 5; p = 0.0, dist = 2.0, seed = 3, od = false, pbc = true)
	@test ne(g2) == 2 * (6 * 5 + 6 * 4)
	# correlated weights on the torus
	g3, _, W3, _ = buildSquareNetwork(8, 8; p = 0.3, dist = 5.0, seed = 3, od = false,
		pbc = true, pbc_y = true, hurst = 0.5)
	@test all(W3[src(e), dst(e)] > 0 for e in edges(g3))
end

@testset "buildNetworkFromTables Tests" begin
    test_buildNetworkFromTables()
end

@testset "DiverCity" begin
    # net from (src, dst, weight) triples; lengths all 1
    function mknet(n, es)
        g = SimpleDiGraph(n)
        for (a, b, _) in es
            add_edge!(g, a, b)
        end
        W = sparse([e[1] for e in es], [e[2] for e in es], [Float64(e[3]) for e in es], n, n)
        return CrackNetwork(g, W)
    end
    # two disjoint equal routes 1-2-4, 1-3-4: two NSR, no overlap -> D = 2
    net = mknet(4, [(1, 2, 1), (2, 4, 1), (1, 3, 1), (3, 4, 1)])
    r = diverCity(net, 1, 4)
    @test r.n_routes == 2 && r.n_nsr == 2 && r.spread ≈ 1 && r.D ≈ 2
    # a third disjoint route 1.5 times as costly: out at ε = 0.3, in at ε = 0.6
    net = mknet(6, [(1, 2, 1), (2, 4, 1), (1, 3, 1), (3, 4, 1), (1, 5, 1), (5, 6, 1), (6, 4, 1)])
    @test diverCity(net, 1, 4).D ≈ 2
    @test diverCity(net, 1, 4; epsilon = 0.6).D ≈ 3
    @test diverCity(net, 1, 4; epsilon = 0.6, k = 2).n_routes == 2
    # shared first edge 1->2, then 2-3-5 or 2-4-5: J = 1/5, spread 0.8, D = 1.6
    net = mknet(5, [(1, 2, 1), (2, 3, 1), (3, 5, 1), (2, 4, 1), (4, 5, 1)])
    r = diverCity(net, 1, 5)
    @test r.n_nsr == 2 && r.spread ≈ 0.8 && r.D ≈ 1.6
    # lengths weight the Jaccard: long shared edge -> more similar
    r = diverCity(net, 1, 5; lengths = [8.0, 1, 1, 1, 1])
    @test r.spread ≈ 1 - 8 / 12
    # one route -> 0; no route -> NaN
    net = mknet(3, [(1, 2, 1), (2, 3, 1)])
    @test diverCity(net, 1, 3).D == 0
    @test isnan(diverCity(net, 3, 1).D)
    # statistics: thread-independent streams, one row per pair
    g, coords, dm, wm, roads = buildCityLattice(20, 20; seed = 2)
    net = CrackNetwork(g, wm)
    ods = [(1, 400), (21, 210), (5, 300)]
    s = diverCityStatistics(net, ods; lengths = edgeValues(net, dm))
    @test nrow(s) == 3 && all(0 .<= s.D .<= 10) && all(s.n_nsr .>= 1)
    @test s.D == diverCityStatistics(net, ods; lengths = edgeValues(net, dm)).D
end

@testset "city lattice attractor rings" begin
    nx = 21
    g, coords, dm, wm, roads = buildCityLattice(nx, nx; pbc = false, attractor_rings = [3, 6],
        attractor_speedup = 4.0, arterial_segment_cv = 0.0, mean_degree = 3.0, seed = 3)
    ring = metadata(roads, "attractor_nodes")
    @test length(ring) == 8 * 3 + 8 * 6
    node(i, j) = i + nx * (j - 1)
    c = (nx + 1) ÷ 2
    for d in (3, 6), i in (c - d):(c + d - 1)      # bottom side, both directions, fast
        a, b = node(i, c - d), node(i + 1, c - d)
        @test has_edge(g, a, b) && has_edge(g, b, a)
        @test wm[a, b] ≈ 0.25 && wm[b, a] ≈ 0.25
    end
    @test count(==(0.25), nonzeros(wm)) == 2 * 8 * (3 + 6)
    @test_throws ArgumentError buildCityLattice(nx, nx; pbc = false, attractor_rings = [11])
    # no rings: same lattice as before the option existed
    g0, _, _, w0, _ = buildCityLattice(nx, nx; seed = 3)
    g1, _, _, w1, _ = buildCityLattice(nx, nx; seed = 3, attractor_rings = Int[])
    @test g0 == g1 && w0 == w1
end
