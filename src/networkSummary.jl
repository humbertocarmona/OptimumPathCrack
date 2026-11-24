"""
	networkSummary(g::SimpleDiGraph, weight_matrix = nothing) -> NamedTuple

Structural measures of a directed street network, for comparing generated lattices
with real cities and for use as variables in a parameter study:
- `n_nodes`, `n_edges`;
- `mean_degree`: mean number of distinct street neighbours (direction ignored);
- `mean_out_degree`: directed edges per node;
- `oneway_fraction`: fraction of street segments (neighbour pairs) with one direction only;
- `dead_end_fraction`: nodes with a single street neighbour;
- `scc_fraction`: fraction of nodes in the largest strongly connected component;
- `imbalance`: mean |in-degree − out-degree|;
- `weight_mean`, `weight_cv` (if `weight_matrix` is given).
"""
function networkSummary(g::SimpleDiGraph, weight_matrix = nothing)
	N = nv(g)
	und = zeros(Int, N)
	n_segments = 0
	n_oneway = 0
	for u in 1:N
		nbrs = all_neighbors(g, u)
		und[u] = length(nbrs)
		for v in nbrs
			u < v || continue
			n_segments += 1
			(has_edge(g, u, v) && has_edge(g, v, u)) || (n_oneway += 1)
		end
	end
	scc = maximum(length, strongly_connected_components(g))
	imbalance = sum(abs(indegree(g, u) - outdegree(g, u)) for u in 1:N) / N

	w_mean, w_cv = NaN, NaN
	if weight_matrix !== nothing
		w = [weight_matrix[src(e), dst(e)] for e in edges(g)]
		w_mean = sum(w) / length(w)
		w_cv = sqrt(sum(abs2, w .- w_mean) / length(w)) / w_mean
	end

	return (n_nodes = N, n_edges = ne(g), mean_degree = sum(und) / N,
		mean_out_degree = ne(g) / N, oneway_fraction = n_oneway / n_segments,
		dead_end_fraction = count(==(1), und) / N, scc_fraction = scc / N,
		imbalance = imbalance, weight_mean = w_mean, weight_cv = w_cv)
end

"""
	integratedRemovals(od_df; column = :n_removed) -> (R, per_distance)

Robustness measure from a `crackStatistics` OD table: the mean of `column` at each
distance ℓ, integrated over ℓ with the trapezoidal rule. Also returns the per-distance
table (`ℓ`, `mean`, `sem`, `n`) so the curve can be plotted. Divide `R` by the ℓ range
to get the average number of removals per pair.
"""
function integratedRemovals(od_df::DataFrame; column::Symbol = :n_removed)
	per = combine(groupby(od_df, :ℓ), column => (x -> sum(x) / length(x)) => :mean,
		column => (x -> length(x) > 1 ?
						sqrt(sum(abs2, x .- sum(x) / length(x)) / (length(x) - 1) / length(x)) :
						NaN) => :sem,
		nrow => :n)
	sort!(per, :ℓ)
	ℓ, m = per.ℓ, per.mean
	R = sum((ℓ[k+1] - ℓ[k]) * (m[k+1] + m[k]) / 2 for k in 1:(length(ℓ)-1); init = 0.0)
	return R, per
end
