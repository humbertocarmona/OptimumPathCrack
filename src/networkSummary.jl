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
- `weight_mean`, `weight_cv`, and `equivalent_beta = √12 · std(log w)`, the `β` of a
  `DisorderDist` with the same spread of `log w` (if `weight_matrix` is given).
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

	w_mean, w_cv, β_eq = NaN, NaN, NaN
	if weight_matrix !== nothing
		w = [weight_matrix[src(e), dst(e)] for e in edges(g)]
		w_mean = sum(w) / length(w)
		w_cv = sqrt(sum(abs2, w .- w_mean) / length(w)) / w_mean
		lw = log.(w)
		β_eq = sqrt(12 * sum(abs2, lw .- sum(lw) / length(lw)) / length(lw))
	end

	return (n_nodes = N, n_edges = ne(g), mean_degree = sum(und) / N,
		mean_out_degree = ne(g) / N, oneway_fraction = n_oneway / n_segments,
		dead_end_fraction = count(==(1), und) / N, scc_fraction = scc / N,
		imbalance = imbalance, weight_mean = w_mean, weight_cv = w_cv, equivalent_beta = β_eq)
end

"""
	integratedRemovals(od_df; column = :n_removed, normalize = false) -> (R, per_distance)

Robustness measure from a `crackStatistics` OD table: the mean of `column` at each
distance ℓ, integrated over ℓ with the trapezoidal rule. Also returns the per-distance
table (`ℓ`, `mean`, `sem`, `n`) so the curve can be plotted. With `normalize = true` the
integral is divided by the ℓ range, giving the average number of removals per pair,
R = (1/(ℓmax − ℓmin)) ∫ ⟨N⟩ dℓ; with `column = :n_eps_0p3` (`crackStatistics(...; eps)`)
this is the resilience R_ε at ε = 0.3.
"""
function integratedRemovals(od_df::DataFrame; column::Symbol = :n_removed,
	normalize::Bool = false)
	per = combine(groupby(od_df, :ℓ), column => (x -> sum(x) / length(x)) => :mean,
		column => (x -> length(x) > 1 ?
						sqrt(sum(abs2, x .- sum(x) / length(x)) / (length(x) - 1) / length(x)) :
						NaN) => :sem,
		nrow => :n)
	sort!(per, :ℓ)
	ℓ, m = per.ℓ, per.mean
	R = sum((ℓ[k+1] - ℓ[k]) * (m[k+1] + m[k]) / 2 for k in 1:(length(ℓ)-1); init = 0.0)
	normalize && length(ℓ) > 1 && (R /= (last(ℓ) - first(ℓ)))
	return R, per
end
