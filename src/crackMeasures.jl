#=
Measures of efficiency and resilience computed from a crack.

A crack records the cost of every optimal path it finds, `path_costs`: `path_costs[k]` is
the cost of the path from which the k-th edge is removed, so the costs never decrease.
Stopping the crack at a cost bound gives the start of the uncapped crack, and the number of
removals within any bound can be read from a single run.
=#

"""
	removalsWithin(path_costs, n_removed, ε) -> Int
	removalsWithin(path_costs, n_removed, εs::AbstractVector) -> Vector{Int}

Number of removals made while the optimal path cost at most `(1 + ε)` times the first one,
N_ε = max{k : C_k ≤ (1 + ε) C_1}: the removals that still leave a route at most a fraction
`ε` costlier than the original. `path_costs` and `n_removed` are those of one crack
(`_crack`, `crackOptimalPaths`), stopped at any cost ratio of at least `1 + ε`. When the
pair is disconnected first, every removal counts. Returns 0 for a crack without a path.
"""
function removalsWithin(path_costs::AbstractVector{<:Real}, n_removed::Integer, ε::Real)
	ε >= 0 || throw(ArgumentError("ε must be ≥ 0"))
	isempty(path_costs) && return 0
	bound = (1 + ε) * path_costs[1] * (1 + TIE_RTOL)
	return count(c -> c <= bound, view(path_costs, 1:min(length(path_costs), n_removed)))
end

removalsWithin(path_costs::AbstractVector{<:Real}, n_removed::Integer,
	εs::AbstractVector{<:Real}) = [removalsWithin(path_costs, n_removed, ε) for ε in εs]

"""
	routeEfficiency(T1, distance; speed = 1.0) -> Float64

Efficiency of a trip, η = (distance / speed) / T1: the time of a straight drive at the
reference `speed` divided by the cost (travel time) `T1` of the optimal path. On a grid
with uniform travel times η = 1 along a road and 1/√2 on the diagonal; fast roads can make
it larger than 1. With travel times in units of the mean block time and the distance in
blocks, `speed = 1`.
"""
function routeEfficiency(T1::Real, distance::Real; speed::Real = 1.0)
	speed > 0 || throw(ArgumentError("speed must be > 0"))
	return distance / speed / T1
end

"""
	epsTag(ε) -> String

Column suffix for a bound ε, as used by `crackStatistics(...; eps)`: 0.3 → "0p3".
"""
epsTag(ε::Real) = replace(string(Float64(ε)), "." => "p")
