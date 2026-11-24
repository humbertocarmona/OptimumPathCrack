"""
	DisorderDist(β)

Disorder distribution for edge weights: `w = exp(β(r - 1))` with `r` uniform on [0, 1].
`log w` is uniform on `[-β, 0]`, so weights lie in `[e^-β, 1]`, span `β/ln 10` orders of
magnitude, and have density `p(w) = 1/(β w)`. `β` sets the disorder strength: for large
`β` a path's cost is dominated by its largest weight (strong disorder); for small `β`
costs behave like sums (weak disorder). For any set of weights, `√12 · std(log w)` gives
the equivalent `β`.

Requires `0 < β ≤ 700`, so every weight is strictly positive (`e^-β` would underflow to
0 for `β ≳ 745`). Implements the Distributions.jl interface: `rand(rng, d)`,
`rand(d, n)`, `pdf`, `logpdf`, `cdf`, `quantile`, `mean`, `var`, `minimum`, `maximum`.
"""
struct DisorderDist <: ContinuousUnivariateDistribution
	β::Float64
	function DisorderDist(β::Real)
		0 < β <= 700 || throw(ArgumentError("DisorderDist needs 0 < β ≤ 700, got β = $β"))
		return new(Float64(β))
	end
end

Distributions.params(d::DisorderDist) = (d.β,)
Base.minimum(d::DisorderDist) = exp(-d.β)
Base.maximum(::DisorderDist) = 1.0
Distributions.insupport(d::DisorderDist, w::Real) = minimum(d) <= w <= 1

Base.rand(rng::AbstractRNG, d::DisorderDist) = exp(d.β * (rand(rng) - 1))

Distributions.logpdf(d::DisorderDist, w::Real) =
	insupport(d, w) ? -log(d.β) - log(w) : -Inf
Distributions.pdf(d::DisorderDist, w::Real) = exp(logpdf(d, w))
Distributions.cdf(d::DisorderDist, w::Real) =
	w < minimum(d) ? 0.0 : w >= 1 ? 1.0 : 1 + log(w) / d.β
Distributions.quantile(d::DisorderDist, p::Real) = exp(d.β * (p - 1))

Distributions.mean(d::DisorderDist) = -expm1(-d.β) / d.β
Distributions.var(d::DisorderDist) = -expm1(-2d.β) / (2d.β) - mean(d)^2

"""
	_positive_rand(rng, dist; max_tries = 1000)

Draws from `dist` conditioned on a strictly positive value (rejection sampling). Throws
if `dist` practically never produces positive values.
"""
function _positive_rand(rng::AbstractRNG, dist; max_tries::Int = 1000)
	for _ in 1:max_tries
		w = rand(rng, dist)
		w > 0 && return Float64(w)
	end
	throw(ArgumentError("weight distribution $dist gave no positive value in $max_tries draws"))
end
