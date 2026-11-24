"""
	fbmSurface(Lx, Ly, H; rng = Random.default_rng()) -> Matrix{Float64}

Random correlated surface on a periodic `Lx × Ly` grid, made by Fourier filtering:
white Gaussian noise whose Fourier modes are scaled by `|k|^-(H+1)`, i.e. power spectrum
`S(k) ∝ |k|^-2(H+1)`. The result is normalized to zero mean and unit variance.

- `H = -1`: uncorrelated (white noise).
- `-1 < H < 0`: correlations decaying as a power law, `C(r) ∝ r^2H`.
- `0 < H < 1`: fractional Brownian surface with Hurst exponent `H`; height differences
  grow as `⟨(h(x+r) - h(x))²⟩ ∝ r^2H` for `r` much smaller than the grid.

The surface is periodic in both directions.
"""
function fbmSurface(Lx::Int, Ly::Int, H::Real; rng::AbstractRNG = Random.default_rng())
	H >= -1 || throw(ArgumentError("Hurst exponent must be ≥ -1 (got H = $H)"))
	kx = 2π .* fftfreq(Lx)
	ky = 2π .* fftfreq(Ly)
	amplitude = [(k2 = kx[i]^2 + ky[j]^2) == 0 ? 0.0 : k2^(-(H + 1) / 2)
				 for i in 1:Lx, j in 1:Ly]
	h = real(ifft(fft(randn(rng, Lx, Ly)) .* amplitude))
	h .-= sum(h) / length(h)
	h ./= sqrt(sum(abs2, h) / length(h))
	return h
end
