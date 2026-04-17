#= ── Multivariate CDF estimation via MLMC product-kernel KDE ──────────── =#

"""
    estimate_cdf_multivariate_mlmc_kernel_density(samples::MLMCSamples,
        qoi_indices::AbstractVector{<:Integer}; bandwidth=nothing) -> Function

Estimate the joint multivariate CDF for the specified QoIs from MLMC samples
using a product of Gaussian-CDF kernels with the MLMC telescoping sum.

The estimator uses a product of univariate Gaussian CDF kernels:

```math
\\hat{F}(\\mathbf{x}) = \\frac{1}{N_1} \\sum_{i=1}^{N_1}
  \\prod_{k=1}^{d} \\Phi\\!\\left(\\frac{x_k - Q_{1,k}^{(i)}}{h_k}\\right)
  + \\sum_{l=2}^{L} \\frac{1}{N_l} \\sum_{i=1}^{N_l}
    \\left[\\prod_{k} \\Phi\\!\\left(\\frac{x_k - F_{l,k}^{(i)}}{h_k}\\right)
    - \\prod_{k} \\Phi\\!\\left(\\frac{x_k - C_{l,k}^{(i)}}{h_k}\\right)\\right]
```

where ``\\Phi`` is the standard normal CDF and ``h_k`` is the per-dimension
bandwidth (Silverman's rule by default).

# Arguments
- `samples`: An [`MLMCSamples`](@ref) struct.
- `qoi_indices`: Which QoI rows to include as dimensions of the joint CDF.

# Keyword Arguments
- `bandwidth`: Per-dimension bandwidths as a vector. If `nothing`, Silverman's
  rule is applied per dimension using finest-level fine samples.

# Returns
A callable `F(x::AbstractVector) -> Float64` evaluating the estimated joint CDF.
"""
function estimate_cdf_multivariate_mlmc_kernel_density(
    samples::MLMCSamples,
    qoi_indices::AbstractVector{<:Integer};
    bandwidth::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    d = length(qoi_indices)
    @assert all(1 .<= qoi_indices .<= samples.n_qois)

    # Bandwidth per dimension via Silverman's rule
    h = Vector{Float64}(undef, d)
    for (k, j) in enumerate(qoi_indices)
        if bandwidth === nothing
            finest = samples.fine[end][j, :]
            σ_val = std(finest)
            n = length(finest)
            h[k] = 1.06 * σ_val * n^(-1/5)
        else
            h[k] = Float64(bandwidth[k])
        end
    end

    # Pre-extract data for the closure
    fine_data = [[copy(samples.fine[lvl][j, :]) for j in qoi_indices]
                  for lvl in 1:samples.n_levels]
    coarse_data = [[copy(samples.coarse[lvl][j, :]) for j in qoi_indices]
                    for lvl in 1:samples.n_levels]

    Φ_dist = Normal(0.0, 1.0)

    function cdf_estimate(x::AbstractVector{<:Real})
        # Level 1
        N1 = length(fine_data[1][1])
        val = 0.0
        for i in 1:N1
            p = 1.0
            for k in 1:d
                p *= dist_cdf(Φ_dist, (x[k] - fine_data[1][k][i]) / h[k])
            end
            val += p
        end
        val /= N1

        # Levels 2…L: corrections
        for lvl in 2:samples.n_levels
            Nl = length(fine_data[lvl][1])
            corr = 0.0
            for i in 1:Nl
                pf = 1.0
                pc = 1.0
                for k in 1:d
                    pf *= dist_cdf(Φ_dist, (x[k] - fine_data[lvl][k][i]) / h[k])
                    pc *= dist_cdf(Φ_dist, (x[k] - coarse_data[lvl][k][i]) / h[k])
                end
                corr += pf - pc
            end
            val += corr / Nl
        end

        return val
    end

    return cdf_estimate
end

#= ── Bootstrap resampling for multivariate MLMC ─────────────────────────── =#

"""
    ml_bootstrap_resample_multivariate(samples::MLMCSamples,
        qoi_indices::AbstractVector{<:Integer},
        number_of_resamples::Int) -> Matrix{Float64}

Generate multivariate resamples from an MLMC ensemble using a bootstrap-style
telescoping sum.

For each resample, a random sample index is drawn independently at each level
(shared across all QoI dimensions within that level), and the MLMC telescoping
sum is formed:

```math
\\hat{\\mathbf{x}} = \\mathbf{F}_1^{(i_1)}
  + \\sum_{l=2}^{L} \\bigl[\\mathbf{F}_l^{(i_l)} - \\mathbf{C}_l^{(i_l)}\\bigr]
```

This preserves within-sample correlations between QoI dimensions at each level.

# Arguments
- `samples`: An [`MLMCSamples`](@ref) struct.
- `qoi_indices`: Which QoI rows to include as dimensions.
- `number_of_resamples`: How many resampled vectors to generate.

# Returns
`Matrix{Float64}` of size `(d, number_of_resamples)`.
"""
function ml_bootstrap_resample_multivariate(
    samples::MLMCSamples,
    qoi_indices::AbstractVector{<:Integer},
    number_of_resamples::Int,
)
    d = length(qoi_indices)
    @assert all(1 .<= qoi_indices .<= samples.n_qois)
    resampled = Matrix{Float64}(undef, d, number_of_resamples)

    for r in 1:number_of_resamples
        # Level 1: random sample
        N1 = size(samples.fine[1], 2)
        i1 = rand(1:N1)
        for k in 1:d
            resampled[k, r] = samples.fine[1][qoi_indices[k], i1]
        end

        # Levels 2…L: random correction (same index for all dimensions)
        for lvl in 2:samples.n_levels
            Nl = size(samples.fine[lvl], 2)
            il = rand(1:Nl)
            for k in 1:d
                resampled[k, r] += samples.fine[lvl][qoi_indices[k], il] -
                                   samples.coarse[lvl][qoi_indices[k], il]
            end
        end
    end

    return resampled
end

#= ── Multivariate rank histogram ────────────────────────────────────────── =#

"""
    multivariate_rank_histogram(observations, levels, qoi_functions,
        samples_per_level, draw_parameters;
        number_of_resamples=1000, cdf_method=nothing, parallel=false)
        -> Vector{Float64}

Compute a multivariate rank histogram (MRH) using the MLMC-estimated
multivariate CDF for a given matrix of observations.

Implements the approach of Gneiting et al. (2008): the multivariate rank of an
observation ``\\mathbf{x}_0`` relative to an ensemble is

```math
\\mathrm{rank}_{\\mathrm{MRH}}(\\mathbf{x}_0, \\hat{F}_m)
  = \\hat{F}_{G}\\bigl(G(\\mathbf{x}_0)\\bigr)
```

where ``G(\\mathbf{x}) = \\hat{F}(\\mathbf{x})`` is the multivariate CDF
and ``\\hat{F}_G`` is the empirical CDF of the ``G``-values over the ensemble.

For each observation column in `observations`:

1. Run [`mlmc_sample`](@ref) to obtain MLMC samples.
2. Estimate ``G = \\hat{F}`` via the chosen CDF method.
3. Generate an ensemble via
   [`ml_bootstrap_resample_multivariate`](@ref).
4. Compute ``g_0 = G(\\mathbf{y})`` and ``g_j = G(\\mathbf{x}_j)`` for each
   ensemble member.
5. Record the PIT value ``\\hat{F}_G(g_0) = \\frac{1}{m}\\#\\{j : g_j \\le g_0\\}``.

If the MLMC ensemble is well calibrated, the returned PIT values are
approximately ``\\mathrm{Uniform}(0,1)``.

# Arguments
- `observations`: `Matrix{<:Real}` of size `(d, n)` — each column is one
  d-dimensional observation.
- `levels`: Model evaluators `[L₁, …, L_L]`.  Each returns a d-dimensional
  output (or an object from which `qoi_functions` extract d scalars).
- `qoi_functions`: `[q₁, …, q_d]` — d QoI extractors, each
  `qⱼ(model_output) -> scalar`.  Currently must be exactly **2** QoIs.
- `samples_per_level`: MLMC sample counts per level.
- `draw_parameters`: `() -> params`.

# Keyword Arguments
- `number_of_resamples::Int = 1000`: Ensemble size for each rank evaluation.
- `cdf_method`: `(samples, (j₁,j₂)) -> F̂(x₁,x₂)`, a callable that takes
  `MLMCSamples` and a tuple of QoI indices and returns a 2-D CDF function.
  Default: [`estimate_cdf_mlmc_kernel_density_2d`](@ref).
  For MaxEnt pass `(s, idx) -> first(estimate_cdf_maxent_2d(s, idx; R=4))`.
- `parallel::Bool = false`: Thread MLMC sampling within each ensemble.

# Returns
`Vector{Float64}` of PIT values, each ideally in ``[0, 1]``.
"""
function multivariate_rank_histogram(
    observations::AbstractMatrix{<:Real},
    levels::AbstractVector{<:Function},
    qoi_functions::AbstractVector{<:Function},
    samples_per_level::AbstractVector{<:Integer},
    draw_parameters::Function;
    number_of_resamples::Int = 1000,
    cdf_method::Union{Function, Nothing} = nothing,
    parallel::Bool = false,
)
    d = length(qoi_functions)
    @assert d == 2 "multivariate_rank_histogram currently requires exactly 2 QoIs"
    @assert size(observations, 1) == d "observations must have d=$(d) rows"
    n_obs = size(observations, 2)
    qoi_indices = (1, 2)
    pit_values = Vector{Float64}(undef, n_obs)

    _cdf_method = something(cdf_method, estimate_cdf_mlmc_kernel_density_2d)

    for i in 1:n_obs
        y1 = observations[1, i]
        y2 = observations[2, i]

        # MLMC sampling campaign
        samples = mlmc_sample(levels, qoi_functions, samples_per_level,
                              draw_parameters; parallel)

        # Estimate 2-D CDF
        G= _cdf_method(samples, qoi_indices)

        # Bootstrap-resample ensemble
        # ensemble = ml_bootstrap_resample_multivariate(samples, collect(1:d),
        #                                               number_of_resamples)

        samples_G = MLMCSamples(samples, G)


        cdf_F = estimate_cdf_mlmc_kernel_density(samples_G, 1)
        
        pit_values[i] = cdf_F(G(y1, y2)) # PIT value for the observation
    end

    return pit_values
end

"""
    multivariate_rank_histogram(levels, qoi_functions, draw_parameters,
        number_of_rank_samples, samples_per_level;
        number_of_resamples=1000, cdf_method=nothing, parallel=false)
        -> Vector{Float64}

Convenience wrapper that draws `number_of_rank_samples` observations from the
finest level and then calls the observation-based
[`multivariate_rank_histogram`](@ref).
"""
function multivariate_rank_histogram(
    levels::AbstractVector{<:Function},
    qoi_functions::AbstractVector{<:Function},
    draw_parameters::Function,
    number_of_rank_samples::Int,
    samples_per_level::AbstractVector{<:Integer};
    number_of_resamples::Int = 1000,
    cdf_method::Union{Function, Nothing} = nothing,
    parallel::Bool = false,
)
    d = length(qoi_functions)
    observations = Matrix{Float64}(undef, d, number_of_rank_samples)
    for i in 1:number_of_rank_samples
        params = draw_parameters()
        out = levels[end](params)
        for k in 1:d
            observations[k, i] = qoi_functions[k](out)
        end
    end
    return multivariate_rank_histogram(observations, levels, qoi_functions,
                                       samples_per_level, draw_parameters;
                                       number_of_resamples, cdf_method, parallel)
end
