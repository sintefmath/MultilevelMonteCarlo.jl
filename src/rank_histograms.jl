#= ── Gregory MLMC resampling ─────────────────────────────────────────────── =#

"""
    ml_gregory_resample(samples::MLMCSamples, qoi_index::Int,
                        number_of_resamples::Int) -> Vector{Float64}

Generate resampled values from an MLMC ensemble using Gregory's multilevel
inverse-CDF method.

Fine and coarse QoI samples are sorted **independently** at each level to form
per-level empirical quantile functions, which are then combined via the MLMC
telescoping sum:

```math
\\hat{Q}^{-1}(u) = \\hat{F}_1^{-1}(u)
  + \\sum_{l=2}^{L}\\bigl[\\hat{F}_l^{-1}(u) - \\hat{G}_l^{-1}(u)\\bigr]
```

where ``\\hat{F}_l^{-1}`` and ``\\hat{G}_l^{-1}`` are the empirical quantile
functions of the fine and coarse samples at level ``l``.  Random draws
``u \\sim \\mathrm{Uniform}(0,1)`` are passed through this inverse CDF to
produce resamples.

# Arguments
- `samples`: An [`MLMCSamples`](@ref) struct.
- `qoi_index`: Which QoI (row index) to resample.
- `number_of_resamples`: How many i.i.d. resampled values to generate.

# Returns
`Vector{Float64}` of resampled QoI values.
"""
function ml_gregory_resample(samples::MLMCSamples, qoi_index::Int,
                             number_of_resamples::Int)
    j = qoi_index
    @assert 1 <= j <= samples.n_qois

    sorted_fine   = [sort(samples.fine[lvl][j, :]) for lvl in 1:samples.n_levels]
    sorted_coarse = [sort(samples.coarse[lvl][j, :]) for lvl in 1:samples.n_levels]

    function inverse_cdf(u)
        n1 = length(sorted_fine[1])
        val = sorted_fine[1][clamp(ceil(Int, u * n1), 1, n1)]

        for lvl in 2:samples.n_levels
            nl = length(sorted_fine[lvl])
            idx = clamp(ceil(Int, u * nl), 1, nl)
            val += sorted_fine[lvl][idx] - sorted_coarse[lvl][idx]
        end
        return val
    end

    return [inverse_cdf(rand()) for _ in 1:number_of_resamples]
end

#= ── Rank histograms ────────────────────────────────────────────────────── =#

"""
    rank_histogram_gregory(levels, qoi_function, draw_parameters,
                           number_of_rank_samples, samples_per_level;
                           number_of_resamples=1000, parallel=false)
                           -> Vector{Int}

Compute a rank histogram using Gregory's MLMC inverse-CDF resampling.

For each of the `number_of_rank_samples` iterations:

1. Draw a "truth" value ``y = q\\bigl(L_\\text{finest}(\\theta)\\bigr)`` where
   ``\\theta \\sim`` `draw_parameters()`.
2. Run a fresh MLMC sampling campaign via [`mlmc_sample`](@ref).
3. Generate `number_of_resamples` values from the MLMC ensemble using
   [`ml_gregory_resample`](@ref).
4. Sort the resampled array and record the rank of ``y``.

If the MLMC ensemble correctly represents the true distribution, the returned
ranks are approximately ``\\mathrm{Uniform}\\{1,\\ldots,N_r+1\\}`` where
``N_r`` = `number_of_resamples`.

# Arguments
- `levels`: Model evaluators `[L₁, …, L_L]`.
- `qoi_function`: `q(model_output) -> scalar`.
- `draw_parameters`: `() -> params`.
- `number_of_rank_samples`: Number of rank evaluations (outer loop).
- `samples_per_level`: Sample counts for each MLMC ensemble.

# Keyword Arguments
- `number_of_resamples::Int = 1000`: Resampled values per ensemble.
- `parallel::Bool = false`: Thread MLMC sampling within each ensemble.

# Returns
`Vector{Int}` of ranks, each in ``\\{1,\\ldots,N_r+1\\}``.
"""
function rank_histogram_gregory(
    levels::AbstractVector{<:Function},
    qoi_function::Function,
    draw_parameters::Function,
    number_of_rank_samples::Int,
    samples_per_level::AbstractVector{<:Integer};
    number_of_resamples::Int = 1000,
    parallel::Bool = false,
)
    qoi_functions = Function[qoi_function]
    ranks = Vector{Int}(undef, number_of_rank_samples)

    for i in 1:number_of_rank_samples
        y = qoi_function(levels[end](draw_parameters()))

        samples = mlmc_sample(levels, qoi_functions, samples_per_level,
                              draw_parameters; parallel)
        resampled = ml_gregory_resample(samples, 1, number_of_resamples)
        sort!(resampled)
        ranks[i] = searchsortedlast(resampled, y) + 1
    end

    return ranks
end

"""
    rank_histogram_cdf(levels, qoi_function, draw_parameters,
                       number_of_rank_samples, samples_per_level,
                       cdf_method; parallel=false) -> Vector{Float64}

Compute a probability-integral-transform (PIT) histogram using an MLMC-based
CDF estimate.

For each of the `number_of_rank_samples` iterations:

1. Draw a "truth" value ``y = q\\bigl(L_\\text{finest}(\\theta)\\bigr)``.
2. Run a fresh MLMC sampling campaign.
3. Estimate the CDF ``\\hat{F}`` from the MLMC samples using `cdf_method`.
4. Record ``\\hat{F}(y)``.

If the CDF estimate is well calibrated, the returned PIT values are
approximately ``\\mathrm{Uniform}(0,1)``.

# Arguments
- `levels`: Model evaluators `[L₁, …, L_L]`.
- `qoi_function`: `q(model_output) -> scalar`.
- `draw_parameters`: `() -> params`.
- `number_of_rank_samples`: Number of PIT evaluations (outer loop).
- `samples_per_level`: Sample counts for each MLMC ensemble.
- `cdf_method`: `(samples::MLMCSamples, qoi_index::Int) -> cdf_func` where
  `cdf_func(x)::Float64`.  Pass [`estimate_cdf_mlmc_kernel_density`](@ref) for
  KDE, or `(s, i) -> first(estimate_cdf_maxent(s, i; R=6))` for MaxEnt.

# Keyword Arguments
- `parallel::Bool = false`: Thread MLMC sampling within each ensemble.

# Returns
`Vector{Float64}` of PIT values, each ideally in ``[0, 1]``.
"""
function rank_histogram_cdf(
    levels::AbstractVector{<:Function},
    qoi_function::Function,
    draw_parameters::Function,
    number_of_rank_samples::Int,
    samples_per_level::AbstractVector{<:Integer},
    cdf_method::Function;
    parallel::Bool = false,
)
    qoi_functions = Function[qoi_function]
    pit_values = Vector{Float64}(undef, number_of_rank_samples)

    for i in 1:number_of_rank_samples
        y = qoi_function(levels[end](draw_parameters()))

        samples = mlmc_sample(levels, qoi_functions, samples_per_level,
                              draw_parameters; parallel)
        cdf_func = cdf_method(samples, 1)
        pit_values[i] = cdf_func(y)
    end

    return pit_values
end
