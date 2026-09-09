#= ── Online statistics accumulator (Welford / Chan) ─────────────────────── =#

"""
    OnlineAccumulator()

Welford's online accumulator for computing running mean and variance in a
single pass without storing individual samples.

Two accumulators can be combined with `merge` (Chan's parallel algorithm),
making this safe for threaded use with per-thread instances.
"""
mutable struct OnlineAccumulator
    n::Int
    mean::Float64
    M2::Float64          # running sum of squared deviations from the mean
end

OnlineAccumulator() = OnlineAccumulator(0, 0.0, 0.0)

"""Update the accumulator with a new observation (Welford's algorithm)."""
function update!(acc::OnlineAccumulator, x::Real)
    acc.n += 1
    delta = x - acc.mean
    acc.mean += delta / acc.n
    acc.M2 += delta * (x - acc.mean)
    return acc
end

"""Return the sample variance, or `0.0` if fewer than 2 observations."""
function get_variance(acc::OnlineAccumulator)
    acc.n < 2 && return 0.0
    return acc.M2 / (acc.n - 1)
end

get_mean(acc::OnlineAccumulator) = acc.mean
get_count(acc::OnlineAccumulator) = acc.n

"""Merge two accumulators using Chan's parallel algorithm."""
function Base.merge(a::OnlineAccumulator, b::OnlineAccumulator)
    n = a.n + b.n
    n == 0 && return OnlineAccumulator()
    delta = b.mean - a.mean
    combined_mean = (a.n * a.mean + b.n * b.mean) / n
    combined_M2 = a.M2 + b.M2 + delta^2 * a.n * b.n / n
    return OnlineAccumulator(n, combined_mean, combined_M2)
end

"""Merge `src` into `dest` in-place."""
function merge!(dest::OnlineAccumulator, src::OnlineAccumulator)
    m = merge(dest, src)
    dest.n = m.n; dest.mean = m.mean; dest.M2 = m.M2
    return dest
end

#= ── Sample dispatchers ─────────────────────────────────────────────────── =#

"""
    accumulate_samples(f, n; parallel=false) -> OnlineAccumulator

Call `f()` exactly `n` times and accumulate scalar return values online.
When `parallel=true`, per-thread accumulators are merged via Chan's algorithm.

```julia
acc = accumulate_samples(1000; parallel=true) do
    my_expensive_computation()
end
```
"""
function accumulate_samples(f::Function, n::Int; parallel::Bool = false)
    if parallel
        nt = Threads.nthreads()
        accs = [OnlineAccumulator() for _ in 1:nt]
        Threads.@threads for _ in 1:n
            update!(accs[Threads.threadid()], f())
        end
        return reduce(merge, accs)
    else
        acc = OnlineAccumulator()
        for _ in 1:n
            update!(acc, f())
        end
        return acc
    end
end

"""Like [`accumulate_samples`](@ref) but merges into an existing accumulator."""
function accumulate_samples!(f::Function, acc::OnlineAccumulator, n::Int; parallel::Bool = false)
    return merge!(acc, accumulate_samples(f, n; parallel))
end

"""
    accumulate_vector_samples(body!, n, k; parallel=false) -> Vector{OnlineAccumulator}

Execute `body!(accs)` exactly `n` times, where `accs` is a `k`-element vector
of [`OnlineAccumulator`](@ref)s. The caller should call `update!(accs[j], val)`
inside `body!` to record observations. When `parallel=true` each thread gets
its own accumulator vector; results are merged afterwards.

This avoids per-sample allocation — only the accumulators are mutated.

```julia
accs = accumulate_vector_samples(1000, 2; parallel=true) do accs
    x = rand()
    update!(accs[1], x)
    update!(accs[2], x^2)
end
```
"""
function accumulate_vector_samples(body!::Function, n::Int, k::Int; parallel::Bool = false)
    if parallel
        nt = Threads.nthreads()
        all_accs = [[OnlineAccumulator() for _ in 1:k] for _ in 1:nt]
        Threads.@threads for _ in 1:n
            body!(all_accs[Threads.threadid()])
        end
        return [reduce(merge, [all_accs[t][j] for t in 1:nt]) for j in 1:k]
    else
        accs = [OnlineAccumulator() for _ in 1:k]
        for _ in 1:n
            body!(accs)
        end
        return accs
    end
end

"""Like [`accumulate_vector_samples`](@ref) but merges into existing accumulators."""
function accumulate_vector_samples!(body!::Function, accs::Vector{OnlineAccumulator}, n::Int; parallel::Bool = false)
    new_accs = accumulate_vector_samples(body!, n, length(accs); parallel)
    for j in eachindex(accs)
        merge!(accs[j], new_accs[j])
    end
    return accs
end

#= ── MLMC core functions ────────────────────────────────────────────────── =#

"""
    _default_variance_qoi(qoi_functions)

Return a callable `out -> Σᵢ qᵢ(out)²` used as the default scalar QoI
for variance-based sample allocation when multiple QoIs are present.
"""
_default_variance_qoi(qoi_functions) = out -> sum(q(out)^2 for q in qoi_functions)

"""
    variance_per_level(levels, qoi_functions, samples_per_level, draw_parameters;
                       variance_qoi=nothing, parallel=false)
                       -> (variances_corrections, variances)

Estimate the variance of a scalar QoI and its multilevel corrections at each
level. Statistics are computed online (Welford's algorithm).

The returned variances drive optimal sample allocation via
[`optimal_samples_per_level`](@ref). A single scalar `variance_qoi` is used for
allocation; the individual `qoi_functions` are only needed to construct the
default `variance_qoi`.

# Arguments
- `levels`: Vector of model evaluators `[L₁, …, Lₗ]`, one per fidelity level.
  Each `Lₗ(params) -> model_output` runs the forward model.
- `qoi_functions`: Vector of QoI extractors `[q₁, …, qQ]`.
  Each `qⱼ(model_output) -> scalar`. Used to construct the default `variance_qoi`.
- `samples_per_level`: Pilot sample counts, one per level.
- `draw_parameters`: `() -> params`.

# Keyword Arguments
- `variance_qoi`: `v(model_output) -> scalar` whose correction variance drives
  allocation. Default: `v(out) = Σᵢ qᵢ(out)²`.
- `parallel::Bool = false`: Use threading within each level.

# Returns
`(variances_corrections, variances)`:
- `variances_corrections[l]` = `Var[v(Lₗ₊₁(p)) − v(Lₗ(p))]` for `l = 1…L−1`
- `variances[l]` = `Var[v(Lₗ(p))]` for `l = 1…L`

For a single-level ensemble `variances_corrections` is an empty vector.
"""
function variance_per_level(
    levels::AbstractVector{<:Function},
    qoi_functions::AbstractVector{<:Function},
    samples_per_level::AbstractVector{<:Integer},
    draw_parameters::Function;
    variance_qoi::Union{Function, Nothing} = nothing,
    parallel::Bool = false,
)
    n_levels = length(levels)
    @assert length(samples_per_level) == n_levels "samples_per_level length must match levels"

    vqoi = something(variance_qoi, _default_variance_qoi(qoi_functions))

    variances = zeros(Float64, n_levels)
    variances_corrections = zeros(Float64, max(n_levels - 1, 0))

    # Level 1: plain Monte Carlo
    acc1 = accumulate_samples(samples_per_level[1]; parallel) do
        vqoi(levels[1](draw_parameters()))
    end
    variances[1] = get_variance(acc1)

    # Levels 2…L: correction + fine value for the variance QoI
    for lvl in 2:n_levels
        accs = accumulate_vector_samples(samples_per_level[lvl], 2; parallel) do accs
            params = draw_parameters()
            vf = vqoi(levels[lvl](params))
            vc = vqoi(levels[lvl - 1](params))
            update!(accs[1], vf - vc)
            update!(accs[2], vf)
        end
        variances_corrections[lvl - 1] = get_variance(accs[1])
        variances[lvl] = get_variance(accs[2])
    end

    return variances_corrections, variances
end

"""
    optimal_samples_per_level(variances_corrections, variances, cost_per_level, tolerance)

Compute optimal sample counts using the classical MLMC allocation formula that
minimises total work for a target mean-square-error budget.

# Arguments
- `variances_corrections`: `Var[correction]` at levels 2…L (length `L−1`).
- `variances`: `Var[QoI]` at each level (length `L`). Only `variances[1]` is
  used directly; levels 2+ draw from `variances_corrections`.
- `cost_per_level`: Cost per sample at each level (length `L`).
- `tolerance`: Target RMSE. Variance budget = `tolerance² / 2`.

# Returns
Vector of optimal sample counts (integers).
"""
function optimal_samples_per_level(
    variances_corrections::AbstractVector{<:Real},
    variances::AbstractVector{<:Real},
    cost_per_level::AbstractVector{<:Real},
    tolerance::Real,
)
    n_levels = length(variances)
    @assert length(cost_per_level) == n_levels
    @assert length(variances_corrections) == max(n_levels - 1, 0)
    @assert tolerance > 0 "Tolerance must be positive"

    V = zeros(Float64, n_levels)
    V[1] = variances[1]
    for l in 2:n_levels
        V[l] = variances_corrections[l - 1]
    end

    sum_sqrt = sum(sqrt(V[l] * cost_per_level[l]) for l in 1:n_levels)

    optimal_samples = zeros(Int, n_levels)
    for l in 1:n_levels
        raw = (2 / tolerance^2) * sqrt(V[l] / cost_per_level[l]) * sum_sqrt
        optimal_samples[l] = if raw > typemax(Int)
            typemax(Int)
        else
            ceil(Int, raw)
        end
    end

    return optimal_samples
end

"""
    mlmc_estimate(levels, qoi_functions, samples_per_level, draw_parameters;
                  parallel=false) -> Vector{Float64}

Compute MLMC estimates of `E[qⱼ]` for each quantity of interest, using
pre-specified sample counts per level. Statistics are accumulated online.

The estimator for each QoI `qⱼ` is the telescoping sum:

```math
\\hat{Q}_j = \\frac{1}{N_1}\\sum_{i=1}^{N_1} q_j\\bigl(L_1(p^{(i)})\\bigr)
    + \\sum_{l=2}^{L} \\frac{1}{N_l}\\sum_{i=1}^{N_l}
        \\Bigl[q_j\\bigl(L_l(p^{(i)})\\bigr) - q_j\\bigl(L_{l-1}(p^{(i)})\\bigr)\\Bigr]
```

For a single level this reduces to standard Monte Carlo.

# Arguments
- `levels`: `[L₁, …, Lₗ]` — model evaluators, one per fidelity level.
  Each `Lₗ(params) -> model_output`.
- `qoi_functions`: `[q₁, …, qQ]` — QoI extractors.
  Each `qⱼ(model_output) -> scalar`.
- `samples_per_level`: `[N₁, …, Nₗ]` — sample counts at each level.
- `draw_parameters`: `() -> params`.

# Keyword Arguments
- `parallel::Bool = false`: Use threading within each level.

# Returns
`Vector{Float64}` of length `Q` with estimates `[E[q₁], …, E[qQ]]`.
"""
function mlmc_estimate(
    levels::AbstractVector{<:Function},
    qoi_functions::AbstractVector{<:Function},
    samples_per_level::AbstractVector{<:Integer},
    draw_parameters::Function;
    parallel::Bool = false,
)
    n_levels = length(levels)
    n_qois = length(qoi_functions)
    @assert length(samples_per_level) == n_levels "samples_per_level length must match levels"
    @assert all(n -> n >= 1, samples_per_level) "All sample counts must be ≥ 1"

    estimates = zeros(Float64, n_qois)

    # Level 1: plain expectations
    accs = accumulate_vector_samples(samples_per_level[1], n_qois; parallel) do accs
        out = levels[1](draw_parameters())
        for j in 1:n_qois
            update!(accs[j], qoi_functions[j](out))
        end
    end
    for j in 1:n_qois
        estimates[j] = get_mean(accs[j])
    end

    # Levels 2…L: correction estimators
    for lvl in 2:n_levels
        accs = accumulate_vector_samples(samples_per_level[lvl], n_qois; parallel) do accs
            params = draw_parameters()
            out_f = levels[lvl](params)
            out_c = levels[lvl - 1](params)
            for j in 1:n_qois
                update!(accs[j], qoi_functions[j](out_f) - qoi_functions[j](out_c))
            end
        end
        for j in 1:n_qois
            estimates[j] += get_mean(accs[j])
        end
    end

    return estimates
end

"""
    mlmc_estimate_adaptive(levels, qoi_functions, draw_parameters;
                          tolerance=missing, relative_tolerance=1e-3,
                          cost_per_level=missing, variance_qoi=nothing,
                          initial_samples=100, parallel=false,
                          max_iterations=20, info_level=1)
                          -> (estimates, samples_per_level)

Adaptively compute MLMC estimates by running pilot samples, computing the
optimal allocation, and iteratively adding samples until the budget is met.

The cost model is handled as follows:
- `cost_per_level` may be either a single `missing` value or a vector of
  real-valued costs.
- If `cost_per_level === missing`, the level costs are estimated online from the
  actual wall-clock time of each level evaluation, or from an explicit
  `(output, cost)` tuple returned by the level function when available.
- If a fixed cost vector is supplied, those values are used directly.

All QoI means and the allocation variance are maintained as online accumulators,
so new batches are merged without reprocessing earlier data.

# Algorithm
1. Run `initial_samples` at every level to obtain pilot variance estimates.
2. Use [`optimal_samples_per_level`](@ref) to compute target sample counts.
3. Draw additional samples at any level where the current count is below target.
4. Re-estimate variances from all accumulated samples and repeat from step 2.
5. Converge when no level needs more samples.

# Arguments
- `levels`: Model evaluators `[L₁, …, Lₗ]`, one per fidelity level.
- `qoi_functions`: QoI extractors `[q₁, …, qQ]`, each `q(model_output) -> scalar`.
- `draw_parameters`: `() -> params`.

# Keyword Arguments
- `cost_per_level`: Either `missing` or a length-`L` vector of positive costs.
  If this is `missing`, all level costs are estimated adaptively during the run.
  If a level returns `(output, cost)`, that explicit cost is used; otherwise the
  elapsed wall-clock time for the evaluation is used.
- `tolerance`: Absolute target RMSE. If provided, this is used directly.
- `relative_tolerance`: Relative target RMSE, used only when `tolerance` is
    `missing`. The effective tolerance is computed after warmup as
    `relative_tolerance * qoi_after_warmup`, where `qoi_after_warmup` is
    `abs(Q)` for a single QoI and `norm(Q)` for multiple QoIs.
- `variance_qoi`: Scalar `v(model_output) -> scalar` for allocation.
  Default: `v(out) = Σᵢ qᵢ(out)²`.
- `initial_samples::Int = 100`: Pilot samples per level (≥ 2).
- `parallel::Bool = false`: Thread samples within each level.
- `max_iterations::Int = 20`: Safety limit on refinement iterations.
- `info_level::Int = 1`: Print MLMC progress summaries when `> 0`, including
    warmup and per-iteration statistics.

# Returns
`(estimates, samples_per_level)` where `estimates::Vector{Float64}` has length
`Q` and `samples_per_level::Vector{Int}` is the final allocation.
"""
function mlmc_estimate_adaptive(
    levels::AbstractVector{<:Function},
    qoi_functions::AbstractVector{<:Function},
    draw_parameters::Function;
    relative_tolerance::Union{Real, Missing} = 1e-2,
    tolerance::Union{Real, Missing} = missing,
    cost_per_level::Union{AbstractVector{<:Real}, Missing} = missing,
    variance_qoi::Union{Function, Nothing} = nothing,
    initial_samples::Int = 100,
    parallel::Bool = false,
    max_iterations::Int = 20,
    iteration_cost_limit::Union{Real, Missing} = 1,
    info_level::Int = 1,
)
    n_levels = length(levels)
    n_qois = length(qoi_functions)
    @assert cost_per_level === missing || length(cost_per_level) == n_levels
    @assert !ismissing(tolerance) || !ismissing(relative_tolerance) "Either tolerance or relative_tolerance must be provided"
    if !ismissing(tolerance)
        @assert tolerance > 0 "Tolerance must be positive"
    end
    if !ismissing(relative_tolerance)
        @assert relative_tolerance > 0 "relative_tolerance must be positive"
    end
    @assert initial_samples >= 2 "Need at least 2 initial samples to estimate variance"
    @assert info_level >= 0 "info_level must be non-negative"

    if ismissing(cost_per_level)
        cost_per_level = ones(Float64, n_levels)
        estimate_cost = true
    else
        estimate_cost = false
    end

    vqoi = something(variance_qoi, _default_variance_qoi(qoi_functions))

    function _eval_level(level, params)
        t0 = time_ns()
        out = level(params)
        elapsed = (time_ns() - t0) / 1e9
        if out isa Tuple && length(out) == 2
            out_value, explicit_cost = out
            if explicit_cost isa Real && isfinite(Float64(explicit_cost))
                return out_value, Float64(explicit_cost)
            end
        end
        return out, elapsed
    end

    # Per-level accumulators:
    #   indices 1:n_qois   → QoI values (level 1) or QoI corrections (levels 2+)
    #   index   n_qois+1   → variance_qoi value (level 1) or correction (levels 2+)
    k = n_qois + 1
    accs = [[OnlineAccumulator() for _ in 1:k] for _ in 1:n_levels]
    if estimate_cost
        cost_accs = [OnlineAccumulator() for _ in 1:n_levels]
    end

    # Sample body for a given level (zero-allocation inner loop)
    function _sample_body(lvl)
        if lvl == 1
            return function (thread_accs)
                params = draw_parameters()
                out, cost = _eval_level(levels[1], params)
                for j in 1:n_qois
                    update!(thread_accs[j], qoi_functions[j](out))
                end
                update!(thread_accs[k], vqoi(out))
                if estimate_cost
                    update!(cost_accs[lvl], cost)
                    cost_per_level[lvl] = get_mean(cost_accs[lvl])
                end
            end
        else
            return function (thread_accs)
                params = draw_parameters()
                out_f, cost_f = _eval_level(levels[lvl], params)
                out_c, cost_c = _eval_level(levels[lvl - 1], params)
                for j in 1:n_qois
                    update!(thread_accs[j], qoi_functions[j](out_f) - qoi_functions[j](out_c))
                end
                update!(thread_accs[k], vqoi(out_f) - vqoi(out_c))
                if estimate_cost
                    update!(cost_accs[lvl], cost_f + cost_c)
                    cost_per_level[lvl] = get_mean(cost_accs[lvl])
                end
            end
        end
    end

    # Warmup samples
    if info_level > 0
        warmup_plan = fill(initial_samples, n_levels)
        mlmc_print_iteration_start("Warmup", warmup_plan, warmup_plan)
    end
    for lvl in 1:n_levels
        info_level > 1 && @info "\tLevel $lvl: drawing $(initial_samples) samples"
        accumulate_vector_samples!(_sample_body(lvl), accs[lvl], initial_samples; parallel)
        if estimate_cost
            cost_per_level[lvl] = get_mean(cost_accs[lvl])
        end
    end
    current_counts = fill(initial_samples, n_levels)

    warmup_estimate = [sum(get_mean(accs[lvl][j]) for lvl in 1:n_levels) for j in 1:n_qois]
    qoi_after_warmup = n_qois == 1 ? abs(warmup_estimate[1]) : sqrt(sum(abs2, warmup_estimate))
    qoi_scale = max(qoi_after_warmup, eps(Float64))
    effective_tolerance = ismissing(tolerance) ? Float64(relative_tolerance) * qoi_scale : Float64(tolerance)

    if info_level > 0
        mlmc_print_iteration_summary(
            "Warmup",
            current_counts,
            Float64.(cost_per_level),
            accs,
            n_qois,
            zeros(Int, n_levels),
            zeros(Int, n_levels),
        )
    end

    # Iterative refinement
    converged = false
    completed_iterations = 0
    for iter in 1:max_iterations
        completed_iterations = iter
        variances = zeros(Float64, n_levels)
        variances_corrections = zeros(Float64, max(n_levels - 1, 0))
        variances[1] = get_variance(accs[1][k])
        for l in 2:n_levels
            variances_corrections[l - 1] = get_variance(accs[l][k])
        end

        for lvl in 1:n_levels
            if estimate_cost
                if !isfinite(cost_per_level[lvl]) || cost_per_level[lvl] <= 0
                    cost_per_level[lvl] = 1.0
                end
            end
        end

        target = optimal_samples_per_level(variances_corrections, variances, Float64.(cost_per_level), effective_tolerance)
        requested = target .- current_counts
        needed = copy(requested)

        # Cap per-level new work so each level adds at most 10% of total work so far.
        max_cost = cost_per_level[end]*iteration_cost_limit
        for lvl in 1:n_levels
            level_cost = max(Float64(cost_per_level[lvl]), eps(Float64))
            max_new = max(1, floor(Int, max_cost / level_cost))
            needed[lvl] = clamp(needed[lvl], 0, max_new)
        end

        if info_level > 0
            mlmc_print_iteration_start(iter, requested, needed)
        end

        if all(n -> n <= 0, needed)
            converged = true
            break
        end

        for lvl in 1:n_levels
            if needed[lvl] > 0
                info_level > 1 && @info "\tLevel $lvl: drawing $(needed[lvl]) samples"
                accumulate_vector_samples!(_sample_body(lvl), accs[lvl], needed[lvl]; parallel)
                current_counts[lvl] += needed[lvl]
            end
        end

        if info_level > 0
            mlmc_print_iteration_summary("Iteration $iter", current_counts, Float64.(cost_per_level), accs, n_qois, requested, needed)
        end
    end

    if info_level > 0
        mlmc_print_iteration_summary(
            "Final",
            current_counts,
            Float64.(cost_per_level),
            accs,
            n_qois,
            zeros(Int, n_levels),
            zeros(Int, n_levels),
        )
        mlmc_print_final_statistics(
            current_counts,
            Float64.(cost_per_level),
            effective_tolerance,
            qoi_after_warmup,
            completed_iterations,
            converged,
        )
    end

    # Final estimates from accumulated QoI means
    estimates = zeros(Float64, n_qois)
    for j in 1:n_qois
        estimates[j] = sum(get_mean(accs[lvl][j]) for lvl in 1:n_levels)
    end

    return estimates, current_counts
end

"""
    evaluate_monte_carlo(n_samples, draw_parameters, level, qoi_functions;
                         parallel=false) -> Vector{Float64}

Standard (single-level) Monte Carlo estimator for multiple QoIs.

# Arguments
- `n_samples`: Number of samples.
- `draw_parameters`: `() -> params`.
- `level`: Model evaluator `L(params) -> model_output`.
- `qoi_functions`: `[q₁, …, qQ]` — QoI extractors.

# Returns
`Vector{Float64}` of length `Q` with estimates `[E[q₁], …, E[qQ]]`.
"""
function evaluate_monte_carlo(
    n_samples::Integer,
    draw_parameters::Function,
    level::Function,
    qoi_functions::AbstractVector{<:Function};
    parallel::Bool = false,
)
    n_qois = length(qoi_functions)
    accs = accumulate_vector_samples(n_samples, n_qois; parallel) do accs
        out = level(draw_parameters())
        for j in 1:n_qois
            update!(accs[j], qoi_functions[j](out))
        end
    end
    return [get_mean(accs[j]) for j in 1:n_qois]
end

"""
    evaluate_monte_carlo(n_samples, draw_parameters, qoi_function;
                         parallel=false) -> Float64

Standard Monte Carlo estimator of `E[Q]` for a single scalar QoI.
"""
function evaluate_monte_carlo(
    n_samples::Integer,
    draw_parameters::Function,
    qoi_function::Function;
    parallel::Bool = false,
)
    acc = accumulate_samples(n_samples; parallel) do
        qoi_function(draw_parameters())
    end
    return get_mean(acc)
end

"""
    mlmc_estimate_vector_qoi(levels, qoi_function, samples_per_level, draw_parameters;
                              parallel=false) -> Vector{Float64}

MLMC estimator for a single vector-valued QoI function.  The function must
return a vector of the **same length** for every evaluation.  Statistics are
accumulated online per component using the standard telescoping sum.

# Arguments
- `levels`: `[L₁, …, Lₗ]` — model evaluators, one per fidelity level.
- `qoi_function`: `q(model_output) -> AbstractVector` — a single QoI that
  returns a fixed-length vector.
- `samples_per_level`: `[N₁, …, Nₗ]` — sample counts at each level.
- `draw_parameters`: `() -> params`.

# Keyword Arguments
- `parallel::Bool = false`: Use threading within each level.

# Returns
`Vector{Float64}` of the same length as `qoi_function`'s output — the
component-wise MLMC mean estimate.
"""
function mlmc_estimate_vector_qoi(
    levels::AbstractVector{<:Function},
    qoi_function::Function,
    samples_per_level::AbstractVector{<:Integer},
    draw_parameters::Function;
    parallel::Bool = false,
)
    n_levels = length(levels)
    @assert length(samples_per_level) == n_levels

    # Determine output vector length from a probe evaluation
    k = length(qoi_function(levels[1](draw_parameters())))

    estimates = zeros(Float64, k)

    # Level 1: plain expectations
    accs = accumulate_vector_samples(samples_per_level[1], k; parallel) do accs
        qval = qoi_function(levels[1](draw_parameters()))
        for j in 1:k
            update!(accs[j], qval[j])
        end
    end
    for j in 1:k
        estimates[j] = get_mean(accs[j])
    end

    # Levels 2…L: corrections
    for lvl in 2:n_levels
        accs = accumulate_vector_samples(samples_per_level[lvl], k; parallel) do accs
            params = draw_parameters()
            qf = qoi_function(levels[lvl](params))
            qc = qoi_function(levels[lvl - 1](params))
            for j in 1:k
                update!(accs[j], qf[j] - qc[j])
            end
        end
        for j in 1:k
            estimates[j] += get_mean(accs[j])
        end
    end

    return estimates
end

"""
    evaluate_monte_carlo_vector_qoi(n_samples, draw_parameters, level, qoi_function;
                                     parallel=false) -> Vector{Float64}

Standard (single-level) Monte Carlo estimator for a single vector-valued QoI.

# Arguments
- `n_samples`: Number of samples.
- `draw_parameters`: `() -> params`.
- `level`: Model evaluator `L(params) -> model_output`.
- `qoi_function`: `q(model_output) -> AbstractVector` — must return a
  fixed-length vector.

# Returns
`Vector{Float64}` — the component-wise MC mean estimate.
"""
function evaluate_monte_carlo_vector_qoi(
    n_samples::Integer,
    draw_parameters::Function,
    level::Function,
    qoi_function::Function;
    parallel::Bool = false,
)
    # Determine output vector length from a probe evaluation
    k = length(qoi_function(level(draw_parameters())))

    accs = accumulate_vector_samples(n_samples, k; parallel) do accs
        qval = qoi_function(level(draw_parameters()))
        for j in 1:k
            update!(accs[j], qval[j])
        end
    end
    return [get_mean(accs[j]) for j in 1:k]
end

#= ── Sample collection ──────────────────────────────────────────────────── =#

"""
    MLMCSamples

Container holding raw QoI samples from an MLMC run.

# Fields
- `fine::Vector{Matrix{Float64}}`: `fine[lvl]` is `(n_qois × N_lvl)` — QoI
  values evaluated at the fine level.
- `coarse::Vector{Matrix{Float64}}`: `coarse[lvl]` is `(n_qois × N_lvl)` — QoI
  values evaluated at the coarse level (`lvl-1`). For level 1 this is a matrix
  of zeros (no coarser level exists).
- `corrections::Vector{Matrix{Float64}}`: `corrections[lvl] = fine[lvl] - coarse[lvl]`.
- `n_levels::Int`
- `n_qois::Int`
"""
struct MLMCSamples
    fine::Vector{Matrix{Float64}}
    coarse::Vector{Matrix{Float64}}
    corrections::Vector{Matrix{Float64}}
    n_levels::Int
    n_qois::Int
end

function MLMCSamples(samples::MLMCSamples, q::Function)
    fine = [reshape([q(samples.fine[lvl][:, s]...) for s in 1:size(samples.fine[lvl], 2)], 1, size(samples.fine[lvl], 2)) for lvl in 1:samples.n_levels]
    coarse = [reshape([q(samples.coarse[lvl][:, s]...) for s in 1:size(samples.coarse[lvl], 2)], 1, size(samples.coarse[lvl], 2)) for lvl in 1:samples.n_levels]
    corrections = fine .- coarse
    return MLMCSamples(fine, coarse, corrections, samples.n_levels, samples.n_qois)
end

"""
    mlmc_sample(levels, qoi_functions, samples_per_level, draw_parameters;
                parallel=false, netcdf_path=nothing) -> MLMCSamples

Run an MLMC sampling campaign and return (or save) all raw QoI samples.

For each level `l` and each sample `i`, the model is evaluated at the fine
level `l` and the coarse level `l−1` (with the *same* parameters), and every
QoI extractor is applied to both outputs.

# Arguments
- `levels`, `qoi_functions`, `samples_per_level`, `draw_parameters`: same as
  [`mlmc_estimate`](@ref).

# Keyword Arguments
- `parallel::Bool = false`: thread samples within each level.
- `netcdf_path::Union{String,Nothing} = nothing`: if a file path is given,
  all samples are written to a single NetCDF-4 file *in addition* to being
  returned.

# Returns
An [`MLMCSamples`](@ref) struct containing `fine`, `coarse`, and `corrections`
matrices for every level.
"""
function mlmc_sample(
    levels::AbstractVector{<:Function},
    qoi_functions::AbstractVector{<:Function},
    samples_per_level::AbstractVector{<:Integer},
    draw_parameters::Function;
    parallel::Bool = false,
    netcdf_path::Union{String, Nothing} = nothing,
)
    n_levels = length(levels)
    n_qois   = length(qoi_functions)
    @assert length(samples_per_level) == n_levels

    fine_arrays   = Vector{Matrix{Float64}}(undef, n_levels)
    coarse_arrays = Vector{Matrix{Float64}}(undef, n_levels)

    for lvl in 1:n_levels
        ns = samples_per_level[lvl]
        F = zeros(Float64, n_qois, ns)
        C = zeros(Float64, n_qois, ns)

        if parallel
            Threads.@threads for s in 1:ns
                params = draw_parameters()
                out_f = levels[lvl](params)
                for j in 1:n_qois
                    F[j, s] = qoi_functions[j](out_f)
                end
                if lvl > 1
                    out_c = levels[lvl - 1](params)
                    for j in 1:n_qois
                        C[j, s] = qoi_functions[j](out_c)
                    end
                end
            end
        else
            for s in 1:ns
                params = draw_parameters()
                out_f = levels[lvl](params)
                for j in 1:n_qois
                    F[j, s] = qoi_functions[j](out_f)
                end
                if lvl > 1
                    out_c = levels[lvl - 1](params)
                    for j in 1:n_qois
                        C[j, s] = qoi_functions[j](out_c)
                    end
                end
            end
        end

        fine_arrays[lvl]   = F
        coarse_arrays[lvl] = C
    end

    corr_arrays = [fine_arrays[l] .- coarse_arrays[l] for l in 1:n_levels]
    result = MLMCSamples(fine_arrays, coarse_arrays, corr_arrays, n_levels, n_qois)

    if netcdf_path !== nothing
        _write_mlmc_samples_netcdf(result, netcdf_path)
    end

    return result
end

"""
    mlmc_estimate_from_samples(samples::MLMCSamples) -> Vector{Float64}

Compute the MLMC telescoping-sum estimate from pre-collected samples.
"""
function mlmc_estimate_from_samples(samples::MLMCSamples)
    estimates = zeros(Float64, samples.n_qois)
    for j in 1:samples.n_qois
        for lvl in 1:samples.n_levels
            estimates[j] += mean(samples.corrections[lvl][j, :])
        end
    end
    return estimates
end

import Statistics: mean

import PrettyTables
import NCDatasets

function _write_mlmc_samples_netcdf(samples::MLMCSamples, path::String)

    NCDatasets.NCDataset(path, "c") do ds
        ds.attrib["n_levels"] = samples.n_levels
        ds.attrib["n_qois"]   = samples.n_qois
        for lvl in 1:samples.n_levels
            ns = size(samples.fine[lvl], 2)
            sdim = "sample_l$(lvl)"
            qdim = "qoi_l$(lvl)"
            NCDatasets.defDim(ds, sdim, ns)
            NCDatasets.defDim(ds, qdim, samples.n_qois)

            vf = NCDatasets.defVar(ds, "fine_l$(lvl)", Float64, (qdim, sdim))
            vc = NCDatasets.defVar(ds, "coarse_l$(lvl)", Float64, (qdim, sdim))
            vd = NCDatasets.defVar(ds, "correction_l$(lvl)", Float64, (qdim, sdim))
            vf[:, :] = samples.fine[lvl]
            vc[:, :] = samples.coarse[lvl]
            vd[:, :] = samples.corrections[lvl]
        end
    end
end

"""
    read_mlmc_samples_netcdf(path::String) -> MLMCSamples

Read an [`MLMCSamples`](@ref) struct from a NetCDF file previously written
by [`mlmc_sample`](@ref).
"""
function read_mlmc_samples_netcdf(path::String)
    NCDatasets.NCDataset(path, "r") do ds
        n_levels = Int(ds.attrib["n_levels"])
        n_qois   = Int(ds.attrib["n_qois"])
        fine   = Vector{Matrix{Float64}}(undef, n_levels)
        coarse = Vector{Matrix{Float64}}(undef, n_levels)
        corr   = Vector{Matrix{Float64}}(undef, n_levels)
        for lvl in 1:n_levels
            fine[lvl]   = Array(ds["fine_l$(lvl)"])
            coarse[lvl] = Array(ds["coarse_l$(lvl)"])
            corr[lvl]   = Array(ds["correction_l$(lvl)"])
        end
        return MLMCSamples(fine, coarse, corr, n_levels, n_qois)
    end
end