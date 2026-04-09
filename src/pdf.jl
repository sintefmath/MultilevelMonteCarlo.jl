using Distributions: Normal, pdf as dist_pdf, cdf as dist_cdf
using ForwardDiff
using QuadGK
using LinearAlgebra

#= ── Kernel Density Estimation ──────────────────────────────────────────── =#

"""
    estimate_pdf_mlmc_kernel_density(samples::MLMCSamples, qoi_index::Int;
                                     bandwidth=nothing) -> Function

Construct a kernel density estimate of the PDF for QoI `qoi_index` from MLMC
samples, using the multilevel telescoping-sum structure.

The estimator is:

```math
\\hat{f}(u) = \\frac{1}{N_1} \\sum_{i=1}^{N_1} K_h(u - Q_1^{(i)})
  + \\sum_{l=2}^{L} \\frac{1}{N_l} \\sum_{i=1}^{N_l}
    \\bigl[K_h(u - Q_l^{(i)}) - K_h(u - Q_{l-1}^{(i)})\\bigr]
```

where ``K_h(x) = \\frac{1}{h} \\varphi(x/h)`` is a Gaussian kernel with
bandwidth ``h``.

# Arguments
- `samples`: An [`MLMCSamples`](@ref) struct.
- `qoi_index`: Which QoI (column) to estimate the PDF for.

# Keyword Arguments
- `bandwidth`: Kernel bandwidth. If `nothing`, Silverman's rule of thumb is
  used based on the finest-level fine samples.

# Returns
A callable `f(u) -> Float64` that evaluates the estimated PDF at point `u`.
"""
function estimate_pdf_mlmc_kernel_density(samples::MLMCSamples, qoi_index::Int;
                                          bandwidth::Union{Nothing, Real} = nothing)
    j = qoi_index
    @assert 1 <= j <= samples.n_qois

    # Determine bandwidth via Silverman's rule on finest-level fine samples
    if bandwidth === nothing
        finest_fine = samples.fine[end][j, :]
        σ = std(finest_fine)
        n = length(finest_fine)
        bandwidth = 1.06 * σ * n^(-1/5)
    end
    h = Float64(bandwidth)

    K_h(x) = (1.0 / h) * dist_pdf(Normal(0.0, 1.0), x / h)

    function pdf_estimate(u::Real)
        # Level 1: plain MC density
        fine1 = @view samples.fine[1][j, :]
        val = sum(K_h(u - q) for q in fine1) / length(fine1)

        # Levels 2…L: correction
        for lvl in 2:samples.n_levels
            fine_l   = @view samples.fine[lvl][j, :]
            coarse_l = @view samples.coarse[lvl][j, :]
            ns = length(fine_l)
            corr = sum(K_h(u - fine_l[s]) - K_h(u - coarse_l[s]) for s in 1:ns) / ns
            val += corr
        end
        return val
    end

    return pdf_estimate
end

"""
    estimate_cdf_mlmc_kernel_density(samples::MLMCSamples, qoi_index::Int;
                                     bandwidth=nothing) -> Function

Construct a kernel CDF estimate for QoI `qoi_index` from MLMC samples.

Uses the same MLMC telescoping sum as the PDF estimator, but with the Gaussian
CDF kernel ``\\Phi((u - x)/h)`` instead of the PDF kernel.

# Returns
A callable `F(u) -> Float64` that evaluates the estimated CDF at point `u`.
"""
function estimate_cdf_mlmc_kernel_density(samples::MLMCSamples, qoi_index::Int;
                                          bandwidth::Union{Nothing, Real} = nothing)
    j = qoi_index
    @assert 1 <= j <= samples.n_qois

    if bandwidth === nothing
        finest_fine = samples.fine[end][j, :]
        σ = std(finest_fine)
        n = length(finest_fine)
        bandwidth = 1.06 * σ * n^(-1/5)
    end
    h = Float64(bandwidth)

    Φ_h(x) = dist_cdf(Normal(0.0, 1.0), x / h)

    function cdf_estimate(u::Real)
        fine1 = @view samples.fine[1][j, :]
        val = sum(Φ_h(u - q) for q in fine1) / length(fine1)

        for lvl in 2:samples.n_levels
            fine_l   = @view samples.fine[lvl][j, :]
            coarse_l = @view samples.coarse[lvl][j, :]
            ns = length(fine_l)
            corr = sum(Φ_h(u - fine_l[s]) - Φ_h(u - coarse_l[s]) for s in 1:ns) / ns
            val += corr
        end
        return val
    end

    return cdf_estimate
end

#= ── Maximum Entropy PDF estimation ─────────────────────────────────────── =#

"""
    _legendre_polynomials(x, R)

Evaluate orthonormal Legendre polynomials ``P_0(x), …, P_R(x)`` on ``[-1,1]``
via the three-term recurrence.  Returns a vector of length `R+1`.
"""
function _legendre_polynomials(x::Real, R::Int)
    P = zeros(typeof(float(x)), R + 1)
    # P_0 (orthonormal on [-1,1]: ∫ P_0² dx = 1  ⟹  P_0 = 1/√2)
    P[1] = 1.0 / sqrt(2.0)
    if R == 0
        return P
    end
    # P_1 = √(3/2) x
    P[2] = sqrt(3.0 / 2.0) * x
    for k in 2:R
        # Recurrence for orthonormal Legendre:
        #   √((2k+1)/(k+1)) * x * P_k  -  √(k(2k+1)) / √((k+1)(2k-1)) * P_{k-1}
        #   = √((k+1)/(2k+1))^{-1} * P_{k+1}   ... simpler:
        #   (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}   (standard Legendre)
        # For orthonormal: P_k = √((2k+1)/2) * L_k  where L_k is standard Legendre
        # Use recurrence directly on orthonormal:
        a = sqrt((2k + 1.0) / (2k - 1.0)) * (2k - 1.0) / k
        b = sqrt((2k + 1.0) / (2k - 3.0)) * (k - 1.0) / k
        P[k + 1] = a * x * P[k] - b * P[k - 1]
    end
    return P
end

"""
    estimate_pdf_maxent(samples::MLMCSamples, qoi_index::Int;
                        R::Int=10, maxiter::Int=100, tol::Float64=1e-12,
                        support=nothing) -> (pdf_func, λ, a, b)

Estimate the PDF via the Maximum Entropy method using moments estimated from
MLMC samples.

The PDF is approximated as:

```math
\\tilde{f}_U(u) = \\exp\\!\\left(\\sum_{k=0}^{R} \\lambda_k \\phi_k\\!\\left(\\frac{2(u-a)}{b-a} - 1\\right)\\right)
```

where ``\\phi_k`` are orthonormal Legendre polynomials on ``[-1,1]``, and
``\\lambda_k`` are determined by matching generalized moments via Newton's
method. The Jacobian is computed using `ForwardDiff.jl`.

# Arguments
- `samples`: [`MLMCSamples`](@ref) struct.
- `qoi_index`: Which QoI to estimate the PDF for.

# Keyword Arguments
- `R::Int = 10`: Number of polynomial moments (polynomial degree).
- `maxiter::Int = 100`: Maximum Newton iterations.
- `tol::Float64 = 1e-12`: Newton convergence tolerance (‖residual‖₂).
- `support`: Tuple `(a, b)` for the support interval. If `nothing`, estimated
  from the sample range with 10% padding.

# Returns
A tuple `(pdf_func, λ, a, b)` where:
- `pdf_func(u)` evaluates the MaxEnt PDF at `u` (in original coordinates).
- `λ` is the coefficient vector of length `R+1`.
- `a, b` are the support bounds.
"""
function estimate_pdf_maxent(samples::MLMCSamples, qoi_index::Int;
                             R::Int = 10, maxiter::Int = 100, tol::Float64 = 1e-12,
                             support::Union{Nothing, Tuple{Real,Real}} = nothing)
    j = qoi_index
    @assert 1 <= j <= samples.n_qois

    # --- Estimate MLMC moments of Legendre polynomials ---
    # Collect all fine/coarse QoI values to determine support
    all_vals = Float64[]
    for lvl in 1:samples.n_levels
        append!(all_vals, @view samples.fine[lvl][j, :])
    end

    if support === nothing
        lo, hi = extrema(all_vals)
        pad = 0.1 * (hi - lo)
        if pad == 0.0; pad = 1.0; end
        a, b = lo - pad, hi + pad
    else
        a, b = Float64.(support)
    end

    # Map original → [-1,1]
    to_ref(u) = 2.0 * (u - a) / (b - a) - 1.0

    # MLMC estimate of generalized moments α_k = E[P_k(to_ref(U))]
    # Using the telescoping sum on the stored samples
    α = zeros(Float64, R + 1)
    for k in 0:R
        # Level 1
        fine1 = @view samples.fine[1][j, :]
        α[k + 1] = sum(_legendre_polynomials(to_ref(v), R)[k + 1] for v in fine1) / length(fine1)

        # Levels 2…L corrections
        for lvl in 2:samples.n_levels
            fine_l   = @view samples.fine[lvl][j, :]
            coarse_l = @view samples.coarse[lvl][j, :]
            ns = length(fine_l)
            corr = sum(
                _legendre_polynomials(to_ref(fine_l[s]), R)[k + 1] -
                _legendre_polynomials(to_ref(coarse_l[s]), R)[k + 1]
                for s in 1:ns
            ) / ns
            α[k + 1] += corr
        end
    end

    # --- Newton solver for λ ---
    # We seek λ such that ∫ P_k(x) exp(Σ λ_j P_j(x)) dx = α_k for k=0..R
    # on [-1,1].  The PDF in original coords is (2/(b-a)) * exp(Σ λ_k P_k(t(u))).

    function _moments_from_lambda(λ)
        # Compute ∫ P_k(x) exp(Σ λ_j P_j(x)) dx for k=0..R on [-1,1]
        μ = zeros(eltype(λ), R + 1)
        for k in 0:R
            val, _ = quadgk(-1.0, 1.0; rtol=1e-10) do x
                P = _legendre_polynomials(x, R)
                P[k + 1] * exp(dot(λ, P))
            end
            μ[k + 1] = val
        end
        return μ
    end

    function _residual(λ)
        return _moments_from_lambda(λ) .- α
    end

    # Initial guess: uniform distribution on [-1,1] → ρ = 1/2 → λ_0 = ln(1/2), rest 0
    λ = zeros(Float64, R + 1)
    λ[1] = log(0.5) / _legendre_polynomials(0.0, R)[1]  # so that exp(λ_0 P_0) ≈ 1/2

    best_λ = copy(λ)
    best_res = Inf

    for iter in 1:maxiter
        res = _residual(λ)
        res_norm = norm(res)

        if res_norm < best_res
            best_res = res_norm
            best_λ .= λ
        end

        if res_norm < tol
            break
        end

        J = ForwardDiff.jacobian(_residual, λ)
        try
            δ = J \ res
            λ .-= δ
        catch
            # Jacobian singular — use best so far
            λ .= best_λ
            break
        end
    end

    # Use best λ found
    λ .= best_λ

    # Build PDF function in original coordinates
    function pdf_func(u::Real)
        x = to_ref(u)
        if x < -1.0 || x > 1.0
            return 0.0
        end
        P = _legendre_polynomials(x, R)
        return (2.0 / (b - a)) * exp(dot(λ, P))
    end

    return pdf_func, λ, a, b
end

"""
    estimate_cdf_maxent(samples::MLMCSamples, qoi_index::Int; kwargs...)
                        -> (cdf_func, λ, a, b)

Estimate the CDF via the Maximum Entropy method. First computes the MaxEnt PDF
via [`estimate_pdf_maxent`](@ref), then integrates it numerically.

All keyword arguments are forwarded to [`estimate_pdf_maxent`](@ref).

# Returns
A tuple `(cdf_func, λ, a, b)` where `cdf_func(u)` evaluates the CDF at `u`.
"""
function estimate_cdf_maxent(samples::MLMCSamples, qoi_index::Int; kwargs...)
    pdf_func, λ, a, b = estimate_pdf_maxent(samples, qoi_index; kwargs...)

    function cdf_func(u::Real)
        if u <= a
            return 0.0
        elseif u >= b
            return 1.0
        end
        val, _ = quadgk(pdf_func, a, u; rtol=1e-8)
        return clamp(val, 0.0, 1.0)
    end

    return cdf_func, λ, a, b
end
