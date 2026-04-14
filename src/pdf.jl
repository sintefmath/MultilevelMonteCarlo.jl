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

#= ── 2-D Kernel Density Estimation ─────────────────────────────────────── =#

"""
    estimate_pdf_mlmc_kernel_density_2d(samples::MLMCSamples,
        qoi_indices::NTuple{2,Int}; bandwidth=nothing) -> Function

Construct a 2-D kernel density estimate of the joint PDF for two QoIs using the
MLMC telescoping sum with a bivariate Gaussian kernel:

```math
K_{\\mathbf{h}}(\\mathbf{x}) = \\frac{1}{h_1 h_2}
  \\varphi\\!\\left(\\frac{x_1}{h_1}\\right)
  \\varphi\\!\\left(\\frac{x_2}{h_2}\\right)
```

# Arguments
- `samples`: An [`MLMCSamples`](@ref) struct.
- `qoi_indices`: Tuple `(j₁, j₂)` of the two QoI row indices.

# Keyword Arguments
- `bandwidth`: Per-dimension bandwidths `(h₁, h₂)`. Default: Silverman's rule
  per dimension using finest-level fine samples.

# Returns
A callable `f(x₁, x₂) -> Float64` evaluating the estimated joint PDF.
"""
function estimate_pdf_mlmc_kernel_density_2d(
    samples::MLMCSamples,
    qoi_indices::NTuple{2,Int};
    bandwidth::Union{Nothing, NTuple{2,<:Real}} = nothing,
)
    j1, j2 = qoi_indices
    @assert 1 <= j1 <= samples.n_qois && 1 <= j2 <= samples.n_qois

    h = Vector{Float64}(undef, 2)
    for (k, j) in enumerate((j1, j2))
        if bandwidth === nothing
            finest = samples.fine[end][j, :]
            σ_val = std(finest)
            n = length(finest)
            h[k] = 1.06 * σ_val * n^(-1/5)
        else
            h[k] = Float64(bandwidth[k])
        end
    end

    # Pre-extract data
    fine1   = [copy(samples.fine[lvl][j1, :])   for lvl in 1:samples.n_levels]
    fine2   = [copy(samples.fine[lvl][j2, :])   for lvl in 1:samples.n_levels]
    coarse1 = [copy(samples.coarse[lvl][j1, :]) for lvl in 1:samples.n_levels]
    coarse2 = [copy(samples.coarse[lvl][j2, :]) for lvl in 1:samples.n_levels]

    ndist = Normal(0.0, 1.0)

    function pdf_estimate(x1::Real, x2::Real)
        # Level 1
        N1 = length(fine1[1])
        val = 0.0
        for i in 1:N1
            val += dist_pdf(ndist, (x1 - fine1[1][i]) / h[1]) *
                   dist_pdf(ndist, (x2 - fine2[1][i]) / h[2])
        end
        val /= (N1 * h[1] * h[2])

        # Levels 2…L: corrections
        for lvl in 2:samples.n_levels
            Nl = length(fine1[lvl])
            corr = 0.0
            for i in 1:Nl
                pf = dist_pdf(ndist, (x1 - fine1[lvl][i]) / h[1]) *
                     dist_pdf(ndist, (x2 - fine2[lvl][i]) / h[2])
                pc = dist_pdf(ndist, (x1 - coarse1[lvl][i]) / h[1]) *
                     dist_pdf(ndist, (x2 - coarse2[lvl][i]) / h[2])
                corr += pf - pc
            end
            val += corr / (Nl * h[1] * h[2])
        end
        return val
    end

    return pdf_estimate
end

"""
    estimate_cdf_mlmc_kernel_density_2d(samples::MLMCSamples,
        qoi_indices::NTuple{2,Int}; bandwidth=nothing) -> Function

Construct a 2-D kernel CDF estimate for two QoIs using
product Gaussian CDF kernels and the MLMC telescoping sum:

```math
\\hat{F}(x_1, x_2) = \\frac{1}{N_1} \\sum_{i=1}^{N_1}
  \\Phi\\!\\left(\\frac{x_1 - Q_{1,1}^{(i)}}{h_1}\\right)
  \\Phi\\!\\left(\\frac{x_2 - Q_{1,2}^{(i)}}{h_2}\\right)
  + \\sum_{l=2}^{L} \\frac{1}{N_l}\\sum_{i=1}^{N_l}\\bigl[\\cdots\\bigr]
```

# Returns
A callable `F(x₁, x₂) -> Float64` evaluating the estimated joint CDF.
"""
function estimate_cdf_mlmc_kernel_density_2d(
    samples::MLMCSamples,
    qoi_indices::NTuple{2,Int};
    bandwidth::Union{Nothing, NTuple{2,<:Real}} = nothing,
)
    j1, j2 = qoi_indices
    @assert 1 <= j1 <= samples.n_qois && 1 <= j2 <= samples.n_qois

    h = Vector{Float64}(undef, 2)
    for (k, j) in enumerate((j1, j2))
        if bandwidth === nothing
            finest = samples.fine[end][j, :]
            σ_val = std(finest)
            n = length(finest)
            h[k] = 1.06 * σ_val * n^(-1/5)
        else
            h[k] = Float64(bandwidth[k])
        end
    end

    fine1   = [copy(samples.fine[lvl][j1, :])   for lvl in 1:samples.n_levels]
    fine2   = [copy(samples.fine[lvl][j2, :])   for lvl in 1:samples.n_levels]
    coarse1 = [copy(samples.coarse[lvl][j1, :]) for lvl in 1:samples.n_levels]
    coarse2 = [copy(samples.coarse[lvl][j2, :]) for lvl in 1:samples.n_levels]

    ndist = Normal(0.0, 1.0)

    function cdf_estimate(x1::Real, x2::Real)
        N1 = length(fine1[1])
        val = 0.0
        for i in 1:N1
            val += dist_cdf(ndist, (x1 - fine1[1][i]) / h[1]) *
                   dist_cdf(ndist, (x2 - fine2[1][i]) / h[2])
        end
        val /= N1

        for lvl in 2:samples.n_levels
            Nl = length(fine1[lvl])
            corr = 0.0
            for i in 1:Nl
                pf = dist_cdf(ndist, (x1 - fine1[lvl][i]) / h[1]) *
                     dist_cdf(ndist, (x2 - fine2[lvl][i]) / h[2])
                pc = dist_cdf(ndist, (x1 - coarse1[lvl][i]) / h[1]) *
                     dist_cdf(ndist, (x2 - coarse2[lvl][i]) / h[2])
                corr += pf - pc
            end
            val += corr / Nl
        end
        return val
    end

    return cdf_estimate
end

#= ── 2-D Maximum Entropy PDF/CDF ───────────────────────────────────────── =#

function _tensor_legendre_2d(x, y, R::Int)
    Px = _legendre_polynomials(x, R)
    Py = _legendre_polynomials(y, R)
    return Px * Py'     # (R+1) × (R+1) outer product
end

"""
    estimate_pdf_maxent_2d(samples::MLMCSamples, qoi_indices::NTuple{2,Int};
        R::Int=4, maxiter::Int=200, tol::Float64=1e-10,
        support=nothing) -> (pdf_func, Λ, a, b)

Estimate the 2-D joint PDF via Maximum Entropy using tensor-product Legendre
moments estimated from MLMC samples.

The PDF is:

```math
f(x,y) = \\frac{4}{(b_1-a_1)(b_2-a_2)}
  \\exp\\!\\left(\\sum_{k=0}^{R}\\sum_{l=0}^{R}
  \\Lambda_{kl}\\, P_k(\\xi_1)\\, P_l(\\xi_2)\\right)
```

where ``\\xi_i = 2(x_i - a_i)/(b_i - a_i) - 1`` maps to ``[-1,1]``.

# Arguments
- `samples`: [`MLMCSamples`](@ref).
- `qoi_indices`: Tuple `(j₁, j₂)`.

# Keyword Arguments
- `R::Int = 4`: Polynomial degree per dimension. Total parameters = `(R+1)²`.
- `maxiter`, `tol`: Newton iteration controls.
- `support`: `((a₁,b₁), (a₂,b₂))`. Default: sample range + 10% padding.

# Returns
`(pdf_func, Λ, (a₁,b₁), (a₂,b₂))` where `pdf_func(x₁,x₂)` evaluates the PDF.
"""
function estimate_pdf_maxent_2d(
    samples::MLMCSamples,
    qoi_indices::NTuple{2,Int};
    R::Int = 4,
    maxiter::Int = 200,
    tol::Float64 = 1e-10,
    support::Union{Nothing, NTuple{2, Tuple{Real,Real}}} = nothing,
)
    j1, j2 = qoi_indices
    @assert 1 <= j1 <= samples.n_qois && 1 <= j2 <= samples.n_qois

    # Determine support per dimension
    bounds = Vector{Tuple{Float64,Float64}}(undef, 2)
    for (dim, j) in enumerate((j1, j2))
        all_vals = Float64[]
        for lvl in 1:samples.n_levels
            append!(all_vals, @view samples.fine[lvl][j, :])
        end
        if support === nothing
            lo, hi = extrema(all_vals)
            pad = 0.1 * (hi - lo)
            if pad == 0.0; pad = 1.0; end
            bounds[dim] = (lo - pad, hi + pad)
        else
            bounds[dim] = Float64.(support[dim])
        end
    end
    (a1, b1), (a2, b2) = bounds

    to_ref1(u) = 2.0 * (u - a1) / (b1 - a1) - 1.0
    to_ref2(u) = 2.0 * (u - a2) / (b2 - a2) - 1.0

    M = R + 1   # basis size per dimension
    N_params = M * M

    # --- MLMC moments: α[k,l] = E[P_k(ξ₁) P_l(ξ₂)] ---
    # Stored as a flattened vector in column-major order: kron(Py, Px)
    α_vec = zeros(Float64, N_params)
    for lvl in 1:samples.n_levels
        Nl = size(samples.fine[lvl], 2)
        for i in 1:Nl
            f1 = to_ref1(samples.fine[lvl][j1, i])
            f2 = to_ref2(samples.fine[lvl][j2, i])
            Pxf = _legendre_polynomials(f1, R)
            Pyf = _legendre_polynomials(f2, R)
            Tf = kron(Pyf, Pxf)
            if lvl == 1
                α_vec .+= Tf ./ Nl
            else
                c1 = to_ref1(samples.coarse[lvl][j1, i])
                c2 = to_ref2(samples.coarse[lvl][j2, i])
                Pxc = _legendre_polynomials(c1, R)
                Pyc = _legendre_polynomials(c2, R)
                Tc = kron(Pyc, Pxc)
                α_vec .+= (Tf .- Tc) ./ Nl
            end
        end
    end

    # --- Newton solver ---
    # Compute ALL moments in a single 2D quadrature pass (vector-valued integrand)
    function _moments_from_lambda(λ_vec)
        any(isnan, λ_vec) && return fill(NaN, N_params)
        μ_vec, _ = quadgk(-1.0, 1.0; rtol=1e-8) do y
            Py = _legendre_polynomials(y, R)
            inner, _ = quadgk(-1.0, 1.0; rtol=1e-8) do x
                Px = _legendre_polynomials(x, R)
                T_vec = kron(Py, Px)   # vec(Px * Py') in column-major order
                logval = dot(T_vec, λ_vec)
                exp_val = isfinite(logval) ? exp(clamp(logval, -50.0, 50.0)) : 0.0
                return T_vec .* exp_val
            end
            return inner
        end
        return μ_vec
    end

    function _residual(λ_vec)
        return _moments_from_lambda(λ_vec) .- α_vec
    end

    # Initial guess: uniform on [-1,1]² → density = 1/4
    λ_vec = zeros(Float64, N_params)
    # P_0(x)P_0(y) = 1/√2 · 1/√2 = 1/2, so exp(λ₀₀ · 1/2) = 1/4 → λ₀₀ = 2ln(1/4)
    P00 = _legendre_polynomials(0.0, R)[1]^2
    λ_vec[1] = log(0.25) / P00

    best_λ = copy(λ_vec)
    best_res = Inf

    for iter in 1:maxiter
        res = try
            _residual(λ_vec)
        catch
            λ_vec .= best_λ
            break
        end
        any(isnan, res) && (λ_vec .= best_λ; break)
        res_norm = norm(res)

        if res_norm < best_res
            best_res = res_norm
            best_λ .= λ_vec
        end

        if res_norm < tol
            break
        end

        J = try
            ForwardDiff.jacobian(_residual, λ_vec)
        catch
            λ_vec .= best_λ
            break
        end
        try
            δ = J \ res
            # Limit step size to prevent divergence
            δ_norm = norm(δ)
            if δ_norm > 10.0
                δ .*= 10.0 / δ_norm
            end
            λ_vec .-= δ
        catch
            λ_vec .= best_λ
            break
        end
    end

    λ_vec .= best_λ
    Λ = reshape(λ_vec, M, M)

    scale = 4.0 / ((b1 - a1) * (b2 - a2))

    function pdf_func(x1::Real, x2::Real)
        ξ1 = to_ref1(x1)
        ξ2 = to_ref2(x2)
        if ξ1 < -1.0 || ξ1 > 1.0 || ξ2 < -1.0 || ξ2 > 1.0
            return 0.0
        end
        Px = _legendre_polynomials(ξ1, R)
        Py = _legendre_polynomials(ξ2, R)
        T_vec = kron(Py, Px)
        return scale * exp(clamp(dot(T_vec, λ_vec), -500.0, 500.0))
    end

    return pdf_func, Λ, (a1, b1), (a2, b2)
end

"""
    estimate_cdf_maxent_2d(samples::MLMCSamples, qoi_indices::NTuple{2,Int};
        kwargs...) -> (cdf_func, Λ, (a₁,b₁), (a₂,b₂))

Estimate the 2-D joint CDF via the Maximum Entropy method. Computes the MaxEnt
PDF via [`estimate_pdf_maxent_2d`](@ref) and integrates it numerically.

# Returns
`(cdf_func, Λ, bounds₁, bounds₂)` where `cdf_func(x₁,x₂)` evaluates the CDF.
"""
function estimate_cdf_maxent_2d(samples::MLMCSamples, qoi_indices::NTuple{2,Int}; kwargs...)
    pdf_func, Λ, bounds1, bounds2 = estimate_pdf_maxent_2d(samples, qoi_indices; kwargs...)
    a1, b1 = bounds1
    a2, b2 = bounds2

    function cdf_func(x1::Real, x2::Real)
        if x1 <= a1 || x2 <= a2
            return 0.0
        end
        u1 = min(x1, b1)
        u2 = min(x2, b2)
        val, _ = quadgk(a2, u2; rtol=1e-6) do t2
            inner, _ = quadgk(a1, u1; rtol=1e-6) do t1
                pdf_func(t1, t2)
            end
            inner
        end
        return clamp(val, 0.0, 1.0)
    end

    return cdf_func, Λ, bounds1, bounds2
end
