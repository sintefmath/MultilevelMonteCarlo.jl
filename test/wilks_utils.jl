using LinearAlgebra
using Random
using Distributions

# ============================================================
# Chapter 3 setup from Wilks (2017): data-generation only
# Shared utilities for the Wilks test / experiment scripts.
# ============================================================

"""
    truth_covariance(d; rho=0.6, sigma2=1.0)

Construct the "truth" covariance matrix Σ0 with entries
`cov(x_j, x_k) = sigma2 * rho^abs(j-k)`.
"""
function truth_covariance(d::Int; rho::Float64 = 0.6, sigma2::Float64 = 1.0)
    Σ = Matrix{Float64}(undef, d, d)
    for j in 1:d, k in 1:d
        Σ[j, k] = sigma2 * rho^abs(j - k)
    end
    return Σ
end

"""
    type1_covariance(d; sigma2_forecast, rho=0.6)

Type 1 covariance miscalibration: correct correlation, wrong variance.
"""
function type1_covariance(d::Int; sigma2_forecast::Float64, rho::Float64 = 0.6)
    return truth_covariance(d; rho = rho, sigma2 = sigma2_forecast)
end

"""
    type2_covariance(d; rho_forecast)

Type 2 covariance miscalibration: correct variance, wrong correlation.
"""
function type2_covariance(d::Int; rho_forecast::Float64)
    return truth_covariance(d; rho = rho_forecast, sigma2 = 1.0)
end

"""
    rotation_matrix_axis_angle(axis, theta)

Rodrigues rotation formula around `axis` by `theta` radians.
"""
function rotation_matrix_axis_angle(axis::AbstractVector{<:Real}, theta::Real)
    u = collect(Float64, axis)
    u ./= norm(u)
    ux, uy, uz = u

    K = [  0.0  -uz    uy;
          uz    0.0  -ux;
         -uy    ux    0.0 ]

    I3 = Matrix{Float64}(I, 3, 3)
    return I3 * cos(theta) + (1 - cos(theta)) * (u * u') + sin(theta) * K
end

"""
    type3_covariance(theta_deg)

Type 3 covariance miscalibration for d = 3:
rotate Σ0 by `theta_deg` degrees around [-0.1853, -0.5494, 0.8174].
"""
function type3_covariance(theta_deg::Float64)
    Σ0 = truth_covariance(3; rho = 0.6, sigma2 = 1.0)
    axis = [-0.1853, -0.5494, 0.8174]
    θ = deg2rad(theta_deg)
    R = rotation_matrix_axis_angle(axis, θ)
    return R * Σ0 * R'
end

"""
    eigensystem_truth()

Eigenvalues and eigenvectors of Σ0 in descending eigenvalue order.
"""
function eigensystem_truth()
    Σ0 = truth_covariance(3; rho = 0.6, sigma2 = 1.0)
    F = eigen(Symmetric(Σ0))
    idx = sortperm(F.values, rev = true)
    return F.values[idx], F.vectors[:, idx]
end

"""
    bias_mean(which_eigenvector; sign=+1)

Bias forecast mean along the `which_eigenvector`-th eigenvector of Σ0.
"""
function bias_mean(which_eigenvector::Int; sign::Int = +1)
    λ, E = eigensystem_truth()
    return sign * 0.635 * sqrt(λ[which_eigenvector]) * E[:, which_eigenvector]
end

"""
    simulate_case(n, m; μ, Σf, d=3, seed=1234)

Draw `n` observed truths from N(0, Σ0) and `n*m` ensemble members from
N(μ, Σf). Returns `(obs, ens)` with sizes `(n, d)` and `(n, m, d)`.
"""
function simulate_case(n::Int, m::Int; μ::AbstractVector, Σf::AbstractMatrix,
                       d::Int = 3, seed::Int = 1234)
    Random.seed!(seed)
    Σ0 = truth_covariance(d; rho = 0.6, sigma2 = 1.0)
    dist_obs = MvNormal(zeros(d), Symmetric(Σ0))
    dist_fc  = MvNormal(Vector{Float64}(μ), Symmetric(Matrix(Σf)))

    obs = Matrix{Float64}(undef, n, d)
    ens = Array{Float64}(undef, n, m, d)

    for t in 1:n
        obs[t, :] = rand(dist_obs)
        for i in 1:m
            ens[t, i, :] = rand(dist_fc)
        end
    end
    return obs, ens
end

# ============================================================
# Ready-made generators for the chapter-3 cases
# ============================================================

function generate_calibrated(n::Int, m::Int; seed::Int = 1234)
    d = 3
    μ = zeros(d)
    Σf = truth_covariance(d; rho = 0.6, sigma2 = 1.0)
    return simulate_case(n, m; μ = μ, Σf = Σf, d = d, seed = seed)
end

function generate_type1(n::Int, m::Int, sigma2_forecast::Float64; seed::Int = 1234)
    d = 3
    μ = zeros(d)
    Σf = type1_covariance(d; sigma2_forecast = sigma2_forecast, rho = 0.6)
    return simulate_case(n, m; μ = μ, Σf = Σf, d = d, seed = seed)
end

function generate_type2(n::Int, m::Int, rho_forecast::Float64; seed::Int = 1234)
    d = 3
    μ = zeros(d)
    Σf = type2_covariance(d; rho_forecast = rho_forecast)
    return simulate_case(n, m; μ = μ, Σf = Σf, d = d, seed = seed)
end

function generate_type3(n::Int, m::Int, theta_deg::Float64; seed::Int = 1234)
    d = 3
    μ = zeros(d)
    Σf = type3_covariance(theta_deg)
    return simulate_case(n, m; μ = μ, Σf = Σf, d = d, seed = seed)
end

function generate_bias_case(n::Int, m::Int, which_eigenvector::Int;
                            sign::Int = +1, seed::Int = 1234)
    d = 3
    μ = bias_mean(which_eigenvector; sign = sign)
    Σf = truth_covariance(d; rho = 0.6, sigma2 = 1.0)
    return simulate_case(n, m; μ = μ, Σf = Σf, d = d, seed = seed)
end

# ============================================================
# Figure-1 case wrappers
# ============================================================

"Figure 1 calibrated case wrapper."
case_calibrated(n::Int, m::Int) = generate_calibrated(n, m; seed = 1)

"Figure 1a wrapper (underforecast variance)."
case_type1_low_variance(n::Int, m::Int) = generate_type1(n, m, 0.65; seed = 2)

"Figure 1b wrapper (overforecast variance)."
case_type1_high_variance(n::Int, m::Int) = generate_type1(n, m, 1.35; seed = 3)

"Figure 1c wrapper (underforecast correlation)."
case_type2_low_correlation(n::Int, m::Int) = generate_type2(n, m, 0.45; seed = 4)

"Figure 1d wrapper (overforecast correlation)."
case_type2_high_correlation(n::Int, m::Int) = generate_type2(n, m, 0.75; seed = 5)

"Figure 1e wrapper (rotated forecast distribution)."
case_type3_rotated(n::Int, m::Int) = generate_type3(n, m, 20.0; seed = 6)
