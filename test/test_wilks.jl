using LinearAlgebra
using Random
using Distributions
using CairoMakie

# ============================================================
# Chapter 3 setup from Wilks (2017): data-generation only
# ============================================================

"""
    truth_covariance(d; rho=0.6, sigma2=1.0)

Construct the "truth" covariance matrix Σ0 with entries
cov(x_j, x_k) = sigma2 * rho^abs(j-k).

In the paper's main experiments:
- d = 3
- sigma2 = 1
- rho = 0.6
"""
function truth_covariance(d::Int; rho::Float64=0.6, sigma2::Float64=1.0)
    Σ = Matrix{Float64}(undef, d, d)
    for j in 1:d, k in 1:d
        Σ[j, k] = sigma2 * rho^abs(j-k)
    end
    return Σ
end

"""
    type1_covariance(d; sigma2_forecast, rho=0.6)

Type 1 covariance miscalibration:
correct correlation structure, wrong marginal variance.

cov_f(x_j, x_k) = sigma2_forecast * 0.6^abs(j-k)
"""
function type1_covariance(d::Int; sigma2_forecast::Float64, rho::Float64=0.6)
    return truth_covariance(d; rho=rho, sigma2=sigma2_forecast)
end

"""
    type2_covariance(d; rho_forecast)

Type 2 covariance miscalibration:
correct marginal variances, wrong correlation structure.

cov_f(x_j, x_k) = rho_forecast^abs(j-k)
"""
function type2_covariance(d::Int; rho_forecast::Float64)
    return truth_covariance(d; rho=rho_forecast, sigma2=1.0)
end

"""
    rotation_matrix_axis_angle(axis, theta)

Rodrigues rotation formula.
- axis: 3-vector
- theta: angle in radians
"""
function rotation_matrix_axis_angle(axis::AbstractVector{<:Real}, theta::Real)
    u = collect(Float64, axis)
    u ./= norm(u)
    ux, uy, uz = u

    K = [  0.0  -uz    uy;
          uz    0.0  -ux;
         -uy    ux    0.0 ]

    I3 = Matrix{Float64}(I, 3, 3)
    return I3*cos(theta) + (1 - cos(theta))*(u*u') + sin(theta)*K
end

"""
    type3_covariance(theta_deg)

Type 3 covariance miscalibration for d = 3:
rotate the truth covariance ellipsoid by angle theta around
the axis [-0.1853, -0.5494, 0.8174].

Σf = R * Σ0 * R'
"""
function type3_covariance(theta_deg::Float64)
    Σ0 = truth_covariance(3; rho=0.6, sigma2=1.0)
    axis = [-0.1853, -0.5494, 0.8174]
    θ = deg2rad(theta_deg)
    R = rotation_matrix_axis_angle(axis, θ)
    return R * Σ0 * R'
end

"""
    eigensystem_truth()

Return eigenvalues and eigenvectors of Σ0 in descending eigenvalue order.
Used for the bias cases.
"""
function eigensystem_truth()
    Σ0 = truth_covariance(3; rho=0.6, sigma2=1.0)
    F = eigen(Symmetric(Σ0))
    idx = sortperm(F.values, rev=true)
    λ = F.values[idx]
    E = F.vectors[:, idx]
    return λ, E
end

"""
    bias_mean(which_eigenvector; sign=+1)

Bias-case forecast mean:
μ = sign * 0.635 * sqrt(λ_j) * e_j

This matches the paper's Eq. (18).
"""
function bias_mean(which_eigenvector::Int; sign::Int=+1)
    λ, E = eigensystem_truth()
    j = which_eigenvector
    return sign * 0.635 * sqrt(λ[j]) * E[:, j]
end

"""
    simulate_case(n, m; μ, Σf, d=3, seed=1234)

Generate:
- observed values x0_t ~ N_d(0, Σ0), for t = 1,...,n
- simulated ensemble values x_{t,i} ~ N_d(μ, Σf), i = 1,...,m

Returns:
- obs :: Matrix{Float64} of size (n, d)
- ens :: Array{Float64,3} of size (n, m, d)

Each ens[t, i, :] is ensemble member i on case t.
"""
function simulate_case(n::Int, m::Int; μ::AbstractVector, Σf::AbstractMatrix, d::Int=3, seed::Int=1234)
    Random.seed!(seed)

    Σ0 = truth_covariance(d; rho=0.6, sigma2=1.0)

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

"""
    generate_calibrated(n, m; seed=1234)

Calibrated case:
μ = 0, Σf = Σ0
"""
function generate_calibrated(n::Int, m::Int; seed::Int=1234)
    d = 3
    μ = zeros(d)
    Σf = truth_covariance(d; rho=0.6, sigma2=1.0)
    return simulate_case(n, m; μ=μ, Σf=Σf, d=d, seed=seed)
end

"""
    generate_type1(n, m, sigma2_forecast; seed=1234)

Type 1 miscalibration:
correct correlations, wrong marginal variances
Examples from Figure 1: sigma2_forecast = 0.65 or 1.35
"""
function generate_type1(n::Int, m::Int, sigma2_forecast::Float64; seed::Int=1234)
    d = 3
    μ = zeros(d)
    Σf = type1_covariance(d; sigma2_forecast=sigma2_forecast, rho=0.6)
    return simulate_case(n, m; μ=μ, Σf=Σf, d=d, seed=seed)
end

"""
    generate_type2(n, m, rho_forecast; seed=1234)

Type 2 miscalibration:
correct marginal variances, wrong correlations
Examples from Figure 1: rho_forecast = 0.45 or 0.75
"""
function generate_type2(n::Int, m::Int, rho_forecast::Float64; seed::Int=1234)
    d = 3
    μ = zeros(d)
    Σf = type2_covariance(d; rho_forecast=rho_forecast)
    return simulate_case(n, m; μ=μ, Σf=Σf, d=d, seed=seed)
end

"""
    generate_type3(n, m, theta_deg; seed=1234)

Type 3 miscalibration:
rotated covariance ellipsoid
Example from Figure 1: theta_deg = 20.0
"""
function generate_type3(n::Int, m::Int, theta_deg::Float64; seed::Int=1234)
    d = 3
    μ = zeros(d)
    Σf = type3_covariance(theta_deg)
    return simulate_case(n, m; μ=μ, Σf=Σf, d=d, seed=seed)
end

include("wilks_utils.jl")

"""
    generate_bias_case(n, m, which_eigenvector; sign=+1, seed=1234)

Bias case:
correct covariance, shifted forecast mean along eigenvector e1/e2/e3
with μ = ±0.635 * sqrt(λ_j) * e_j
"""
function generate_bias_case(n::Int, m::Int, which_eigenvector::Int; sign::Int=+1, seed::Int=1234)
    d = 3
    μ = bias_mean(which_eigenvector; sign=sign)
    Σf = truth_covariance(d; rho=0.6, sigma2=1.0)
    return simulate_case(n, m; μ=μ, Σf=Σf, d=d, seed=seed)
end

# ============================================================
# Examples matching the chapter-3 / Figure-1 cases
# ============================================================

n = 1000
m = 1

# Calibrated
obs_cal, ens_cal = case_calibrated(n, m)

# Type 1: underforecast variance (Figure 1a)
obs_t1_low, ens_t1_low = case_type1_low_variance(n, m)

# Type 1: overforecast variance (Figure 1b)
obs_t1_high, ens_t1_high = case_type1_high_variance(n, m)

# Type 2: underforecast correlation (Figure 1c)
obs_t2_low, ens_t2_low = case_type2_low_correlation(n, m)

# Type 2: overforecast correlation (Figure 1d)
obs_t2_high, ens_t2_high = case_type2_high_correlation(n, m)

# Type 3: rotated forecast distribution (Figure 1e)
obs_t3, ens_t3 = case_type3_rotated(n, m)

# Bias in direction of e1 (positive / negative)
obs_b1p, ens_b1p = generate_bias_case(n, m, 1; sign=+1, seed=7)
obs_b1n, ens_b1n = generate_bias_case(n, m, 1; sign=-1, seed=8)

# Bias in direction of e2 (positive / negative)
obs_b2p, ens_b2p = generate_bias_case(n, m, 2; sign=+1, seed=9)
obs_b2n, ens_b2n = generate_bias_case(n, m, 2; sign=-1, seed=10)

# Bias in direction of e3 (positive / negative)
obs_b3p, ens_b3p = generate_bias_case(n, m, 3; sign=+1, seed=11)
obs_b3n, ens_b3n = generate_bias_case(n, m, 3; sign=-1, seed=12)

# Example: inspect first observed vector and first ensemble member
println("First observed vector in Type 1 low-variance case:")
println(obs_t1_low[1, :])

println("\nFirst ensemble member for first forecast occasion:")
println(ens_t1_low[1, 1, :])

println("\nForecast covariance for Type 3, θ = 20°:")
println(type3_covariance(20.0))

"""
    plot_truth_vs_ensemble(obs, ens, case_name; outdir="test/plots")

Create 2D scatter comparisons for each dimension pair:
- truth/observation vectors (one point per forecast case)
- ensemble members (all members across all cases)
"""
function plot_truth_vs_ensemble(obs::Matrix{Float64}, ens::Array{Float64,3}, case_name::AbstractString; outdir::AbstractString="test/plots")
    mkpath(outdir)

    fig = Figure(size=(1200, 400), fontsize=14)
    pairs = [(1, 2), (1, 3), (2, 3)]

    # Flatten all ensemble members so the cloud can be compared to the truth/obs cloud.
    ens_flat = reshape(permutedims(ens, (1, 3, 2)), :, size(ens, 2))
    ens_flat = reshape(ens_flat, size(obs, 1) * size(ens, 2), size(obs, 2))

    for (k, (i, j)) in enumerate(pairs)
        ax = Axis(
            fig[1, k],
            title="dims ($i, $j)",
            xlabel="x_$i",
            ylabel="x_$j",
        )

        scatter!(
            ax,
            ens_flat[:, i],
            ens_flat[:, j],
            markersize=3,
            color=(:steelblue, 0.25),
            label="ensemble",
        )

        scatter!(
            ax,
            obs[:, i],
            obs[:, j],
            markersize=5,
            color=(:orangered, 0.8),
            label="truth/obs",
        )

        axislegend(ax, position=:lt)
    end

    Label(fig[0, :], "Wilks case: $case_name", fontsize=18, font=:bold)

    outfile = joinpath(outdir, "wilks_scatter_$(replace(lowercase(case_name), ' ' => '_')).png")
    save(outfile, fig)
    return outfile
end

saved_files = String[]
push!(saved_files, plot_truth_vs_ensemble(obs_cal, ens_cal, "calibrated"))
push!(saved_files, plot_truth_vs_ensemble(obs_t1_low, ens_t1_low, "type1_low_variance"))
push!(saved_files, plot_truth_vs_ensemble(obs_t1_high, ens_t1_high, "type1_high_variance"))
push!(saved_files, plot_truth_vs_ensemble(obs_t2_low, ens_t2_low, "type2_low_correlation"))
push!(saved_files, plot_truth_vs_ensemble(obs_t2_high, ens_t2_high, "type2_high_correlation"))
push!(saved_files, plot_truth_vs_ensemble(obs_t3, ens_t3, "type3_rotated"))

println("\nSaved scatter plots:")
foreach(println, saved_files)