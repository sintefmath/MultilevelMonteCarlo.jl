using MultilevelMonteCarlo
using Statistics
using Random

Random.seed!(77)

# --------------------------------------------------------------------------- #
# Test problem: 2D Gaussian with noisy levels
# (X₁, X₂) ~ N([2, -1], diag(0.25, 0.16))  (independent components)
# --------------------------------------------------------------------------- #
μ = [2.0, -1.0]
σ = [0.5, 0.4]

draw_parameters() = μ .+ σ .* randn(2)

levels = Function[
    params -> params + 0.3 * randn(2),    # coarse — added noise
    params -> params + 0.03 * randn(2),   # medium — small noise
    params -> Float64.(params),            # fine   — exact
]

qoi_functions = Function[x -> x[1], x -> x[2]]

# --- Test 1: multivariate KDE CDF ---
println("=== Test 1: multivariate KDE CDF ===")
samples = mlmc_sample(levels, qoi_functions, [4000, 2000, 500], draw_parameters)

F̂ = estimate_cdf_multivariate_mlmc_kernel_density(samples, [1, 2])

# CDF at the mean should be ≈ 0.25 for 2D independent (0.5 × 0.5)
cdf_at_mean = F̂(μ)
println("  CDF at mean = $(round(cdf_at_mean, digits=3))  (expected ≈ 0.25)")
@assert 0.10 < cdf_at_mean < 0.45 "Multivariate CDF at mean out of range"

# CDF far in the upper right should be ≈ 1.0
cdf_far_right = F̂(μ .+ 5 .* σ)
println("  CDF at μ+5σ = $(round(cdf_far_right, digits=3))  (expected ≈ 1.0)")
@assert cdf_far_right > 0.9 "Multivariate CDF at far right too small"

# CDF far in the lower left should be ≈ 0.0
cdf_far_left = F̂(μ .- 5 .* σ)
println("  CDF at μ-5σ = $(round(cdf_far_left, digits=3))  (expected ≈ 0.0)")
@assert cdf_far_left < 0.1 "Multivariate CDF at far left too large"

# --- Test 2: bootstrap resampling ---
println("\n=== Test 2: bootstrap resampling ===")
resampled = ml_bootstrap_resample_multivariate(samples, [1, 2], 5000)

resample_mean1 = mean(resampled[1, :])
resample_mean2 = mean(resampled[2, :])
resample_std1  = std(resampled[1, :])
resample_std2  = std(resampled[2, :])
println("  Dim 1: mean=$(round(resample_mean1, digits=3)), std=$(round(resample_std1, digits=3))  (expected ≈ $(μ[1]), $(σ[1]))")
println("  Dim 2: mean=$(round(resample_mean2, digits=3)), std=$(round(resample_std2, digits=3))  (expected ≈ $(μ[2]), $(σ[2]))")
@assert abs(resample_mean1 - μ[1]) < 0.3 "Bootstrap dim 1 mean off"
@assert abs(resample_mean2 - μ[2]) < 0.3 "Bootstrap dim 2 mean off"
@assert abs(resample_std1 - σ[1]) < 0.3 "Bootstrap dim 1 std off"
@assert abs(resample_std2 - σ[2]) < 0.3 "Bootstrap dim 2 std off"

# --- Test 3: multivariate rank histogram (wrapper) ---
println("\n=== Test 3: multivariate rank histogram (wrapper) ===")
n_rank = 80
n_resamples = 200
samples_per_level = [800, 400, 100]

pit_values = multivariate_rank_histogram(levels, qoi_functions, draw_parameters,
                                         n_rank, samples_per_level;
                                         number_of_resamples=n_resamples)

println("  PIT range = [$(round(minimum(pit_values), digits=3)), $(round(maximum(pit_values), digits=3))]")
@assert all(0.0 .<= pit_values .<= 1.0) "MRH PIT values out of [0,1]"

pit_mean = mean(pit_values)
println("  PIT mean = $(round(pit_mean, digits=3))  (expected ≈ 0.5)")
@assert abs(pit_mean - 0.5) < 0.2 "MRH PIT mean too far from 0.5"

# --- Test 3b: multivariate rank histogram (observations-based) ---
println("\n=== Test 3b: multivariate rank histogram (observations) ===")
obs_mrh = hcat([μ .+ σ .* randn(2) for _ in 1:n_rank]...)
pit_values_obs = multivariate_rank_histogram(obs_mrh, levels, qoi_functions,
                                             samples_per_level, draw_parameters;
                                             number_of_resamples=n_resamples)

println("  PIT range = [$(round(minimum(pit_values_obs), digits=3)), $(round(maximum(pit_values_obs), digits=3))]")
@assert all(0.0 .<= pit_values_obs .<= 1.0) "MRH PIT values (obs) out of [0,1]"

pit_mean_obs = mean(pit_values_obs)
println("  PIT mean = $(round(pit_mean_obs, digits=3))  (expected ≈ 0.5)")
@assert abs(pit_mean_obs - 0.5) < 0.2 "MRH PIT mean (obs) too far from 0.5"

println("\n✓ All multivariate rank histogram tests passed.")
