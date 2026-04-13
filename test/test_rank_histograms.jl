using MultilevelMonteCarlo
using Statistics
using Random

Random.seed!(123)

# --------------------------------------------------------------------------- #
# Test problem: X ~ N(2, 0.5²) with noisy levels
# --------------------------------------------------------------------------- #
μ_true = 2.0
σ_true = 0.5
levels = Function[
    params -> params + 0.2 * randn(),    # coarse
    params -> params + 0.02 * randn(),   # medium
    params -> Float64(params),           # exact
]
qoi_function = identity
draw_parameters() = μ_true + σ_true * randn()

# --- Test 1: ml_gregory_resample ---
println("=== Test 1: ml_gregory_resample ===")
samples_big = mlmc_sample(levels, Function[qoi_function], [4000, 2000, 500],
                          draw_parameters)
resampled = ml_gregory_resample(samples_big, 1, 5000)

resample_mean = mean(resampled)
resample_std  = std(resampled)
println("  Resample mean = $(round(resample_mean, digits=3))  (expected ≈ $μ_true)")
println("  Resample std  = $(round(resample_std, digits=3))   (expected ≈ $σ_true)")
@assert abs(resample_mean - μ_true) < 0.3 "Gregory resample mean too far off"
@assert abs(resample_std - σ_true) < 0.3 "Gregory resample std too far off"

# --- Test 2: rank_histogram_gregory (wrapper) ---
println("\n=== Test 2: rank_histogram_gregory (wrapper) ===")
n_rank = 100
n_resamples = 200
samples_per_level = [400, 200, 50]

ranks = rank_histogram_gregory(levels, qoi_function, draw_parameters,
                               n_rank, samples_per_level;
                               number_of_resamples=n_resamples)

println("  min rank = $(minimum(ranks)), max rank = $(maximum(ranks))")
@assert all(1 .<= ranks .<= n_resamples + 1) "Ranks out of valid range"
mean_rank = mean(ranks)
expected_mean = (n_resamples + 1) / 2
println("  mean rank = $(round(mean_rank, digits=1))  (expected ≈ $expected_mean)")
@assert abs(mean_rank - expected_mean) / expected_mean < 0.3 "Mean rank too far from expected"

# --- Test 2b: rank_histogram_gregory (observations-based) ---
println("\n=== Test 2b: rank_histogram_gregory (observations) ===")
observations_greg = [μ_true + σ_true * randn() for _ in 1:n_rank]
ranks_obs = rank_histogram_gregory(observations_greg, levels, Function[qoi_function],
                                   samples_per_level, draw_parameters;
                                   number_of_resamples=n_resamples)

println("  min rank = $(minimum(ranks_obs)), max rank = $(maximum(ranks_obs))")
@assert all(1 .<= ranks_obs .<= n_resamples + 1) "Ranks out of valid range"
mean_rank_obs = mean(ranks_obs)
println("  mean rank = $(round(mean_rank_obs, digits=1))  (expected ≈ $expected_mean)")
@assert abs(mean_rank_obs - expected_mean) / expected_mean < 0.3 "Mean rank too far from expected"

# --- Test 3: rank_histogram_cdf with KDE (wrapper) ---
println("\n=== Test 3: rank_histogram_cdf (KDE, wrapper) ===")
pit_kde = rank_histogram_cdf(levels, qoi_function, draw_parameters,
                             n_rank, samples_per_level,
                             estimate_cdf_mlmc_kernel_density)

println("  PIT range = [$(round(minimum(pit_kde), digits=3)), $(round(maximum(pit_kde), digits=3))]")
@assert all(0.0 .<= pit_kde .<= 1.0) "KDE PIT values out of [0,1]"
pit_kde_mean = mean(pit_kde)
println("  PIT mean = $(round(pit_kde_mean, digits=3))  (expected ≈ 0.5)")
@assert abs(pit_kde_mean - 0.5) < 0.15 "KDE PIT mean too far from 0.5"

# --- Test 3b: rank_histogram_cdf with KDE (observations-based) ---
println("\n=== Test 3b: rank_histogram_cdf (KDE, observations) ===")
observations_kde = [μ_true + σ_true * randn() for _ in 1:n_rank]
pit_kde_obs = rank_histogram_cdf(observations_kde, levels, Function[qoi_function],
                                 samples_per_level,
                                 estimate_cdf_mlmc_kernel_density,
                                 draw_parameters)

println("  PIT range = [$(round(minimum(pit_kde_obs), digits=3)), $(round(maximum(pit_kde_obs), digits=3))]")
@assert all(0.0 .<= pit_kde_obs .<= 1.0) "KDE PIT values out of [0,1]"
pit_kde_obs_mean = mean(pit_kde_obs)
println("  PIT mean = $(round(pit_kde_obs_mean, digits=3))  (expected ≈ 0.5)")
@assert abs(pit_kde_obs_mean - 0.5) < 0.15 "KDE PIT mean too far from 0.5"

# --- Test 4: rank_histogram_cdf with MaxEnt ---
println("\n=== Test 4: rank_histogram_cdf (MaxEnt) ===")
maxent_cdf_method(s, i) = first(estimate_cdf_maxent(s, i; R=4))
samples_per_level_maxent = [2000, 1000, 200]

pit_maxent = rank_histogram_cdf(levels, qoi_function, draw_parameters,
                                n_rank, samples_per_level_maxent,
                                maxent_cdf_method)

println("  PIT range = [$(round(minimum(pit_maxent), digits=3)), $(round(maximum(pit_maxent), digits=3))]")
@assert all(-0.05 .<= pit_maxent .<= 1.05) "MaxEnt PIT values far out of [0,1]"
pit_maxent_mean = mean(pit_maxent)
println("  PIT mean = $(round(pit_maxent_mean, digits=3))  (expected ≈ 0.5)")
@assert abs(pit_maxent_mean - 0.5) < 0.15 "MaxEnt PIT mean too far from 0.5"

println("\n✓ All rank histogram tests passed.")
