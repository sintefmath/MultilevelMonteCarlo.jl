using MultilevelMonteCarlo
using Statistics
using Random

Random.seed!(88)

# --------------------------------------------------------------------------- #
# Test problem: 2D Gaussian (X₁, X₂) ~ N([2, -1], diag(0.25, 0.16))
# --------------------------------------------------------------------------- #
μ = [2.0, -1.0]
σ = [0.5, 0.4]

draw_parameters() = μ .+ σ .* randn(2)

levels = Function[
    params -> params + 0.3 * randn(2),
    params -> params + 0.03 * randn(2),
    params -> Float64.(params),
]
qoi_functions = Function[x -> x[1], x -> x[2]]

samples = mlmc_sample(levels, qoi_functions, [4000, 2000, 500], draw_parameters)

# --- Test 1: 2D KDE PDF ---
println("=== Test 1: 2D KDE PDF ===")
pdf_2d = estimate_pdf_mlmc_kernel_density_2d(samples, (1, 2))

# At the mean, 2D Gaussian PDF peak = 1/(2π σ₁ σ₂) ≈ 0.796
peak = pdf_2d(μ[1], μ[2])
println("  PDF at mean = $(round(peak, digits=3))  (expected ≈ 0.796)")
@assert 0.3 < peak < 1.5 "2D KDE PDF peak out of range"

# Far from mean should be small
tail = pdf_2d(μ[1] + 4σ[1], μ[2] + 4σ[2])
println("  PDF at μ+4σ = $(round(tail, digits=5))  (expected ≈ 0)")
@assert tail < 0.05 "2D KDE PDF tail too large"

# --- Test 2: 2D KDE CDF ---
println("\n=== Test 2: 2D KDE CDF ===")
cdf_2d = estimate_cdf_mlmc_kernel_density_2d(samples, (1, 2))

# At the mean: F(μ₁, μ₂) ≈ 0.25 for independent components
cdf_at_mean = cdf_2d(μ[1], μ[2])
println("  CDF at mean = $(round(cdf_at_mean, digits=3))  (expected ≈ 0.25)")
@assert 0.10 < cdf_at_mean < 0.45 "2D KDE CDF at mean out of range"

# Far upper-right ≈ 1
cdf_ur = cdf_2d(μ[1] + 5σ[1], μ[2] + 5σ[2])
println("  CDF at μ+5σ = $(round(cdf_ur, digits=3))  (expected ≈ 1.0)")
@assert cdf_ur > 0.9 "2D KDE CDF at far right too small"

# Far lower-left ≈ 0
cdf_ll = cdf_2d(μ[1] - 5σ[1], μ[2] - 5σ[2])
println("  CDF at μ-5σ = $(round(cdf_ll, digits=3))  (expected ≈ 0.0)")
@assert cdf_ll < 0.1 "2D KDE CDF at far left too large"

# --- Test 3: 2D MaxEnt PDF ---
println("\n=== Test 3: 2D MaxEnt PDF ===")
pdf_me, Λ, bounds1, bounds2 = estimate_pdf_maxent_2d(samples, (1, 2); R=3)

peak_me = pdf_me(μ[1], μ[2])
println("  MaxEnt PDF at mean = $(round(peak_me, digits=3))  (expected ≈ 0.796)")
@assert 0.2 < peak_me < 2.0 "2D MaxEnt PDF peak out of range"

# Outside support → 0
println("  bounds₁ = $(bounds1), bounds₂ = $(bounds2)")
@assert pdf_me(bounds1[1] - 1.0, μ[2]) == 0.0 "MaxEnt PDF outside support should be 0"

# --- Test 4: 2D MaxEnt CDF ---
println("\n=== Test 4: 2D MaxEnt CDF ===")
cdf_me, _, _, _ = estimate_cdf_maxent_2d(samples, (1, 2); R=3)

cdf_me_mean = cdf_me(μ[1], μ[2])
println("  MaxEnt CDF at mean = $(round(cdf_me_mean, digits=3))  (expected ≈ 0.25)")
@assert 0.05 < cdf_me_mean < 0.5 "2D MaxEnt CDF at mean out of range"

# --- Test 5: MRH with 2D KDE CDF ---
println("\n=== Test 5: MRH with 2D KDE CDF ===")
n_rank = 60
samples_per_level = [800, 400, 100]

pit_kde = multivariate_rank_histogram(levels, qoi_functions, draw_parameters,
                                      n_rank, samples_per_level;
                                      number_of_resamples=200)

println("  PIT range = [$(round(minimum(pit_kde), digits=3)), $(round(maximum(pit_kde), digits=3))]")
@assert all(0.0 .<= pit_kde .<= 1.0) "MRH KDE PIT out of [0,1]"
pit_mean = mean(pit_kde)
println("  PIT mean = $(round(pit_mean, digits=3))  (expected ≈ 0.5)")
@assert abs(pit_mean - 0.5) < 0.2 "MRH KDE PIT mean too far from 0.5"

# --- Test 6: MRH with 2D MaxEnt CDF ---
println("\n=== Test 6: MRH with MaxEnt CDF ===")
maxent_cdf_2d(s, idx) = first(estimate_cdf_maxent_2d(s, idx; R=3))
samples_per_level_me = [2000, 1000, 200]

pit_me = multivariate_rank_histogram(levels, qoi_functions, draw_parameters,
                                     n_rank, samples_per_level_me;
                                     number_of_resamples=200,
                                     cdf_method=maxent_cdf_2d)

println("  PIT range = [$(round(minimum(pit_me), digits=3)), $(round(maximum(pit_me), digits=3))]")
@assert all(-0.05 .<= pit_me .<= 1.05) "MRH MaxEnt PIT far out of [0,1]"
pit_me_mean = mean(pit_me)
println("  PIT mean = $(round(pit_me_mean, digits=3))  (expected ≈ 0.5)")
@assert abs(pit_me_mean - 0.5) < 0.25 "MRH MaxEnt PIT mean too far from 0.5"

println("\n✓ All 2D PDF/CDF and MRH tests passed.")
