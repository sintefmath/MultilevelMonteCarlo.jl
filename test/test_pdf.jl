using MultilevelMonteCarlo
using Statistics
using Random

Random.seed!(42)

# Test problem: X ~ N(2, 0.5²)
# "Levels" approximate X with decreasing additive noise
μ_true = 2.0
σ_true = 0.5
levels = Function[
    params -> params + 0.2 * randn(),    # coarse
    params -> params + 0.02 * randn(),   # medium
    params -> Float64(params),            # exact
]
qoi_functions = Function[identity]
draw_parameters() = μ_true + σ_true * randn()

# Collect MLMC samples with enough data for density estimation
samples_per_level = [4000, 2000, 500]
samples = mlmc_sample(levels, qoi_functions, samples_per_level, draw_parameters)

# --- Test 1: KDE PDF ---
println("=== Test 1: KDE PDF ===")
pdf_kde = estimate_pdf_mlmc_kernel_density(samples, 1)

# Evaluate at the mean — should be close to N(2, 0.5) pdf peak ≈ 0.7979
peak_val = pdf_kde(μ_true)
println("  PDF at mean = $peak_val  (expected ≈ 0.798)")
@assert 0.5 < peak_val < 1.2 "KDE PDF peak out of range"

# PDF should be small far from the mean
tail_val = pdf_kde(μ_true + 5 * σ_true)
println("  PDF at 5σ   = $tail_val  (expected ≈ 0)")
@assert tail_val < 0.05 "KDE PDF tail too large"

# Symmetry check
left  = pdf_kde(μ_true - 0.5)
right = pdf_kde(μ_true + 0.5)
println("  PDF at μ-0.5 = $(round(left, digits=4)), μ+0.5 = $(round(right, digits=4))")
@assert abs(left - right) / max(left, right) < 0.2 "KDE PDF not roughly symmetric"

# --- Test 2: KDE CDF ---
println("\n=== Test 2: KDE CDF ===")
cdf_kde = estimate_cdf_mlmc_kernel_density(samples, 1)

# CDF at the mean should be ≈ 0.5
cdf_mid = cdf_kde(μ_true)
println("  CDF at mean = $cdf_mid  (expected ≈ 0.5)")
@assert 0.35 < cdf_mid < 0.65 "KDE CDF at mean out of range"

# CDF should be monotone and bounded
cdf_low  = cdf_kde(μ_true - 3 * σ_true)
cdf_high = cdf_kde(μ_true + 3 * σ_true)
println("  CDF at μ-3σ = $(round(cdf_low, digits=4)), μ+3σ = $(round(cdf_high, digits=4))")
@assert cdf_low < 0.05 "CDF too large in left tail"
@assert cdf_high > 0.95 "CDF too small in right tail"
@assert cdf_low < cdf_mid < cdf_high "CDF not monotone"

# --- Test 3: Maximum Entropy PDF ---
println("\n=== Test 3: Maximum Entropy PDF ===")
pdf_maxent, λ, a, b = estimate_pdf_maxent(samples, 1; R=6)
println("  Support: [$a, $b]")
println("  λ coefficients: $(round.(λ, digits=4))")

# MaxEnt PDF at mean should be near the true peak
me_peak = pdf_maxent(μ_true)
println("  MaxEnt PDF at mean = $me_peak  (expected ≈ 0.798)")
@assert 0.4 < me_peak < 1.5 "MaxEnt PDF peak out of range"

# Should be small in the tails
me_tail = pdf_maxent(μ_true + 4 * σ_true)
println("  MaxEnt PDF at μ+4σ = $me_tail")
@assert me_tail < 0.1 "MaxEnt PDF tail too large"

# The MaxEnt PDF should integrate to ≈ 1 over the support
using QuadGK
integral, _ = quadgk(pdf_maxent, a, b; rtol=1e-6)
println("  ∫ PDF dx = $(round(integral, digits=4))  (expected ≈ 1.0)")
@assert abs(integral - 1.0) < 0.1 "MaxEnt PDF does not integrate to ≈ 1"

# --- Test 4: MaxEnt CDF ---
println("\n=== Test 4: Maximum Entropy CDF ===")
cdf_maxent, _, _, _ = estimate_cdf_maxent(samples, 1; R=6)

me_cdf_mid = cdf_maxent(μ_true)
println("  MaxEnt CDF at mean = $me_cdf_mid  (expected ≈ 0.5)")
@assert 0.3 < me_cdf_mid < 0.7 "MaxEnt CDF at mean out of range"

me_cdf_right = cdf_maxent(b - 0.01)
println("  MaxEnt CDF near right = $me_cdf_right  (expected ≈ 1.0)")
@assert me_cdf_right > 0.9 "MaxEnt CDF near right support too low"

println("\nAll PDF/CDF tests passed!")
