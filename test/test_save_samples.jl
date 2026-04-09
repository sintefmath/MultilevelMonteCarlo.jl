using MultilevelMonteCarlo
using Statistics
using NCDatasets

# Simple test problem: X ~ N(0,1), QoI = X and X²
# "Levels" are noisy→exact approximations
levels = Function[
    params -> params + 0.1 * randn(),
    params -> params + 0.01 * randn(),
    params -> Float64(params),
]
qoi_functions = Function[identity, x -> x^2]

samples_per_level = [500, 300, 100]

# --- Test 1: collect samples in memory ---
println("=== Test 1: mlmc_sample (in memory) ===")
samples = mlmc_sample(levels, qoi_functions, samples_per_level, () -> randn())

@assert samples.n_levels == 3
@assert samples.n_qois == 2
@assert size(samples.fine[1]) == (2, 500)
@assert size(samples.fine[2]) == (2, 300)
@assert size(samples.fine[3]) == (2, 100)
# Level 1 coarse should be zeros (no coarser level)
@assert all(samples.coarse[1] .== 0.0)
# Corrections = fine - coarse
@assert samples.corrections[2] ≈ samples.fine[2] .- samples.coarse[2]
println("  Shapes OK, corrections consistent.")

# Compute MLMC estimate from samples
est = mlmc_estimate_from_samples(samples)
println("  MLMC estimate from samples: E[X] ≈ $(round(est[1], digits=3)), E[X²] ≈ $(round(est[2], digits=3))")

# --- Test 2: save to NetCDF and read back ---
println("\n=== Test 2: mlmc_sample → NetCDF round-trip ===")
ncpath = tempname() * ".nc"
samples_nc = mlmc_sample(levels, qoi_functions, samples_per_level, () -> randn();
                         netcdf_path=ncpath)
println("  Written to $ncpath")

# Read back
samples_read = read_mlmc_samples_netcdf(ncpath)
@assert samples_read.n_levels == samples_nc.n_levels
@assert samples_read.n_qois == samples_nc.n_qois
for lvl in 1:3
    @assert samples_read.fine[lvl] ≈ samples_nc.fine[lvl]
    @assert samples_read.coarse[lvl] ≈ samples_nc.coarse[lvl]
    @assert samples_read.corrections[lvl] ≈ samples_nc.corrections[lvl]
end
println("  Round-trip OK: all arrays match.")

est2 = mlmc_estimate_from_samples(samples_read)
println("  MLMC estimate from NetCDF: E[X] ≈ $(round(est2[1], digits=3)), E[X²] ≈ $(round(est2[2], digits=3))")

# Clean up
rm(ncpath)

# --- Test 3: single-level (plain MC) ---
println("\n=== Test 3: single-level sample collection ===")
samples_single = mlmc_sample(levels[3:3], qoi_functions, [1000], () -> randn())
@assert samples_single.n_levels == 1
@assert all(samples_single.coarse[1] .== 0.0)
est3 = mlmc_estimate_from_samples(samples_single)
println("  Single-level: E[X] ≈ $(round(est3[1], digits=3)), E[X²] ≈ $(round(est3[2], digits=3))")

println("\nAll sample storage tests passed!")
