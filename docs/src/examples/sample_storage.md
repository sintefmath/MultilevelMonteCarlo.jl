# Sample Storage & NetCDF

This example shows how to collect and store raw MLMC samples, optionally
writing them to a NetCDF file for later analysis.

## Collect Samples In Memory

```@example storage
using MultilevelMonteCarlo
using Statistics

# Simple test problem: X ~ N(0,1) with noisy→exact level hierarchy
levels = Function[
    params -> params + 0.1 * randn(),   # coarse (noisy)
    params -> params + 0.01 * randn(),  # medium
    params -> Float64(params),           # fine (exact)
]
qoi_functions = Function[identity, x -> x^2]

samples_per_level = [500, 300, 100]
samples = mlmc_sample(levels, qoi_functions, samples_per_level, () -> randn())

println("Number of levels: ", samples.n_levels)
println("Number of QoIs:   ", samples.n_qois)
for lvl in 1:samples.n_levels
    println("Level $lvl: fine=$(size(samples.fine[lvl])), coarse=$(size(samples.coarse[lvl]))")
end
```

The `corrections` field contains `fine - coarse` at each level (level 1
coarse is zero):

```@example storage
println("Level 1 coarse all zero: ", all(samples.coarse[1] .== 0.0))
println("Corrections consistent:  ", samples.corrections[2] ≈ samples.fine[2] .- samples.coarse[2])
```

## Compute MLMC Estimate from Stored Samples

```@example storage
est = mlmc_estimate_from_samples(samples)
println("MLMC estimate: E[X] ≈ ", round(est[1], digits=3),
        ", E[X²] ≈ ", round(est[2], digits=3))
```

## Save to and Load from NetCDF

Pass `netcdf_path` to write samples to a NetCDF-4 file alongside the
in-memory return:

```@example storage
using NCDatasets

ncpath = tempname() * ".nc"
samples_nc = mlmc_sample(levels, qoi_functions, samples_per_level, () -> randn();
                         netcdf_path=ncpath)

# Read back from disk
samples_read = read_mlmc_samples_netcdf(ncpath)

# Verify round-trip
for lvl in 1:3
    @assert samples_read.fine[lvl] ≈ samples_nc.fine[lvl]
    @assert samples_read.coarse[lvl] ≈ samples_nc.coarse[lvl]
end
println("NetCDF round-trip: all arrays match ✓")

# Estimate from loaded samples
est2 = mlmc_estimate_from_samples(samples_read)
println("Estimate from loaded samples: E[X] ≈ ", round(est2[1], digits=3),
        ", E[X²] ≈ ", round(est2[2], digits=3))

rm(ncpath)  # clean up
```

## Single-Level Collection

With a single level, `mlmc_sample` reduces to standard Monte Carlo sampling:

```@example storage
samples_single = mlmc_sample(levels[3:3], qoi_functions, [1000], () -> randn())
est3 = mlmc_estimate_from_samples(samples_single)
println("Single-level MC: E[X] ≈ ", round(est3[1], digits=3),
        ", E[X²] ≈ ", round(est3[2], digits=3))
```
