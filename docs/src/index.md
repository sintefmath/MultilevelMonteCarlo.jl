# MultilevelMonteCarlo.jl

A Julia package for **Multilevel Monte Carlo** (MLMC) estimation with support
for multiple quantities of interest, adaptive sample allocation, sample
storage (including NetCDF I/O), and PDF/CDF estimation via kernel density
and maximum entropy methods.

## Features

- **MLMC estimation** with the classical telescoping-sum estimator
- **Adaptive MLMC** with iterative optimal sample allocation
- **Multiple QoIs** computed simultaneously from the same model evaluations
- **Online statistics** via Welford's algorithm (no sample storage required for means)
- **Optional threading** with Chan's parallel merge algorithm
- **Sample collection** and persistence to NetCDF files
- **PDF/CDF estimation** from MLMC samples using:
  - Kernel density estimation (Gaussian kernel)
  - Maximum entropy method (Legendre polynomial basis, Newton solver)

## Quick Start

```julia
using MultilevelMonteCarlo

# Define model levels (coarse → fine)
levels = Function[
    params -> my_model(params; resolution=0.1),
    params -> my_model(params; resolution=0.01),
    params -> my_model(params; resolution=0.001),
]

# Define quantities of interest
qoi_functions = Function[
    output -> output.distance,
    output -> output.max_height,
]

# Draw random parameters
draw_parameters() = rand_params()

# MLMC estimation
samples_per_level = [10000, 1000, 100]
estimates = mlmc_estimate(levels, qoi_functions, samples_per_level, draw_parameters)
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/sintefmath/MultilevelMonteCarlo.jl")
```
