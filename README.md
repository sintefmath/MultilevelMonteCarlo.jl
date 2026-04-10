# MultilevelMonteCarlo.jl

[![CI](https://github.com/sintefmath/MultilevelMonteCarlo.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/sintefmath/MultilevelMonteCarlo.jl/actions/workflows/ci.yml)
[![Docs](https://github.com/sintefmath/MultilevelMonteCarlo.jl/actions/workflows/docs.yml/badge.svg)](https://sintefmath.github.io/MultilevelMonteCarlo.jl/dev/)

Tools for (single-level) Monte Carlo and multilevel Monte Carlo estimation in Julia.

## Features

- **MLMC estimation** — fixed and adaptive sample allocation with online variance tracking (Welford/Chan)
- **Sample storage** — collect raw fine/coarse samples and export to NetCDF
- **PDF & CDF estimation** — kernel density and maximum-entropy density estimation from MLMC samples
- **Rank histograms** — Gregory inverse-CDF resampling and PIT histograms for ensemble validation
- **Vector QoI** — estimators for vector-valued quantities of interest (e.g. trajectories)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/sintefmath/MultilevelMonteCarlo.jl")
```

## Quick start

```julia
using MultilevelMonteCarlo

# Define levels (cheap → expensive), a QoI, and a parameter sampler
levels = Function[
    params -> params + 0.2 * randn(),
    params -> params + 0.02 * randn(),
    params -> Float64(params),
]
qoi_functions = Function[identity]
draw_parameters() = 2.0 + 0.5 * randn()

# Run MLMC estimation
result = mlmc_estimate(levels, qoi_functions, draw_parameters;
                       epsilon=0.05, initial_samples=100)
```

## Documentation

See the [full documentation](https://sintefmath.github.io/MultilevelMonteCarlo.jl/dev/) for examples on projectile motion, sample storage, PDF/CDF estimation, and rank histograms.
