# API Reference

## MLMC Estimation

```@docs
mlmc_estimate
mlmc_estimate_adaptive
mlmc_estimate_vector_qoi
evaluate_monte_carlo
evaluate_monte_carlo_vector_qoi
```

## Sample Allocation

```@docs
variance_per_level
optimal_samples_per_level
```

## Sample Collection & Storage

```@docs
MLMCSamples
mlmc_sample
mlmc_estimate_from_samples
read_mlmc_samples_netcdf
```

## PDF & CDF Estimation

```@docs
estimate_pdf_mlmc_kernel_density
estimate_cdf_mlmc_kernel_density
estimate_pdf_maxent
estimate_cdf_maxent
```

## Rank Histograms

```@docs
ml_gregory_resample
rank_histogram_gregory
rank_histogram_cdf
```

## Online Statistics (Internals)

```@docs
MultilevelMonteCarlo.OnlineAccumulator
MultilevelMonteCarlo.update!
MultilevelMonteCarlo.get_variance
Base.merge(::MultilevelMonteCarlo.OnlineAccumulator, ::MultilevelMonteCarlo.OnlineAccumulator)
MultilevelMonteCarlo.merge!
MultilevelMonteCarlo.accumulate_samples
MultilevelMonteCarlo.accumulate_samples!
MultilevelMonteCarlo.accumulate_vector_samples
MultilevelMonteCarlo.accumulate_vector_samples!
MultilevelMonteCarlo._default_variance_qoi
MultilevelMonteCarlo._legendre_polynomials
```
