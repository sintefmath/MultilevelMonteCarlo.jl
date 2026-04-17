using Random

include("experiments_univariate.jl")
include("experiments_multivariate.jl")

Random.seed!(1234)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Univariate
const UNI_N_OBS              = 100
const UNI_N_RESAMPLES        = 500
const UNI_SAMPLES_PER_LEVEL  = [4000, 2000, 500]

# Multivariate
const MV_N_OBS               = 60
const MV_N_RESAMPLES         = 200
const MV_SAMPLES_PER_LEVEL   = [800, 400, 100]

# Applications
uni_applications = [
    normal_univariate_app(),
    projectile_univariate_app(),
]

mv_applications = [
    wilks_mvn_app(),
    projectile_wind_app(),
]

# ------------------------------------------------------------
# Run all univariate experiments
# ------------------------------------------------------------

println("=== Univariate experiments ===")
for app in uni_applications
    results = Any[]
    for variant in UNIVARIATE_OBS_VARIANTS, method in UNIVARIATE_METHODS
        println("  $(app.name): variant=$(variant), method=$(method)")
        r = run_univariate_experiment(
            app, variant, method;
            n_obs = UNI_N_OBS,
            n_resamples = UNI_N_RESAMPLES,
            samples_per_level = UNI_SAMPLES_PER_LEVEL,
        )
        push!(results, r)
    end
    outfile = plot_univariate_results(results, app.name)
    println("  → saved $(outfile)")
end

# ------------------------------------------------------------
# Run all multivariate experiments
# ------------------------------------------------------------

println("\n=== Multivariate experiments ===")
for app in mv_applications
    results = Any[]
    for variant in MULTIVARIATE_OBS_VARIANTS, method in MULTIVARIATE_METHODS
        println("  $(app.name): variant=$(variant), method=$(method)")
        r = run_multivariate_experiment(
            app, variant, method;
            n_obs = MV_N_OBS,
            n_resamples = MV_N_RESAMPLES,
            samples_per_level = MV_SAMPLES_PER_LEVEL,
        )
        push!(results, r)
    end
    outfile = plot_multivariate_results(results, app.name)
    println("  → saved $(outfile)")
end

println("\n✓ All experiments finished.")
