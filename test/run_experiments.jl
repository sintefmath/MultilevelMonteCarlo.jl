using MultilevelMonteCarlo
using Random
using Statistics

include("experiments_univariate.jl")
include("experiments_multivariate.jl")

# ============================================================
# Command-line options
#
#   julia --project=test test/run_experiments.jl              # full run
#   julia --project=test test/run_experiments.jl --test       # tiny run for smoke testing
#   julia --project=test test/run_experiments.jl --outdir DIR # override output dir
#   julia --project=test test/run_experiments.jl --skip-variance
#
# ============================================================

const TEST_MODE      = ("--test" in ARGS) || ("-t" in ARGS)
const SKIP_VARIANCE  = ("--skip-variance" in ARGS)

const OUTDIR = let
    i = findfirst(==("--outdir"), ARGS)
    if i !== nothing && i < length(ARGS)
        ARGS[i + 1]
    else
        joinpath("test", "plots", TEST_MODE ? "test_run" : "full_run")
    end
end

# ------------------------------------------------------------
# Sample-budget configurations
# ------------------------------------------------------------

# Baseline samples per level (length 3 — scaled per n_levels below)
const UNI_BASELINE = TEST_MODE ? [40, 20, 5]  : [4000, 2000, 500]
const MV_BASELINE  = TEST_MODE ? [40, 20, 5]  : [4000, 2000, 500]

const UNI_N_OBS         = TEST_MODE ? 30  : 1000
const UNI_N_RESAMPLES   = TEST_MODE ? 50  : 5000
const MV_N_OBS          = TEST_MODE ? 30  : 1000
const MV_N_RESAMPLES    = TEST_MODE ? 20  : 200

const N_LEVELS_LIST   = [3, 5]
const SAMPLING_SCALES = [
    ("baseline",    1.0),
    ("undersampled", 0.1),
    ("oversampled",  2.0),
]
const VARIANCE_TOLERANCES = TEST_MODE ? [0.1] : [0.1, 0.01]

# Methods to actually run. The maxent solvers are numerically fragile and
# memory-hungry at very small sample counts, so skip them in --test mode.
const UNI_METHODS = TEST_MODE ?
    filter(m -> m !== :maxent_cdf, collect(UNIVARIATE_METHODS)) :
    collect(UNIVARIATE_METHODS)
const MV_METHODS  = TEST_MODE ?
    filter(m -> m !== :maxent_cdf, collect(MULTIVARIATE_METHODS)) :
    collect(MULTIVARIATE_METHODS)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

"""
    scaled_samples_per_level(baseline, n_levels, scale)

Build a length-`n_levels` sample budget by geometrically interpolating from
`baseline[1]` to `baseline[end]` and scaling by `scale`.
"""
function scaled_samples_per_level(baseline::AbstractVector{<:Integer},
                                  n_levels::Integer, scale::Real)
    if n_levels == length(baseline)
        v = baseline
    else
        v = _geomspace(baseline[1], baseline[end], n_levels)
    end
    return [max(1, round(Int, scale * x)) for x in v]
end

"""
    estimate_cost_per_level(levels, draw_parameters; n_warm=2, n_time=5)

Return a vector of average wall-clock cost (seconds) per evaluation at each
level, measured by timing `n_time` evaluations after `n_warm` warmups.
"""
function estimate_cost_per_level(levels, draw_parameters; n_warm::Int = 2, n_time::Int = 5)
    n_time = max(1, TEST_MODE ? 2 : n_time)
    costs = zeros(Float64, length(levels))
    for (l, L) in enumerate(levels)
        for _ in 1:n_warm
            L(draw_parameters())
        end
        t0 = time_ns()
        for _ in 1:n_time
            L(draw_parameters())
        end
        costs[l] = max((time_ns() - t0) / 1e9 / n_time, 1e-9)
    end
    return costs
end

"""
    optimal_samples_for_app(levels, qoi_functions, draw_parameters,
                            tolerance; pilot=20)

Estimate an MLMC optimal sample allocation for the given `tolerance`, using a
small pilot run for the variance estimates and a timing-based cost.
"""
function optimal_samples_for_app(levels, qoi_functions, draw_parameters,
                                 tolerance::Real; pilot::Int = 20)
    pilot = TEST_MODE ? max(8, pilot ÷ 2) : pilot
    pilot_spl = fill(pilot, length(levels))
    vc, vs = variance_per_level(levels, qoi_functions, pilot_spl, draw_parameters)
    cost = estimate_cost_per_level(levels, draw_parameters)
    spl = optimal_samples_per_level(vc, vs, cost, tolerance)
    if TEST_MODE
        spl = [clamp(s, 4, 200) for s in spl]
    else
        spl = [max(s, 8) for s in spl]
    end
    return spl, cost, vc, vs
end

"""
    save_config_summary(dir, info::Dict)

Write a small human-readable summary file describing how a config was built.
"""
function save_config_summary(dir::AbstractString, info::AbstractDict)
    mkpath(dir)
    open(joinpath(dir, "config.txt"), "w") do io
        for (k, v) in info
            println(io, "$(k): $(v)")
        end
    end
end

# ------------------------------------------------------------
# Application factories (parameterised by n_levels)
# ------------------------------------------------------------

uni_app_factories = [
    n -> normal_univariate_app(n_levels = n),
    n -> projectile_univariate_app(n_levels = n),
]

mv_app_factories = [
    n -> wilks_mvn_app(n_levels = n),
    n -> projectile_wind_app(n_levels = n),
]

# ------------------------------------------------------------
# Per-config runners
# ------------------------------------------------------------

function run_univariate_config(app, samples_per_level, config_dir;
                               n_obs, n_resamples)
    results = Any[]
    for variant in UNIVARIATE_OBS_VARIANTS, method in UNI_METHODS
        println("    $(app.name) | $(variant) | $(method) | spl=$(samples_per_level)")
        try
            r = run_univariate_experiment(
                app, variant, method;
                n_obs = n_obs,
                n_resamples = n_resamples,
                samples_per_level = samples_per_level,
            )
            push!(results, r)
        catch err
            @warn "univariate experiment failed" app=app.name variant=variant method=method exception=(err, catch_backtrace())
        end
    end
    appdir = joinpath(config_dir, "univariate_$(app.name)")
    mkpath(appdir)
    outfile = plot_univariate_results(
        results, app.name;
        outdir = appdir,
        save_individual = true,
        save_data = true,
    )
    println("    → $(outfile)")
end

function run_multivariate_config(app, samples_per_level, config_dir;
                                 n_obs, n_resamples)
    results = Any[]
    for variant in MULTIVARIATE_OBS_VARIANTS, method in MV_METHODS
        println("    $(app.name) | $(variant) | $(method) | spl=$(samples_per_level)")
        try
            r = run_multivariate_experiment(
                app, variant, method;
                n_obs = n_obs,
                n_resamples = n_resamples,
                samples_per_level = samples_per_level,
            )
            push!(results, r)
        catch err
            @warn "multivariate experiment failed" app=app.name variant=variant method=method exception=(err, catch_backtrace())
        end
    end
    appdir = joinpath(config_dir, "multivariate_$(app.name)")
    mkpath(appdir)
    outfile = plot_multivariate_results(
        results, app.name;
        outdir = appdir,
        save_individual = true,
        save_data = true,
    )
    println("    → $(outfile)")
end

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

Random.seed!(1234)

println("=== run_experiments.jl ===")
println("  test mode    : $(TEST_MODE)")
println("  outdir       : $(OUTDIR)")
println("  n_levels     : $(N_LEVELS_LIST)")
println("  scales       : $([s[1] for s in SAMPLING_SCALES])")
println("  tolerances   : $(SKIP_VARIANCE ? "[skipped]" : VARIANCE_TOLERANCES)")
println()

mkpath(OUTDIR)

# --- Fixed-budget configurations -------------------------------------------

for n_levels in N_LEVELS_LIST, (scale_name, scale) in SAMPLING_SCALES
    config_name = "fixed_$(scale_name)_$(n_levels)L"
    config_dir = joinpath(OUTDIR, config_name)
    println("--- config: $(config_name) ---")

    uni_spl = scaled_samples_per_level(UNI_BASELINE, n_levels, scale)
    mv_spl  = scaled_samples_per_level(MV_BASELINE,  n_levels, scale)

    save_config_summary(config_dir, Dict(
        "kind" => "fixed_budget",
        "n_levels" => n_levels,
        "scale_name" => scale_name,
        "scale" => scale,
        "univariate_samples_per_level" => uni_spl,
        "multivariate_samples_per_level" => mv_spl,
    ))

    println("  [univariate]")
    for factory in uni_app_factories
        app = factory(n_levels)
        run_univariate_config(app, uni_spl, config_dir;
                              n_obs = UNI_N_OBS, n_resamples = UNI_N_RESAMPLES)
    end

    println("  [multivariate]")
    for factory in mv_app_factories
        app = factory(n_levels)
        run_multivariate_config(app, mv_spl, config_dir;
                                n_obs = MV_N_OBS, n_resamples = MV_N_RESAMPLES)
    end
end

# --- Variance-based (target-error) configurations --------------------------

if !SKIP_VARIANCE
    for n_levels in N_LEVELS_LIST, tol in VARIANCE_TOLERANCES
        config_name = "variance_tol$(tol)_$(n_levels)L"
        config_dir = joinpath(OUTDIR, config_name)
        println("--- config: $(config_name) ---")

        println("  [univariate]")
        for factory in uni_app_factories
            app = factory(n_levels)
            spl, cost, vc, vs = optimal_samples_for_app(
                app.levels, Function[app.qoi_function],
                app.sim_draw, tol,
            )
            save_config_summary(joinpath(config_dir, "univariate_$(app.name)"), Dict(
                "kind" => "variance_target",
                "tolerance" => tol,
                "n_levels" => n_levels,
                "samples_per_level" => spl,
                "cost_per_level" => cost,
                "variances" => vs,
                "variances_corrections" => vc,
            ))
            run_univariate_config(app, spl, config_dir;
                                  n_obs = UNI_N_OBS, n_resamples = UNI_N_RESAMPLES)
        end

        println("  [multivariate]")
        for factory in mv_app_factories
            app = factory(n_levels)
            spl, cost, vc, vs = optimal_samples_for_app(
                app.levels, app.qoi_functions,
                app.sim_draw, tol,
            )
            save_config_summary(joinpath(config_dir, "multivariate_$(app.name)"), Dict(
                "kind" => "variance_target",
                "tolerance" => tol,
                "n_levels" => n_levels,
                "samples_per_level" => spl,
                "cost_per_level" => cost,
                "variances" => vs,
                "variances_corrections" => vc,
            ))
            run_multivariate_config(app, spl, config_dir;
                                    n_obs = MV_N_OBS, n_resamples = MV_N_RESAMPLES)
        end
    end
end

println("\n✓ All experiments finished. Output in $(OUTDIR)")
