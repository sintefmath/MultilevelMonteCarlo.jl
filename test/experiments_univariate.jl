using MultilevelMonteCarlo
using CairoMakie
using Random
using Statistics

# ============================================================
# Generic univariate experiment framework
#
# An "application" is a NamedTuple:
#   (name, levels, qoi_function, sim_draw, obs_draws)
# where
#   - levels        :: Vector{Function}
#   - qoi_function  :: Function (scalar output)
#   - sim_draw      :: () -> params   (used by MLMC sampler)
#   - obs_draws     :: Dict{Symbol, Function}
#                      obs_draws[variant]() -> observed scalar value
#
# A "method" is a Symbol in {:singlelevel, :gregory, :kde_cdf, :maxent_cdf}.
# ============================================================

const UNIVARIATE_OBS_VARIANTS = (:matched, :under, :over, :bias)
const UNIVARIATE_METHODS      = (:singlelevel, :gregory, :kde_cdf, :maxent_cdf)

# -- applications ---------------------------------------------------------

"""
    normal_univariate_app()

Baseline univariate app: Normal(0, 1) with three MLMC levels.
"""
function normal_univariate_app()
    levels = Function[
        params -> params + 0.2 * randn(),
        params -> params + 0.02 * randn(),
        params -> Float64(params),
    ]

    qoi_function = identity
    sim_draw() = randn()  # N(0, 1)

    obs_draws = Dict{Symbol,Function}(
        :matched => () -> randn(),
        :under   => () -> 0.5 * randn(),
        :over    => () -> 2.0 * randn(),
        :bias    => () -> 0.2 + randn(),
    )

    return (
        name = "normal",
        levels = levels,
        qoi_function = qoi_function,
        sim_draw = sim_draw,
        obs_draws = obs_draws,
    )
end

"""
    projectile_univariate_app()

Projectile-distance app: 1-D ballistic motion with level-dependent timestep.
Parameter is initial speed v0. The QoI is the landing distance.
"""
function projectile_univariate_app()
    g = 9.81
    θ = deg2rad(45.0)

    # Simple drag-free projectile distance (analytic-ish using RK1 with level dt)
    function simulate(v0; dt)
        vx = v0 * cos(θ)
        vy = v0 * sin(θ)
        x, y, t = 0.0, 1.0, 0.0
        x_prev, y_prev = x, y
        while y >= 0.0 && t < 20.0
            x_prev, y_prev = x, y
            x += vx * dt
            y += vy * dt
            vy -= g * dt
            t += dt
        end
        # interpolate to y = 0
        if y_prev > 0.0 && y < 0.0
            α = y_prev / (y_prev - y)
            x = x_prev + α * (x - x_prev)
        end
        return x
    end

    dts = [0.1, 0.02, 0.004]
    levels = Function[v0 -> simulate(v0; dt = dt) for dt in dts]
    qoi_function = identity

    μv = 10.0
    σv = 1.0
    sim_draw() = μv + σv * randn()

    obs_draws = Dict{Symbol,Function}(
        :matched => () -> μv + σv * randn(),
        :under   => () -> μv + 0.5 * σv * randn(),
        :over    => () -> μv + 2.0 * σv * randn(),
        :bias    => () -> (μv + 0.5) + σv * randn(),
    )

    return (
        name = "projectile",
        levels = levels,
        qoi_function = qoi_function,
        sim_draw = sim_draw,
        obs_draws = obs_draws,
    )
end

# -- generic runner -------------------------------------------------------

"""
    draw_univariate_observations(app, variant, n_obs)

Generate `n_obs` scalar observations via `app.levels[end]` applied to the
variant's parameter draw function.
"""
function draw_univariate_observations(app, variant::Symbol, n_obs::Int)
    draw = app.obs_draws[variant]
    return [app.qoi_function(app.levels[end](draw())) for _ in 1:n_obs]
end

"""
    run_univariate_experiment(app, variant, method;
                              n_obs=100, n_resamples=500,
                              samples_per_level=[4000, 2000, 500])

Return a `NamedTuple` `(pit, method, variant, app_name)` where `pit` is the
vector of normalized rank/PIT values in `[0, 1]`.
"""
function run_univariate_experiment(
    app,
    variant::Symbol,
    method::Symbol;
    n_obs::Int = 100,
    n_resamples::Int = 500,
    samples_per_level::AbstractVector{<:Integer} = [4000, 2000, 500],
)
    @assert variant in UNIVARIATE_OBS_VARIANTS
    @assert method in UNIVARIATE_METHODS

    observations = draw_univariate_observations(app, variant, n_obs)
    qoi_functions = Function[app.qoi_function]

    if method === :gregory
        ranks = rank_histogram_gregory(
            observations, app.levels, qoi_functions,
            samples_per_level, app.sim_draw;
            number_of_resamples = n_resamples,
        )
        pit = (ranks .- 0.5) ./ (n_resamples + 1)
    elseif method === :kde_cdf
        pit = rank_histogram_cdf(
            observations, app.levels, qoi_functions,
            samples_per_level, estimate_cdf_mlmc_kernel_density,
            app.sim_draw,
        )
    elseif method === :maxent_cdf
        cdf_method = (s, i) -> first(estimate_cdf_maxent(s, i; R = 4))
        pit = rank_histogram_cdf(
            observations, app.levels, qoi_functions,
            samples_per_level, cdf_method, app.sim_draw,
        )
    elseif method === :singlelevel
        single_levels = app.levels[end:end]
        single_spl = [sum(samples_per_level)]
        pit = rank_histogram_cdf(
            observations, single_levels, qoi_functions,
            single_spl, estimate_cdf_mlmc_kernel_density,
            app.sim_draw,
        )
    else
        error("unknown method $method")
    end

    return (
        pit = Float64.(pit),
        method = method,
        variant = variant,
        app_name = app.name,
    )
end

# -- generic plotter ------------------------------------------------------

"""
    plot_univariate_results(results, app_name; outdir="test/plots", n_bins=12)

`results` is an iterable of NamedTuples as returned by
[`run_univariate_experiment`](@ref) for a single application.
Produces a (variant × method) grid of rank histograms.
"""
function plot_univariate_results(
    results,
    app_name::AbstractString;
    outdir::AbstractString = "test/plots",
    n_bins::Int = 12,
)
    mkpath(outdir)

    variants = UNIVARIATE_OBS_VARIANTS
    methods  = UNIVARIATE_METHODS
    lookup = Dict((r.variant, r.method) => r.pit for r in results)

    fig = Figure(size = (320 * length(methods), 260 * length(variants)), fontsize = 12)
    Label(fig[0, :], "Univariate: $app_name", fontsize = 18, font = :bold)

    for (i, v) in enumerate(variants), (j, m) in enumerate(methods)
        ax = Axis(
            fig[i, j];
            title = "$(v) · $(m)",
            xlabel = "PIT",
            ylabel = "count",
        )
        if haskey(lookup, (v, m))
            pit = lookup[(v, m)]
            hist!(ax, pit; bins = n_bins, color = :steelblue, strokewidth = 1)
            hlines!(ax, [length(pit) / n_bins]; color = :red, linestyle = :dash)
        end
    end

    outfile = joinpath(outdir, "univariate_$(app_name)_rank_histograms.png")
    save(outfile, fig)
    return outfile
end
