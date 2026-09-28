using MultilevelMonteCarlo
using CairoMakie
using DelimitedFiles
using Distributions
using GaussianRandomFields
using LinearAlgebra
using Random
using Statistics

# Wilks Σ0 construction + miscalibration helpers
include("wilks_utils.jl")

# ============================================================
# Generic multivariate experiment framework
#
# A multivariate application is a NamedTuple:
#   (name, levels, qoi_functions, sim_draw, obs_providers)
# where
#   - levels        :: Vector{Function}     (each returns a d-vector or
#                                            a data object from which
#                                            qoi_functions extract d scalars)
#   - qoi_functions :: Vector{Function}     (currently d == 2 or d == 3)
#   - sim_draw      :: () -> params
#   - obs_providers :: Dict{Symbol, Function}
#                       obs_providers[variant](n_obs) -> Matrix d×n_obs
#
# NOTE: `multivariate_rank_histogram` currently supports d == 2 only.
# For the Wilks application we project to the first two coordinates.
# ============================================================

const MULTIVARIATE_OBS_VARIANTS = (:matched, :under, :over, :bias, :skew_covariance)
const MULTIVARIATE_METHODS      = (:singlelevel, :kde_cdf, :maxent_cdf)

# -- Wilks multivariate Gaussian application -----------------------------

"""
    wilks_mvn_app()

Multivariate normal application (d = 2) matching the Wilks (2017) setup
projected to the first two coordinates.
"""
function wilks_mvn_app(; n_levels::Integer = 3)
    @assert n_levels >= 2
    d = 3
    Σ0 = truth_covariance(d; rho = 0.6, sigma2 = 1.0)

    # Levels add geometrically decreasing isotropic noise; finest is exact.
    σs = _geomspace(0.3, 0.03, n_levels - 1)
    levels = Function[(params -> params .+ σ .* randn(d)) for σ in σs]
    push!(levels, params -> Float64.(params))
    qoi_functions = Function[x -> x[1], x -> x[2]]

    dist_truth = MvNormal(zeros(d), Symmetric(Σ0))
    sim_draw() = rand(dist_truth)

    # Observations: use level[end] applied to a params draw from the
    # variant's distribution, then project to (x1, x2).
    function make_obs_provider(draw_obs_params::Function)
        function provider(n_obs::Int)
            mat = Matrix{Float64}(undef, 2, n_obs)
            for i in 1:n_obs
                p = draw_obs_params()
                y = levels[end](p)
                mat[1, i] = qoi_functions[1](y)
                mat[2, i] = qoi_functions[2](y)
            end
            return mat
        end
        return provider
    end

    # Variant parameter distributions (what generates the "observed" truth)
    dist_matched = dist_truth
    dist_under   = MvNormal(zeros(d), Symmetric(type1_covariance(d; sigma2_forecast = 0.5)))
    dist_over    = MvNormal(zeros(d), Symmetric(type1_covariance(d; sigma2_forecast = 2.0)))
    dist_bias    = MvNormal([0.4, 0.0, 0.0], Symmetric(Σ0))
    dist_skew    = MvNormal(zeros(d), Symmetric(type3_covariance(30.0)))

    obs_providers = Dict{Symbol,Function}(
        :matched         => make_obs_provider(() -> rand(dist_matched)),
        :under           => make_obs_provider(() -> rand(dist_under)),
        :over            => make_obs_provider(() -> rand(dist_over)),
        :bias            => make_obs_provider(() -> rand(dist_bias)),
        :skew_covariance => make_obs_provider(() -> rand(dist_skew)),
    )

    return (
        name = "wilks_mvn_$(n_levels)L",
        levels = levels,
        qoi_functions = qoi_functions,
        sim_draw = sim_draw,
        obs_providers = obs_providers,
    )
end

# -- Projectile-in-wind application --------------------------------------

"""
    projectile_wind_app(; seed=123)

2-D landing-position application for a projectile flying through an
uncertain wind field. Levels differ in the Euler integration timestep.
"""
function projectile_wind_app(; seed::Int = 123, n_levels::Integer = 3)
    @assert n_levels >= 1
    Random.seed!(seed)

    g_acc = 9.81
    T_max = 6.0
    n_wind = 201
    t_wind = collect(range(0.0, T_max; length = n_wind))
    cov_fn = CovarianceFunction(1, GaussianRandomFields.Exponential(1.5))
    grf_wind = GaussianRandomField(cov_fn, GaussianRandomFields.Cholesky(), t_wind)

    function _wind_at(vals, t)
        t <= t_wind[1]   && return vals[1]
        t >= t_wind[end] && return vals[end]
        idx = searchsortedlast(t_wind, t)
        idx = clamp(idx, 1, length(t_wind) - 1)
        α = (t - t_wind[idx]) / (t_wind[idx+1] - t_wind[idx])
        return (1 - α) * vals[idx] + α * vals[idx+1]
    end

    function make_level(dt)
        function simulate(params)
            vx0, vy0, vz0, wx, wy = params
            x, y, z = 0.0, 0.0, 0.0
            vx, vy, vz = vx0, vy0, vz0
            t = 0.0
            x_prev, y_prev, z_prev = x, y, z
            while true
                ax = _wind_at(wx, t)
                ay = _wind_at(wy, t)
                x_prev, y_prev, z_prev = x, y, z
                vx += ax * dt; vy += ay * dt; vz -= g_acc * dt
                x  += vx * dt; y  += vy * dt; z  += vz * dt
                t  += dt
                if z <= 0.0 || t >= T_max
                    if z_prev > 0.0 && z < 0.0
                        α = z_prev / (z_prev - z)
                        x = x_prev + α * (x - x_prev)
                        y = y_prev + α * (y - y_prev)
                    end
                    break
                end
            end
            return [x, y]
        end
        return simulate
    end

    dts = _geomspace(0.1, 0.004, n_levels)
    levels = Function[make_level(dt) for dt in dts]
    qoi_functions = Function[r -> r[1], r -> r[2]]

    make_draw(; μx = 20.0, μy = 0.0, μz = 20.0, σv = 1.0) = function ()
        vx0 = μx + σv * randn()
        vy0 = μy + σv * randn()
        vz0 = μz + σv * randn()
        wx  = 2.0 .* GaussianRandomFields.sample(grf_wind)
        wy  = 2.0 .* GaussianRandomFields.sample(grf_wind)
        return (vx0, vy0, vz0, wx, wy)
    end

    sim_draw = make_draw()

    function make_obs_provider(draw::Function)
        function provider(n_obs::Int)
            mat = Matrix{Float64}(undef, 2, n_obs)
            for i in 1:n_obs
                r = levels[end](draw())
                mat[1, i] = qoi_functions[1](r)
                mat[2, i] = qoi_functions[2](r)
            end
            return mat
        end
        return provider
    end

    obs_providers = Dict{Symbol,Function}(
        :matched         => make_obs_provider(make_draw()),
        :under           => make_obs_provider(make_draw(σv = 0.5)),
        :over            => make_obs_provider(make_draw(σv = 2.0)),
        :bias            => make_obs_provider(make_draw(μx = 22.0)),
        :skew_covariance => make_obs_provider(make_draw(μy = 1.0, σv = 1.5)),
    )

    return (
        name = "projectile_wind_$(n_levels)L",
        levels = levels,
        qoi_functions = qoi_functions,
        sim_draw = sim_draw,
        obs_providers = obs_providers,
    )
end

# -- generic runner -------------------------------------------------------

"""
    run_multivariate_experiment(app, variant, method;
                                n_obs=60, n_resamples=200,
                                samples_per_level=[800, 400, 100])

Return `(pit, method, variant, app_name)`.
"""
function run_multivariate_experiment(
    app,
    variant::Symbol,
    method::Symbol;
    n_obs::Int = 60,
    n_resamples::Int = 200,
    samples_per_level::AbstractVector{<:Integer} = [800, 400, 100],
)
    @assert variant in MULTIVARIATE_OBS_VARIANTS
    @assert method in MULTIVARIATE_METHODS

    observations = app.obs_providers[variant](n_obs)

    if method === :kde_cdf
        pit = multivariate_rank_histogram(
            observations, app.levels, app.qoi_functions,
            samples_per_level, app.sim_draw;
            number_of_resamples = n_resamples,
            cdf_method = estimate_cdf_mlmc_kernel_density_2d,
        )
    elseif method === :maxent_cdf
        cdf_method = (s, idx) -> first(estimate_cdf_maxent_2d(s, idx; R = 3))
        pit = multivariate_rank_histogram(
            observations, app.levels, app.qoi_functions,
            samples_per_level, app.sim_draw;
            number_of_resamples = n_resamples,
            cdf_method = cdf_method,
        )
    elseif method === :singlelevel
        single_levels = app.levels[end:end]
        single_spl = [sum(samples_per_level)]
        pit = multivariate_rank_histogram(
            observations, single_levels, app.qoi_functions,
            single_spl, app.sim_draw;
            number_of_resamples = n_resamples,
            cdf_method = estimate_cdf_mlmc_kernel_density_2d,
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
    plot_multivariate_results(results, app_name; outdir="test/plots", n_bins=10)

Plot a (variant × method) grid of multivariate PIT histograms.
"""
function plot_multivariate_results(
    results,
    app_name::AbstractString;
    outdir::AbstractString = "test/plots",
    n_bins::Int = 15,
    title_prefix::AbstractString = "Multivariate",
    save_individual::Bool = false,
    save_data::Bool = false,
)
    mkpath(outdir)

    variants = MULTIVARIATE_OBS_VARIANTS
    methods  = MULTIVARIATE_METHODS
    lookup = Dict((r.variant, r.method) => r.pit for r in results)

    fig = Figure(size = (320 * length(methods), 240 * length(variants)), fontsize = 12)
    Label(fig[0, :], "$(title_prefix): $app_name", fontsize = 18, font = :bold)

    for (i, v) in enumerate(variants), (j, m) in enumerate(methods)
        ax = Axis(
            fig[i, j];
            title = "$(v) · $(m)",
            xlabel = "PIT",
            ylabel = "count",
        )
        if haskey(lookup, (v, m))
            pit = filter(isfinite, lookup[(v, m)])
            if !isempty(pit)
                hist!(ax, pit; bins = n_bins, color = :mediumpurple, strokewidth = 1)
                hlines!(ax, [length(pit) / n_bins]; color = :red, linestyle = :dash)
            end
        end
    end

    outfile = joinpath(outdir, "multivariate_$(app_name)_rank_histograms.png")
    save(outfile, fig)

    if save_individual
        indiv_dir = joinpath(outdir, "individual")
        mkpath(indiv_dir)
        for (v, m) in keys(lookup)
            pit = filter(isfinite, lookup[(v, m)])
            isempty(pit) && continue
            f = Figure(size = (480, 360), fontsize = 12)
            ax = Axis(f[1, 1]; title = "$(app_name): $(v) · $(m)",
                      xlabel = "PIT", ylabel = "count")
            hist!(ax, pit; bins = n_bins, color = :mediumpurple, strokewidth = 1)
            hlines!(ax, [length(pit) / n_bins]; color = :red, linestyle = :dash)
            save(joinpath(indiv_dir, "$(v)_$(m).png"), f)
        end
    end

    if save_data
        data_dir = joinpath(outdir, "data")
        mkpath(data_dir)
        for (v, m) in keys(lookup)
            pit = lookup[(v, m)]
            open(joinpath(data_dir, "$(v)_$(m).csv"), "w") do io
                println(io, "pit")
                writedlm(io, pit)
            end
        end
    end

    return outfile
end
