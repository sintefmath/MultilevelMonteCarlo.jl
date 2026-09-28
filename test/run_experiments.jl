using MultilevelMonteCarlo
using DelimitedFiles
using LinearAlgebra
using Random
using Statistics
using Base.Threads: @spawn, Atomic, atomic_add!, nthreads

include("experiments_univariate.jl")
include("experiments_multivariate.jl")

# ============================================================
# Command-line options
#
#   julia --project=test test/run_experiments.jl              # full run
#   julia --project=test test/run_experiments.jl --test       # tiny run for smoke testing
#   julia --project=test test/run_experiments.jl --outdir DIR # override output dir
#   julia --project=test test/run_experiments.jl --skip-variance
#   julia --threads=auto --project=test test/run_experiments.jl --parallel
#       run experiments concurrently, one worker per Julia thread
#   julia --project=test test/run_experiments.jl --parallel --workers N
#       cap the number of worker tasks
#   julia --project=test test/run_experiments.jl only_missing
#       skip experiments with a valid CSV in test_done (or the output dir)
#   julia --project=test test/run_experiments.jl only_missing --done-dir DIR
#       use DIR as the completed-results directory
#
# ============================================================

const TEST_MODE      = ("--test" in ARGS) || ("-t" in ARGS)
const SKIP_VARIANCE  = ("--skip-variance" in ARGS)
const PARALLEL       = ("--parallel" in ARGS) || ("-p" in ARGS)
const ONLY_MISSING   = ("only_missing" in ARGS) || ("--only-missing" in ARGS)

const NWORKERS = let
    i = findfirst(==("--workers"), ARGS)
    if i !== nothing && i < length(ARGS)
        max(1, parse(Int, ARGS[i + 1]))
    else
        max(1, nthreads())
    end
end

const OUTDIR = let
    i = findfirst(==("--outdir"), ARGS)
    if i !== nothing && i < length(ARGS)
        ARGS[i + 1]
    else
        joinpath("test", "plots", TEST_MODE ? "test_run" : "full_run")
    end
end

# `test_done` is a snapshot of previously completed experiments.  A custom
# directory is useful when resuming results stored elsewhere; it must contain
# the configuration directories directly (for example `fixed_baseline_3L`).
const COMPLETED_DIR = let
    i = findfirst(==("--done-dir"), ARGS)
    if i !== nothing && i < length(ARGS)
        ARGS[i + 1]
    else
        joinpath(dirname(@__DIR__), "test_done", "plots",
                 TEST_MODE ? "test_run" : "full_run")
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

# ------------------------------------------------------------
# Task-based execution
#
# Each task is one full experiment: a single (config, app, variant, method)
# combination. Worker tasks claim experiments from an atomic counter,
# one per Julia thread when --parallel is set. Results for the same
# (config, kind, app) plotting group are accumulated, then plotted once
# all of that group's tasks have completed.
# ------------------------------------------------------------

struct ExperimentTask
    kind::Symbol            # :univariate or :multivariate
    config_dir::String
    app::Any
    variant::Symbol
    method::Symbol
    samples_per_level::Vector{Int}
    n_obs::Int
    n_resamples::Int
    group_key::Tuple{String,Symbol,String}   # (config_dir, kind, app.name)
    seed::UInt64
end

"""Return the per-task CSV path below a run-results directory."""
function task_result_path(task::ExperimentTask, run_dir::AbstractString)
    app_dir = "$(task.kind)_$(task.app.name)"
    return joinpath(run_dir, basename(task.config_dir), app_dir, "data",
                    "$(task.variant)_$(task.method).csv")
end

"""Return the CSV path where a newly completed task is checkpointed."""
function task_output_path(task::ExperimentTask)
    app_dir = "$(task.kind)_$(task.app.name)"
    return joinpath(task.config_dir, app_dir, "data",
                    "$(task.variant)_$(task.method).csv")
end

"""
    load_task_result(task) -> Union{Nothing, NamedTuple}

Load a completed task from the current output directory or `COMPLETED_DIR`.
A CSV is considered complete only when it has the expected header and exactly
`task.n_obs` values, so interrupted/partial writes are automatically rerun.
"""
function load_task_result(task::ExperimentTask)
    paths = (task_output_path(task), task_result_path(task, COMPLETED_DIR))
    for path in unique(paths)
        isfile(path) || continue
        try
            lines = readlines(path)
            isempty(lines) && continue
            strip(first(lines)) == "pit" || continue
            pit = parse.(Float64, filter(x -> !isempty(x), strip.(lines[2:end])))
            length(pit) == task.n_obs || continue
            return (
                pit = pit,
                method = task.method,
                variant = task.variant,
                app_name = task.app.name,
            )
        catch err
            @warn "invalid completed-result CSV; rerunning task" path exception=(err, catch_backtrace())
        end
    end
    return nothing
end

"""Atomically checkpoint one successful task, before the full run finishes."""
function save_task_result(task::ExperimentTask, result)
    path = task_output_path(task)
    dir = dirname(path)
    mkpath(dir)
    temp_path, io = mktemp(dir)
    try
        println(io, "pit")
        writedlm(io, result.pit)
        close(io)
        mv(temp_path, path; force = true)
    finally
        isopen(io) && close(io)
        isfile(temp_path) && rm(temp_path)
    end
    return path
end

function execute_task(task::ExperimentTask)
    Random.seed!(Random.default_rng(), task.seed)
    println("    [$(task.kind)] $(task.app.name) | $(task.variant) | $(task.method) | spl=$(task.samples_per_level)")
    if task.kind === :univariate
        return run_univariate_experiment(
            task.app, task.variant, task.method;
            n_obs = task.n_obs,
            n_resamples = task.n_resamples,
            samples_per_level = task.samples_per_level,
        )
    else
        return run_multivariate_experiment(
            task.app, task.variant, task.method;
            n_obs = task.n_obs,
            n_resamples = task.n_resamples,
            samples_per_level = task.samples_per_level,
        )
    end
end

"""
    build_experiment_tasks(plans)

Given a vector of plan tuples `(kind, config_dir, app, samples_per_level,
n_obs, n_resamples)`, expand into one `ExperimentTask` per
(variant, method) combination.
"""
function build_experiment_tasks(plans)
    tasks = ExperimentTask[]
    for plan in plans
        kind, config_dir, app, spl, n_obs, n_resamples = plan
        variants = kind === :univariate ? UNIVARIATE_OBS_VARIANTS : MULTIVARIATE_OBS_VARIANTS
        methods  = kind === :univariate ? UNI_METHODS              : MV_METHODS
        for variant in variants, method in methods
            seed = hash((config_dir, kind, app.name, variant, method, spl))
            push!(tasks, ExperimentTask(
                kind, config_dir, app, variant, method,
                Int.(collect(spl)), n_obs, n_resamples,
                (config_dir, kind, app.name), seed,
            ))
        end
    end
    return tasks
end

"""
    run_tasks(tasks; parallel, nworkers) -> Dict(group_key => Vector{result})
"""
function run_tasks(tasks::Vector{ExperimentTask}; parallel::Bool, nworkers::Int)
    results = Dict{Tuple{String,Symbol,String},Vector{Any}}()
    for t in tasks
        results[t.group_key] = Any[]
    end

    if !parallel || nworkers <= 1 || length(tasks) <= 1
        for (i, task) in enumerate(tasks)
            try
                started = time_ns()
                r = execute_task(task)
                save_task_result(task, r)
                push!(results[task.group_key], r)
                elapsed = (time_ns() - started) / 1e9
                println("    ✓ task $(i)/$(length(tasks)) finished in $(round(elapsed; digits = 1)) s")
            catch err
                err isa InterruptException && rethrow()
                @warn "experiment failed" kind=task.kind app=task.app.name variant=task.variant method=task.method exception=(err, catch_backtrace())
            end
        end
        return results
    end

    nw = min(nworkers, length(tasks))
    println("  (running $(length(tasks)) tasks across $(nw) worker(s))")

    # Workers claim indices atomically and retain results locally.  Merging is
    # done after fetch, so experiment threads never contend on a shared Dict or
    # result mutex.  Atomic claiming preserves dynamic load balancing for the
    # very different method runtimes.
    next_index = Atomic{Int}(1)
    finished = Atomic{Int}(0)
    workers = Vector{Task}(undef, nw)
    for w in 1:nw
        workers[w] = @spawn begin
            local_results = Tuple{ExperimentTask,Any}[]
            while true
                i = atomic_add!(next_index, 1)
                i > length(tasks) && break
                task = tasks[i]
                try
                    started = time_ns()
                    r = execute_task(task)
                    save_task_result(task, r)
                    push!(local_results, (task, r))
                    done = atomic_add!(finished, 1) + 1
                    elapsed = (time_ns() - started) / 1e9
                    println("    ✓ task $(done)/$(length(tasks)) finished in $(round(elapsed; digits = 1)) s")
                catch err
                    err isa InterruptException && rethrow()
                    @warn "experiment failed" worker=w kind=task.kind app=task.app.name variant=task.variant method=task.method exception=(err, catch_backtrace())
                end
            end
            local_results
        end
    end
    for worker in workers
        for (task, r) in fetch(worker)
            push!(results[task.group_key], r)
        end
    end
    return results
end

"""Split tasks into completed results and work that still needs to run."""
function select_missing_tasks(tasks::Vector{ExperimentTask})
    completed = Dict{Tuple{String,Symbol,String},Vector{Any}}()
    pending = ExperimentTask[]
    for task in tasks
        get!(completed, task.group_key, Any[])
        result = load_task_result(task)
        if result === nothing
            push!(pending, task)
        else
            push!(completed[task.group_key], result)
        end
    end
    return pending, completed
end

"""Merge grouped task results without sharing mutable state between workers."""
function merge_results!(destination, source)
    for (key, values) in source
        append!(get!(destination, key, Any[]), values)
    end
    return destination
end

"""
    plot_grouped_results(plans, results; group_keys=nothing)

For each plan, look up the accumulated results and call the appropriate
plotter. Plotting is done serially after all experiments finish. When
`group_keys` is supplied, unchanged groups are skipped (used by resume mode).
"""
function plot_grouped_results(plans, results; group_keys = nothing)
    for plan in plans
        kind, config_dir, app, _spl, _n_obs, _n_resamples = plan
        key = (config_dir, kind, app.name)
        group_keys !== nothing && key ∉ group_keys && continue
        rs  = get(results, key, Any[])
        if kind === :univariate
            appdir = joinpath(config_dir, "univariate_$(app.name)")
            mkpath(appdir)
            outfile = plot_univariate_results(
                rs, app.name;
                outdir = appdir,
                save_individual = true,
                save_data = true,
            )
            println("    → $(outfile)")
        else
            appdir = joinpath(config_dir, "multivariate_$(app.name)")
            mkpath(appdir)
            outfile = plot_multivariate_results(
                rs, app.name;
                outdir = appdir,
                save_individual = true,
                save_data = true,
            )
            println("    → $(outfile)")
        end
    end
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
println("  parallel     : $(PARALLEL) (julia threads = $(nthreads()), workers = $(NWORKERS))")
println("  only missing : $(ONLY_MISSING)")
ONLY_MISSING && println("  completed dir: $(COMPLETED_DIR)")
println()

mkpath(OUTDIR)

# A "plan" is one (kind, config_dir, app, samples_per_level, n_obs, n_resamples).
# Each plan is expanded into one task per (variant, method).
plans = Any[]

# --- Fixed-budget configurations -------------------------------------------

for n_levels in N_LEVELS_LIST, (scale_name, scale) in SAMPLING_SCALES
    config_name = "fixed_$(scale_name)_$(n_levels)L"
    config_dir = joinpath(OUTDIR, config_name)
    println("--- planning config: $(config_name) ---")

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

    for factory in uni_app_factories
        app = factory(n_levels)
        push!(plans, (:univariate, config_dir, app, uni_spl, UNI_N_OBS, UNI_N_RESAMPLES))
    end
    for factory in mv_app_factories
        app = factory(n_levels)
        push!(plans, (:multivariate, config_dir, app, mv_spl, MV_N_OBS, MV_N_RESAMPLES))
    end
end

# --- Variance-based (target-error) configurations --------------------------
# The variance/cost pilot runs are kept serial (they are cheap and produce
# inputs for the task list).

if !SKIP_VARIANCE
    for n_levels in N_LEVELS_LIST, tol in VARIANCE_TOLERANCES
        config_name = "variance_tol$(tol)_$(n_levels)L"
        config_dir = joinpath(OUTDIR, config_name)
        println("--- planning config: $(config_name) ---")

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
            push!(plans, (:univariate, config_dir, app, spl, UNI_N_OBS, UNI_N_RESAMPLES))
        end

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
            push!(plans, (:multivariate, config_dir, app, spl, MV_N_OBS, MV_N_RESAMPLES))
        end
    end
end

# --- Dispatch experiments and plot -----------------------------------------

tasks = build_experiment_tasks(plans)
println("\n→ $(length(plans)) plans expanded into $(length(tasks)) tasks")

if PARALLEL && nthreads() == 1
    @warn "--parallel was set but Julia is running with a single thread. " *
          "Re-launch with `julia --threads=auto ...` to actually use multiple cores."
end

if PARALLEL && NWORKERS > 1 && BLAS.get_num_threads() > 1
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    println("  (set BLAS threads from $(old_blas_threads) to 1 to avoid oversubscription)")
end

if ONLY_MISSING
    tasks, results = select_missing_tasks(tasks)
    n_completed = sum(length, values(results); init = 0)
    println("→ found $(n_completed) completed tasks; $(length(tasks)) still missing")
    changed_groups = Set(task.group_key for task in tasks)
else
    results = Dict{Tuple{String,Symbol,String},Vector{Any}}()
    changed_groups = nothing
end

if !isempty(tasks)
    new_results = run_tasks(tasks; parallel = PARALLEL, nworkers = NWORKERS)
    merge_results!(results, new_results)
end

if changed_groups !== nothing && isempty(changed_groups)
    println("\n--- no changed groups to plot ---")
else
    println("\n--- plotting ---")
    plot_grouped_results(plans, results; group_keys = changed_groups)
end

println("\n✓ All experiments finished. Output in $(OUTDIR)")
