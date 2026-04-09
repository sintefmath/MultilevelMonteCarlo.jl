using Parameters
import Random: rand, Dims, AbstractRNG

@with_kw struct ProjectileParameters{HeightType, SpeedType, AngleType, MassType, DragType}
    initial_height::HeightType
    initial_speed::SpeedType
    angle_in_degrees::AngleType
    mass::MassType
    drag::DragType
end

function rand(uqParam::ProjectileParameters)
    h0 = uqParam.initial_height isa Real ? uqParam.initial_height : rand(uqParam.initial_height)
    v0 = uqParam.initial_speed isa Real ? uqParam.initial_speed : rand(uqParam.initial_speed)
    angle = uqParam.angle_in_degrees isa Real ? uqParam.angle_in_degrees : rand(uqParam.angle_in_degrees)
    m = uqParam.mass isa Real ? uqParam.mass : rand(uqParam.mass)
    ρ = uqParam.drag.ρ isa Real ? uqParam.drag.ρ : rand(uqParam.drag.ρ)
    C_d = uqParam.drag.C_d isa Real ? uqParam.drag.C_d : rand(uqParam.drag.C_d)
    A = uqParam.drag.A isa Real ? uqParam.drag.A : rand(uqParam.drag.A)
    return ProjectileParameters(
        initial_height = h0,
        initial_speed = v0,
        mass = m,
        angle_in_degrees = angle,
        drag = Drag(ρ=ρ, C_d=C_d, A=A)
    )
end

function rand(rng::AbstractRNG, uqParam::ProjectileParameters)
    h0 = uqParam.initial_height isa Real ? uqParam.initial_height : rand(rng, uqParam.initial_height)
    v0 = uqParam.initial_speed isa Real ? uqParam.initial_speed : rand(rng, uqParam.initial_speed)
    angle = uqParam.angle_in_degrees isa Real ? uqParam.angle_in_degrees : rand(rng, uqParam.angle_in_degrees)
    m = uqParam.mass isa Real ? uqParam.mass : rand(rng, uqParam.mass)
    ρ = uqParam.drag.ρ isa Real ? uqParam.drag.ρ : rand(rng, uqParam.drag.ρ)
    C_d = uqParam.drag.C_d isa Real ? uqParam.drag.C_d : rand(rng, uqParam.drag.C_d)
    A = uqParam.drag.A isa Real ? uqParam.drag.A : rand(rng, uqParam.drag.A)
    return ProjectileParameters(
        initial_height = h0,
        initial_speed = v0,
        mass = m,
        angle_in_degrees = angle,
        drag = Drag(ρ=ρ, C_d=C_d, A=A)
    )
end



function rand(rng::AbstractRNG, uqParam::ProjectileParameters, dims::Dims)
    ElementType = typeof(rand(rng, uqParam.initial_height))
    ParameterType = ProjectileParameters{ElementType, ElementType, ElementType, Drag{ElementType, ElementType, ElementType}}
    results = Array{ParameterType}(undef, dims)
    for idx in CartesianIndices(results)
        results[idx] = rand(rng, uqParam)
    end
    return results
end

function rand(uqParam::ProjectileParameters, dims::Dims)
    ElementType = typeof(rand(uqParam.initial_height))
    ParameterType = ProjectileParameters{ElementType, ElementType, ElementType, ElementType, Drag{ElementType, ElementType, ElementType}}
    results = Array{ParameterType}(undef, dims)
    for idx in CartesianIndices(results)
        results[idx] = rand(uqParam)
    end
    return results
end


function projectile_motion(parameters::ProjectileParameters; endtime=50.0, resolution=0.01)
    return projectile_motion(
        parameters.initial_height,
        endtime,
        parameters.initial_speed,
        parameters.angle_in_degrees,
        parameters.mass;
        drag = parameters.drag, dt = resolution
    )
end

function projectile_motion(initial_height, endtime, initial_speed, launch_angle, mass; drag = v -> 0.0, dt = 0.01)
    g = 9.81  # Acceleration due to gravity (m/s^2)
    θ = deg2rad(launch_angle)  # Convert angle to radians

    vx0 = initial_speed * cos(θ)
    vy0 = initial_speed * sin(θ)
    
    # Time vector
    t = 0:dt:endtime
    position = zeros(length(t), 2)
    velocity = zeros(length(t), 2)
    position[1, 1] = 0.0  # Initial x position
    position[1, 2] = initial_height  # Initial y position
    velocity[1, 1] = vx0  # Initial x velocity
    velocity[1, 2] = vy0  # Initial y velocity
    for i in 1:length(t)-1
        v = sqrt(velocity[i, 1]^2 + velocity[i, 2]^2)
        drag_force = drag(v)
        ax = -drag_force * (velocity[i, 1] / v) / mass
        ay = -g - drag_force * (velocity[i, 2] / v) / mass

        velocity[i+1, 1] = velocity[i, 1] + ax * dt
        velocity[i+1, 2] = velocity[i, 2] + ay * dt

        position[i+1, 1] = position[i, 1] + velocity[i, 1] * dt
        position[i+1, 2] = position[i, 2] + velocity[i, 2] * dt

        if position[i+1, 2] < 0
            position = position[1:i+1, :]
            t = t[1:i+1]
            break
        end
    end
    x = position[:, 1]
    y = position[:, 2]

    return t, x, y
end

function projectile_distance(parameters::ProjectileParameters; kwargs...)
    t, x, y = projectile_motion(parameters; kwargs...)
    return maximum(x)
end

function projectile_distance(args; kwargs...)
    t, x, y = projectile_motion(args...; kwargs...)
    return maximum(x)
end


@with_kw struct Drag{RhoType, CdType, AreaType}
    ρ::RhoType = 1.225  # Air density (kg/m^3)
    C_d::CdType = 0.47  # Drag coefficient
    A::AreaType  = 0.01  # Cross-sectional area (m^2)
end

function (d::Drag)(v)
    return 0.5 * d.ρ * d.C_d * d.A * v^2
end

using MultilevelMonteCarlo
using CairoMakie
using Random
using Distributions
using Parameters
using Statistics

h0 = Uniform(1.5,1.7)
v0 = Uniform(8, 12)
angle = Uniform(30, 60)
ρ = Uniform(1.0, 1.5)
m = Uniform(0.1, 0.5)
C_d = Uniform(0.3, 0.6)
A = Uniform(0.005, 0.015)

parameters_uncertain = ProjectileParameters(
    initial_height = h0,
    initial_speed = v0,
    angle_in_degrees = angle,
    mass = m,
    drag = Drag(ρ=ρ, C_d=C_d, A=A)
)

# Level evaluators: run projectile simulation at each timestep size
timestepsizes = 2.0 .^ (-2:-1:-6)
levels = Function[
    params -> projectile_motion(params; endtime=5.0, resolution=dt)
    for dt in timestepsizes
]

# QoI functions: extract scalar quantities from (t, x, y) model output
qoi_distance(output) = maximum(output[2])
qoi_max_height(output) = maximum(output[3])
qoi_functions = Function[qoi_distance, qoi_max_height]

samples_per_level = 1024 .* ones(Int, length(timestepsizes))

# Variance estimation for sample allocation (uses sum-of-squares by default)
variances_corrections, variances = variance_per_level(
    levels,
    qoi_functions,
    samples_per_level,
    () -> rand(parameters_uncertain),
)

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = "Resolution Δt", ylabel = "Variance", title = "Variance of Details per Level", yscale = log10, xscale = log10)

lines!(ax1, timestepsizes[2:end], variances_corrections,
    label = "Variance of Corrections", color = :blue)
scatter!(ax1, timestepsizes[2:end], variances_corrections,
    color = :blue)
lines!(ax1, timestepsizes, variances,
    label = "Variance of QoI", color = :red)
scatter!(ax1, timestepsizes, variances,
    color = :red)
axislegend(ax1; position = :rt)
fig

save("projectile_mlmc_variance.png", fig)
tolerance = 0.1
cost_per_level = timestepsizes .^ (-1)
samples_per_level_optimal = optimal_samples_per_level(
    variances_corrections,
    variances,
    cost_per_level,
    tolerance,
)

# MLMC estimation — returns [E[distance], E[max_height]]
mlmc_result = mlmc_estimate(
    levels,
    qoi_functions,
    samples_per_level,
    () -> rand(parameters_uncertain),
)
println("MLMC estimates: distance = ", mlmc_result[1], ", max height = ", mlmc_result[2])

# Adaptive MLMC estimation
mlmc_adaptive_result, adaptive_counts = mlmc_estimate_adaptive(
    levels,
    qoi_functions,
    () -> rand(parameters_uncertain),
    cost_per_level,
    tolerance;
    initial_samples = 100,
)
println("Adaptive MLMC estimates: distance = ", mlmc_adaptive_result[1],
        ", max height = ", mlmc_adaptive_result[2],
        " (samples: ", adaptive_counts, ")")

# Adaptive with custom variance_qoi (allocate based on distance only)
mlmc_adaptive_dist, _ = mlmc_estimate_adaptive(
    levels,
    qoi_functions,
    () -> rand(parameters_uncertain),
    cost_per_level,
    tolerance;
    variance_qoi = output -> maximum(output[2])^2,
    initial_samples = 100,
)
println("Adaptive MLMC (distance-based allocation): distance = ", mlmc_adaptive_dist[1])

# Single-level sanity check (should reduce to plain MC)
single_level_result = mlmc_estimate(
    levels[end:end],
    qoi_functions,
    [1000],
    () -> rand(parameters_uncertain),
)
println("Single-level MLMC (== plain MC): distance = ", single_level_result[1],
        ", max height = ", single_level_result[2])

# Standard Monte Carlo estimation for comparison (multi-QoI)
n_samples_mc = sum(samples_per_level_optimal)
mc_result = evaluate_monte_carlo(
    n_samples_mc,
    () -> rand(parameters_uncertain),
    levels[end],
    qoi_functions,
)
println("Standard MC estimates: distance = ", mc_result[1], ", max height = ", mc_result[2])

# Reference solution
reference_solution_number_of_samples = 1_000_000
reference_solution_dt = 2.0 ^ -10
ref_level = params -> projectile_motion(params; endtime=5.0, resolution=reference_solution_dt)
reference_solution = evaluate_monte_carlo(
    reference_solution_number_of_samples,
    () -> rand(parameters_uncertain),
    ref_level,
    qoi_functions,
)
println("Reference solution (", reference_solution_number_of_samples, " samples at dt=",
        reference_solution_dt, "): distance = ", reference_solution[1],
        ", max height = ", reference_solution[2])

# Work vs error comparison (using distance QoI)
errors_monte_carlo = Vector{Float64}()
errors_mlmc = Vector{Float64}()
tolerances = timestepsizes .^ 1.5
work_mlmc = Vector{Float64}()
work_monte_carlo = Vector{Float64}()
for (n, tol) in enumerate(tolerances)
    if n == 1
        continue
    end
    local_levels = levels[1:n]
    local_cost = cost_per_level[1:n]
    local_samples_optimal = optimal_samples_per_level(
        variances_corrections[1:n-1],
        variances[1:n],
        local_cost,
        tol,
    )

    mlmc_est = mlmc_estimate(
        local_levels,
        qoi_functions,
        local_samples_optimal,
        () -> rand(parameters_uncertain),
    )
    push!(errors_mlmc, abs(mlmc_est[1] - reference_solution[1]))
    push!(work_mlmc, sum(local_samples_optimal .* local_cost))

    local n_samples_mc = sum(local_samples_optimal)
    mc_est = evaluate_monte_carlo(
        n_samples_mc,
        () -> rand(parameters_uncertain),
        local_levels[end],
        qoi_functions,
    )
    push!(errors_monte_carlo, abs(mc_est[1] - reference_solution[1]))
    push!(work_monte_carlo, n_samples_mc * cost_per_level[end])
end

fig2 = Figure()
ax2 = Axis(fig2[1, 1], xlabel = "Work", ylabel = "Error (distance)",
           title = "MLMC vs Standard MC Work vs Error", yscale = log10, xscale = log10)
lines!(ax2, work_mlmc, errors_mlmc, label = "MLMC", color = :blue)
scatter!(ax2, work_mlmc, errors_mlmc, color = :blue)
lines!(ax2, work_monte_carlo, errors_monte_carlo, label = "Standard MC", color = :red)
scatter!(ax2, work_monte_carlo, errors_monte_carlo, color = :red)
axislegend(ax2; position = :rt)
fig2
