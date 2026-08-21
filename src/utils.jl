import PrettyTables

"""
    mlmc_print_iteration_start(label, requested, capped)

Print a short message describing how many new samples were requested by the
allocator and how many will actually be drawn after capping. `label` may be an
integer iteration number or a descriptive string such as "Warmup".
"""
function mlmc_print_iteration_start(
    label,
    requested::AbstractVector{<:Integer},
    capped::AbstractVector{<:Integer},
)
    req = max.(requested, 0)
    plan = max.(capped, 0)
    println("MLMC $(label): requested $(req), running $(plan) additional samples at levels $(1:length(plan))")
end

"""
    mlmc_level_qoi_and_variance(accs, lvl, n_qois)

Return the current QoI estimate and variance for a specific level from the
accumulated MLMC statistics.
"""
function mlmc_level_qoi_and_variance(accs, lvl::Integer, n_qois::Integer)
    qoi_mean = [get_mean(accs[lvl][j]) for j in 1:n_qois]
    level_var = get_variance(accs[lvl][n_qois + 1])
    return qoi_mean, level_var
end

"""
    mlmc_print_iteration_summary(label, current_counts, cost_values, accs,
                                n_qois, requested, capped)

Print a PrettyTables summary of the current MLMC state after an iteration.
`label` may be an integer iteration number or a descriptive string such as
"Warmup".
"""
function mlmc_print_iteration_summary(
    label,
    current_counts::AbstractVector{<:Integer},
    cost_values::AbstractVector{<:Real},
    accs,
    n_qois::Integer,
    requested::AbstractVector{<:Integer},
    capped::AbstractVector{<:Integer},
)
    req = max.(requested, 0)
    planned = max.(capped, 0)
    n_levels = length(current_counts)

    summary = Matrix{Any}(undef, length(current_counts), 7)
    for lvl in 1:length(current_counts)
        qoi_mean, level_var = mlmc_level_qoi_and_variance(accs, lvl, n_qois)
        summary[lvl, 1] = lvl
        summary[lvl, 2] = current_counts[lvl]
        summary[lvl, 3] = cost_values[lvl]
        summary[lvl, 4] = length(qoi_mean) == 1 ? qoi_mean[1] : qoi_mean
        summary[lvl, 5] = level_var
        summary[lvl, 6] = req[lvl]
        summary[lvl, 7] = planned[lvl]
    end

    println("\nMLMC $(label) summary")
    PrettyTables.pretty_table(
        stdout,
        summary;
        column_labels = ["Level", "N", "Cost", "Level QoI", "Level variance", "Requested", "Running"],
        alignment = :l,
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
    )

    qoi_summary = Matrix{Any}(undef, n_qois, 3)
    for j in 1:n_qois
        mean_j = sum(get_mean(accs[lvl][j]) for lvl in 1:n_levels)
        variance_j = sum(get_variance(accs[lvl][j]) / max(current_counts[lvl], 1) for lvl in 1:n_levels)
        rmse_j = sqrt(max(variance_j, 0.0))
        qoi_summary[j, 1] = j
        qoi_summary[j, 2] = mean_j
        qoi_summary[j, 3] = rmse_j
    end

    println("QoI estimate ± RMSE")
    PrettyTables.pretty_table(
        stdout,
        qoi_summary;
        column_labels = ["QoI", "Mean", "RMSE"],
        alignment = :l,
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
    )
end
