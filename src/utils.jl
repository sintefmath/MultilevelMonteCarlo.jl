import PrettyTables

"""
    mlmc_print_iteration_start(label, needed)

Print a short message describing how many new samples will be drawn at each
level in the next MLMC refinement step. `label` may be an integer iteration
number or a descriptive string such as "Warmup".
"""
function mlmc_print_iteration_start(label, needed::AbstractVector{<:Integer})
    planned = max.(needed, 0)
    println("MLMC $(label): running $(planned) additional samples at levels $(1:length(planned))")
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
                                n_qois, needed, variances)

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
    needed::AbstractVector{<:Integer},
    variances::AbstractVector{<:Real},
)
    planned = max.(needed, 0)
    current_estimate = [sum(get_mean(accs[lvl][j]) for lvl in 1:length(accs)) for j in 1:n_qois]
    total_estimator_variance = sum(variances[lvl] / current_counts[lvl] for lvl in 1:length(current_counts))

    summary = Matrix{Any}(undef, length(current_counts), 6)
    for lvl in 1:length(current_counts)
        qoi_mean, level_var = mlmc_level_qoi_and_variance(accs, lvl, n_qois)
        summary[lvl, 1] = lvl
        summary[lvl, 2] = current_counts[lvl]
        summary[lvl, 3] = cost_values[lvl]
        summary[lvl, 4] = length(qoi_mean) == 1 ? qoi_mean[1] : qoi_mean
        summary[lvl, 5] = level_var
        summary[lvl, 6] = planned[lvl]
    end

    println("\nMLMC $(label) summary")
    PrettyTables.pretty_table(
        stdout,
        summary;
        column_labels = ["Level", "N", "Cost", "Level QoI", "Level variance", "New samples"],
        alignment = :l,
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
    )
    println("Current QOI estimate: $(current_estimate)")
    println("Current estimator variance: $(total_estimator_variance)")
end
