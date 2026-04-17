"""
    case_calibrated(n, m)

Figure 1 calibrated case wrapper.
"""
case_calibrated(n::Int, m::Int) = generate_calibrated(n, m; seed=1)

"""
    case_type1_low_variance(n, m)

Figure 1a wrapper (underforecast variance).
"""
case_type1_low_variance(n::Int, m::Int) = generate_type1(n, m, 0.65; seed=2)

"""
    case_type1_high_variance(n, m)

Figure 1b wrapper (overforecast variance).
"""
case_type1_high_variance(n::Int, m::Int) = generate_type1(n, m, 1.35; seed=3)

"""
    case_type2_low_correlation(n, m)

Figure 1c wrapper (underforecast correlation).
"""
case_type2_low_correlation(n::Int, m::Int) = generate_type2(n, m, 0.45; seed=4)

"""
    case_type2_high_correlation(n, m)

Figure 1d wrapper (overforecast correlation).
"""
case_type2_high_correlation(n::Int, m::Int) = generate_type2(n, m, 0.75; seed=5)

"""
    case_type3_rotated(n, m)

Figure 1e wrapper (rotated forecast distribution).
"""
case_type3_rotated(n::Int, m::Int) = generate_type3(n, m, 20.0; seed=6)
