using Documenter
using MultilevelMonteCarlo

makedocs(;
    modules = [MultilevelMonteCarlo],
    sitename = "MultilevelMonteCarlo.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://kjetil.github.io/MultilevelMonteCarlo.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => [
            "Projectile Motion" => "examples/projectile.md",
            "Sample Storage & NetCDF" => "examples/sample_storage.md",
            "PDF & CDF Estimation" => "examples/pdf_estimation.md",
            "Rank Histograms" => "examples/rank_histograms.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/kjetil/MultilevelMonteCarlo.jl",
    devbranch = "main",
)
