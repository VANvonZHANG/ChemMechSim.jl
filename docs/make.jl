using ChemMechSim
using Documenter

const SITE = "https://vanvonzhang.github.io/ChemMechSim.jl"

makedocs(;
    modules = [ChemMechSim],
    authors = "Fan Zhang and contributors",
    sitename = "ChemMechSim.jl",
    format = Documenter.HTML(;
        canonical = SITE,
        edit_link = "main",
        # api.md renders all 81 exported docstrings on one page; lift the default
        # 200 KiB cap so strict mode does not fail the build on it.
        size_threshold = 400_000,
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
<<<<<<< HEAD
        "Ignition tutorial" => "tutorial_ignition.md",
        "Atmospheric box tutorial" => "tutorial_atmospheric.md",
        "Reactor modes" => "reactors.md",
        "Mechanism format" => "mechanism_format.md",
        "Rate-law extensibility" => "ratelaw_extensibility.md",
        "Validation" => "validation.md",
        "Performance" => "performance.md",
=======
        "Solver guide" => "solvers.md",
>>>>>>> 629f5e4 (docs(site): solver guide — FBDF, jac strategy, linsolve, tolerances, u0 completeness)
        "API reference" => "api.md",
    ],
    # Documenter 1.19: doc-error classes are fatal by default (warnonly = Symbol[]);
    # `strict = true` of older versions is gone. This keeps markdown warnings fatal too:
    treat_markdown_warnings_as_error = true,
    checkdocs = :exports,   # exported-without-docstring = build failure
    doctest = false,        # v1 non-goal
    linkcheck = false,      # v1 non-goal
)

deploydocs(;
    repo = "github.com/VANvonZHANG/ChemMechSim.jl",
    devbranch = "main",
    push_preview = true,
)
