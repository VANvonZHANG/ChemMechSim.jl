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
        # api.md (follow-up PR) renders all 81 exported docstrings on one page; lift
        # the default 200 KiB cap so strict mode does not fail the build on it.
        size_threshold = 400_000,
    ),
    pages = [
        "Home" => "index.md",
    ],
    # Documenter 1.19: doc-error classes are fatal by default (warnonly = Symbol[]);
    # `strict = true` of older versions is gone. This keeps markdown warnings fatal too:
    treat_markdown_warnings_as_error = true,
    checkdocs = :exports,   # exported-without-docstring = build failure
    # TEMPORARY until docs/src/api.md lands in a follow-up PR: missing_docs is fatal
    # only when every public docstring sits in an @docs block; api.md is what
    # satisfies that requirement.
    warnonly = [:missing_docs],
)

deploydocs(;
    repo = "github.com/VANvonZHANG/ChemMechSim.jl",
    devbranch = "main",
    push_preview = true,
)
