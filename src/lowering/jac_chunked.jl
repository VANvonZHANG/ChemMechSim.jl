# Chunked Jacobian codegen for large mechanisms where the full symbolic Jacobian
# function body OOMs the LLVM JIT (EarlyCSE pass). Splits the *generated iip
# function body* into chunks of `chunk_size` out-writes, evals each chunk as a
# separate closure (each carries the shared setup prefix — param unpacking +
# SymbolicUtils CSE temporaries — and a subset of the `out.nzval[i] = …`
# writes). The dispatcher `jac!` loops over chunk closures at call time to fill
# `J.nzval` in-place.
#
# Design (verified on GRI30, 2026-07-20):
#   - Each chunk = `(out, u, p, t) -> @inbounds begin <setup_prefix>; <chunk_out_writes> end`
#   - Each chunk's LLVM IR is small enough that EarlyCSE JITs cleanly (~25-30 s
#     per chunk on GRI30 at chunk_size=1000 nnz, ~1-2 GB peak RSS per chunk).
#   - Symbolic nonzeros are NOT passed through `build_function` individually:
#     doing so re-runs SymbolicUtils `toexpr` on each fully-expanded derivative
#     expression (386 MB of symbolic text for GRI30) and explodes. Instead we
#     reuse the already-CSE'd body that `generate_jacobian` produces.
#
# Reference: .superpowers/sdd/jac-oom-deep-analysis.md §3 Tier 2 (the design).
# Working probes: examples/oom_probe_fixD3.jl, oom_probe_chunk_sweep.jl,
# examples/diag_compare_to_baseline.jl (proves chunked output == full-jac output).

using SparseArrays

"Walk an Expr and collect every `:block` sub-expression, largest-first."
function _all_blocks(ex::Expr, found=Vector{Expr}())
    if ex.head == :block
        push!(found, ex)
    end
    for a in ex.args
        a isa Expr && _all_blocks(a, found)
    end
    return found
end

"Return true if `target` appears anywhere inside `ex` (recursive)."
function _contains_block(ex, target)
    ex === target && return true
    ex isa Expr || return false
    for a in ex.args
        _contains_block(a, target) && return true
    end
    return false
end

"Build a chunked in-place Jacobian function + sparse prototype for large mechanisms.
 Returns `(jac!, J_prototype)` for `ODEFunction(f; jac=jac!, jac_prototype=J_prototype)`.

 Strategy (the verified Tier-2 design):
   1. `generate_jacobian(sys; sparse=true, expression=Val{true}, wrap_gfw=Val{false}, cse=false)`
      → the iip Expr `(out, u, p, t) -> @inbounds begin <setup>; <wrapper>(<out-writes>) end`.
   2. Locate the inner `out-writes` block (2nd-largest `:block`, one statement per nnz).
   3. Partition the out-writes into chunks of `chunk_size` (count = nnz per chunk).
   4. For each chunk, emit a fresh closure with the same arg tuple as the original
      jac_iip, the full `setup_prefix`, and just that chunk's subset of out-writes.
   5. `eval` each chunk closure.
   6. Return a dispatcher that loops over chunk closures (via `invokelatest` —
      the chunks are eval'd at runtime, so callers in older worlds need this)
      and writes into `J.nzval` in-place.

 `chunk_size` = nonzero entries per chunk (default 2000 → 2 chunks on GRI30,
 ~5 chunks on FFCM2). Each chunk carries the full setup prefix (~37k stmts on
 GRI30, ~93k on FFCM2) so per-chunk JIT cost is roughly constant; pick
 chunk_size small enough that chunk count × per-chunk-JIT fits your time budget."
function build_chunked_jac(sys; chunk_size::Int=2000)
    # Step 1: get the iip Expr. cse=false is a no-op for SymbolicUtils' internal
    # CSE (##cse#N temps are still emitted) but matches the verified probes.
    jac_exprs = ModelingToolkit.generate_jacobian(sys;
        sparse=true, expression=Val{true}, wrap_gfw=Val{false}, cse=false)
    jac_oop, jac_iip = jac_exprs

    # Step 2: locate the main body block (largest) and the inner out-writes block
    # (2nd-largest — one statement per nnz, all of the form `misc3[miscN] = cseK`).
    blocks = _all_blocks(jac_iip)
    sort!(blocks; by=b -> length(b.args), rev=true)
    if length(blocks) < 2
        error("build_chunked_jac: could not locate out-writes block in jac_iip " *
              "(found $(length(blocks)) :block expression(s)); falling back to " *
              "non-chunked generate_jacobian is required.")
    end
    main_body = blocks[1]
    out_writes_block = blocks[2]

    # Step 3: split the main body into setup_prefix (everything before the wrapper
    # that contains the out-writes) and final_suffix (everything after — unused,
    # since the dispatcher doesn't need the original return value).
    wrap_idx = findfirst(s -> _contains_block(s, out_writes_block), main_body.args)
    wrap_idx === nothing && error("build_chunked_jac: out-writes block not found in main body")
    setup_prefix = main_body.args[1:wrap_idx-1]
    orig_arg_tuple = jac_iip.args[1]
    out_writes = out_writes_block.args
    n_out = length(out_writes)

    # Step 4: partition out_writes into chunks; eval each as a fresh closure.
    chunk_fns = Function[]
    for s in 1:chunk_size:n_out
        e = min(s + chunk_size - 1, n_out)
        chunk_writes = out_writes[s:e]
        body = Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(0),
                    Expr(:block, setup_prefix..., chunk_writes...))
        fn_expr = Expr(:(->), orig_arg_tuple, body)
        push!(chunk_fns, Core.eval(@__MODULE__, fn_expr))
    end

    # Step 5: dispatcher loops over chunk closures. `invokelatest` is required
    # because the chunk closures were just eval'd in this world; FBDF's compiled
    # integrator code lives in an older world and would otherwise hit a
    # world-age error on direct call. The per-call overhead (~1 µs) is
    # negligible vs the per-step linear solve (~ms).
    nchunks = length(chunk_fns)
    function jac!(J::SparseArrays.SparseMatrixCSC, u, p, t)
        @inbounds for ci in 1:nchunks
            Base.invokelatest(chunk_fns[ci], J, u, p, t)
        end
        return nothing
    end

    # J_prototype: Float64 SparseMatrixCSC with the same sparsity pattern as
    # J_sym. ODEFunction's jac_prototype requires a concrete numeric matrix.
    J_sym = ModelingToolkit.calculate_jacobian(sys; sparse=true)
    J_proto = SparseArrays.similar(J_sym, Float64)
    return jac!, J_proto
end
