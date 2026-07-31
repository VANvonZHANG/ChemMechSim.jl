# Shared-CSE symbolic Jacobian helpers.
#
# This file starts with small, testable AST utilities used by the shared workspace
# Jacobian probe. The Jacobian expressions still come from ModelingToolkit/Symbolics;
# these helpers only rewrite generated-code temporaries into explicit workspace
# slots.

using SparseArrays

"_cse_symbol(sym) is true for SymbolicUtils generated CSE temporaries."
_cse_symbol(sym::Symbol) = startswith(String(sym), "##cse#")
_cse_symbol(_) = false

"Recursively replace CSE symbols in `ex` with `work[slot]` references."
function _rewrite_cse_refs(ex, slotmap::AbstractDict{Symbol,<:Integer}, work::Symbol)
    return _rewrite_refs(ex, _slot_refmap(slotmap, work), Dict{Symbol,Any}())
end

function _slot_refmap(slotmap::AbstractDict{Symbol,<:Integer}, workspace::Symbol)
    return Dict(sym => Expr(:ref, workspace, slot) for (sym, slot) in slotmap)
end

function _rewrite_refs(ex, cse_refmap::AbstractDict{Symbol}, env_refmap::AbstractDict{Symbol})
    if ex isa Symbol
        cse_ref = get(cse_refmap, ex, nothing)
        cse_ref !== nothing && return cse_ref

        env_ref = get(env_refmap, ex, nothing)
        env_ref !== nothing && return env_ref

        return ex
    elseif ex isa Expr
        return Expr(ex.head, (
            _rewrite_refs(arg, cse_refmap, env_refmap) for arg in ex.args
        )...)
    else
        return ex
    end
end

function _assignment_parts(ex)
    ex isa Expr || return nothing
    ex.head == :(=) || return nothing
    length(ex.args) == 2 || return nothing
    return ex.args[1], ex.args[2]
end

function _expr_node_count(ex)
    count = 0
    stack = Any[ex]
    while !isempty(stack)
        node = pop!(stack)
        count += 1
        if node isa Expr
            append!(stack, node.args)
        elseif node isa AbstractVector
            append!(stack, node)
        end
    end
    return count
end

function _collect_symbols!(found::Set{Symbol}, ex)
    stack = Any[ex]
    while !isempty(stack)
        node = pop!(stack)
        if node isa Symbol
            push!(found, node)
        elseif node isa Expr
            append!(stack, node.args)
        elseif node isa AbstractVector
            append!(stack, node)
        end
    end
    return found
end

_symbols_in(ex) = _collect_symbols!(Set{Symbol}(), ex)

function _needed_setup_prefix(setup_prefix, out_writes)
    defs = Dict{Symbol,Any}()
    for stmt in setup_prefix
        parts = _assignment_parts(stmt)
        parts === nothing && continue

        lhs, rhs = parts
        lhs isa Symbol || continue
        defs[lhs] = rhs
    end

    needed = Set{Symbol}()
    worklist = Symbol[]
    function mark_needed!(symbols)
        for sym in symbols
            if !(sym in needed)
                push!(needed, sym)
                push!(worklist, sym)
            end
        end
        return nothing
    end

    mark_needed!(_symbols_in(out_writes))
    # Non-CSE setup is preserved and executed, so keep its dependencies too.
    for stmt in setup_prefix
        parts = _assignment_parts(stmt)
        if parts === nothing
            mark_needed!(_symbols_in(stmt))
            continue
        end

        lhs, rhs = parts
        if !(lhs isa Symbol) || !_cse_symbol(lhs)
            lhs isa Symbol || mark_needed!(_symbols_in(lhs))
            mark_needed!(_symbols_in(rhs))
        end
    end

    def_deps = Dict{Symbol,Set{Symbol}}()
    while !isempty(worklist)
        sym = pop!(worklist)
        haskey(defs, sym) || continue
        deps = get!(def_deps, sym) do
            _symbols_in(defs[sym])
        end
        mark_needed!(deps)
    end

    return Any[
        stmt for stmt in setup_prefix
        if begin
            parts = _assignment_parts(stmt)
            parts === nothing && true ||
                !(parts[1] isa Symbol && _cse_symbol(parts[1])) ||
                parts[1] in needed
        end
    ]
end

function _chunk_by_node_budget(stmts, node_budget::Int)
    node_budget > 0 || throw(ArgumentError("node_budget must be positive"))
    return _chunk_by_count_and_node_budget(stmts, nothing, node_budget)
end

function _chunk_by_count_and_node_budget(stmts, count_budget::Union{Nothing,Int}, node_budget::Int;
                                         stmt_counts=nothing)
    count_budget === nothing || count_budget > 0 ||
        throw(ArgumentError("count_budget must be positive"))
    node_budget > 0 || throw(ArgumentError("node_budget must be positive"))
    if stmt_counts !== nothing && length(stmt_counts) != length(stmts)
        throw(ArgumentError("stmt_counts length must match stmts length"))
    end

    chunks = Vector{Any}[]
    current = Any[]
    current_count = 0
    current_nodes = 0
    for (i, stmt) in enumerate(stmts)
        stmt_count = stmt_counts === nothing ? 1 : stmt_counts[i]
        stmt_count >= 0 || throw(ArgumentError("stmt_counts must be nonnegative"))
        stmt_nodes = _expr_node_count(stmt)
        exceeds_count = count_budget !== nothing &&
            current_count > 0 &&
            current_count + stmt_count > count_budget
        exceeds_nodes = current_nodes + stmt_nodes > node_budget
        if !isempty(current) && (exceeds_count || exceeds_nodes)
            push!(chunks, current)
            current = Any[]
            current_count = 0
            current_nodes = 0
        end
        push!(current, stmt)
        current_count += stmt_count
        current_nodes += stmt_nodes
    end
    isempty(current) || push!(chunks, current)
    return chunks
end

function _cse_assignment_counts(setup_prefix)
    counts = Int[]
    for stmt in setup_prefix
        parts = _assignment_parts(stmt)
        is_cse_assignment = parts !== nothing && parts[1] isa Symbol && _cse_symbol(parts[1])
        push!(counts, is_cse_assignment ? 1 : 0)
    end
    return counts
end

function _rewrite_assignment_rhs_refs(ex, slotmap::AbstractDict{Symbol,<:Integer}, work::Symbol)
    parts = _assignment_parts(ex)
    parts === nothing && return _rewrite_cse_refs(ex, slotmap, work)
    lhs, rhs = parts
    return Expr(:(=), lhs, _rewrite_cse_refs(rhs, slotmap, work))
end

function _rewrite_setup_stmt(
    stmt,
    cse_refmap::AbstractDict{Symbol},
    env_refmap::AbstractDict{Symbol},
)
    parts = _assignment_parts(stmt)
    if parts === nothing
        return _rewrite_refs(stmt, cse_refmap, env_refmap)
    end

    lhs, rhs = parts
    rewritten_rhs = _rewrite_refs(rhs, cse_refmap, env_refmap)
    if lhs isa Symbol
        cse_ref = get(cse_refmap, lhs, nothing)
        cse_ref !== nothing && return Expr(:(=), cse_ref, rewritten_rhs)

        env_ref = get(env_refmap, lhs, nothing)
        env_ref !== nothing && return Expr(:(=), env_ref, rewritten_rhs)
    end

    return Expr(:(=), _rewrite_refs(lhs, cse_refmap, env_refmap), rewritten_rhs)
end

function _rewrite_setup_prefix(
    setup_prefix,
    cse_slotmap::AbstractDict{Symbol,<:Integer},
    work::Symbol,
    env_slotmap::AbstractDict{Symbol,<:Integer}=Dict{Symbol,Int}(),
    env::Symbol=gensym(:shared_cse_env),
)
    cse_refmap = _slot_refmap(cse_slotmap, work)
    env_refmap = _slot_refmap(env_slotmap, env)
    return Any[
        _rewrite_setup_stmt(stmt, cse_refmap, env_refmap)
        for stmt in setup_prefix
    ]
end

function _is_nzval_ref(ex, out_arg::Symbol)
    ex isa Expr || return false
    ex.head == :. || return false
    length(ex.args) == 2 || return false
    field = ex.args[2]
    return ex.args[1] == out_arg &&
        (field == QuoteNode(:nzval) || field == :nzval)
end

function _sparse_write_aliases(setup_prefix, out_arg::Symbol)
    nzval_aliases = Set{Symbol}()
    index_aliases = Dict{Symbol,Int}()

    for stmt in setup_prefix
        parts = _assignment_parts(stmt)
        parts === nothing && continue

        lhs, rhs = parts
        lhs isa Symbol || continue
        if _is_nzval_ref(rhs, out_arg)
            push!(nzval_aliases, lhs)
        elseif rhs isa Integer
            index_aliases[lhs] = Int(rhs)
        end
    end

    return (; out_arg, nzval_aliases, index_aliases)
end

function _direct_nzval_lhs(lhs, aliases)
    lhs isa Expr || return nothing
    lhs.head == :ref || return nothing
    length(lhs.args) == 2 || return nothing

    out_ref, idx_ref = lhs.args
    out_ref isa Symbol || return nothing
    out_ref in aliases.nzval_aliases || return nothing

    idx = idx_ref isa Symbol ? get(aliases.index_aliases, idx_ref, nothing) : idx_ref
    idx isa Integer || return nothing

    return Expr(:ref, Expr(:., aliases.out_arg, QuoteNode(:nzval)), Int(idx))
end

function _is_allowed_sparse_write_trailer(stmt, aliases)
    stmt isa Symbol && stmt in aliases.nzval_aliases && return true
    parts = _assignment_parts(stmt)
    if parts !== nothing
        lhs, rhs = parts
        lhs isa Symbol && rhs isa Symbol && rhs in aliases.nzval_aliases && return true
    end
    return false
end

function _rewrite_sparse_write(
    stmt,
    aliases,
    cse_slotmap::AbstractDict{Symbol,<:Integer},
    work::Symbol,
    env_slotmap::AbstractDict{Symbol,<:Integer}=Dict{Symbol,Int}(),
    env::Symbol=gensym(:shared_cse_env),
)
    cse_refmap = _slot_refmap(cse_slotmap, work)
    env_refmap = _slot_refmap(env_slotmap, env)
    return _rewrite_sparse_write_refmaps(stmt, aliases, cse_refmap, env_refmap)
end

function _rewrite_sparse_write_refmaps(
    stmt,
    aliases,
    cse_refmap::AbstractDict{Symbol},
    env_refmap::AbstractDict{Symbol},
)
    parts = _assignment_parts(stmt)
    parts === nothing && return _rewrite_refs(stmt, cse_refmap, env_refmap)

    lhs, rhs = parts
    direct_lhs = _direct_nzval_lhs(lhs, aliases)
    direct_lhs === nothing &&
        error("build_shared_cse_jac: unsupported sparse Jacobian write lhs: $(repr(lhs))")

    rewritten_rhs = _rewrite_refs(rhs, cse_refmap, env_refmap)
    return Expr(:(=), direct_lhs, rewritten_rhs)
end

function _arg_tuple_with_workspace(orig_arg_tuple, workspace_args::Symbol...)
    if orig_arg_tuple isa Expr && orig_arg_tuple.head == :tuple
        return Expr(:tuple, workspace_args..., orig_arg_tuple.args...)
    else
        return Expr(:tuple, workspace_args..., orig_arg_tuple)
    end
end

function _eval_shared_cse_chunk(orig_arg_tuple, workspace_args::Tuple{Vararg{Symbol}}, stmts)
    body = Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(0), Expr(:block, stmts...))
    fn_expr = Expr(:(->), _arg_tuple_with_workspace(orig_arg_tuple, workspace_args...), body)
    return Core.eval(@__MODULE__, fn_expr)
end

function _first_arg_symbol(orig_arg_tuple)
    arg = if orig_arg_tuple isa Expr && orig_arg_tuple.head == :tuple
        first(orig_arg_tuple.args)
    else
        orig_arg_tuple
    end
    arg isa Symbol || error("build_shared_cse_jac: unsupported output argument: $(repr(arg))")
    return arg
end

function _boolean_cse_rhs(rhs)
    rhs isa Bool && return true
    rhs isa Expr || return false
    rhs.head == :call || return false
    op = rhs.args[1]
    return string(op) in ("<", "<=", ">", ">=", "==", "!=", "isequal", "!")
end

struct SharedCSEJacStats
    n_setup_stmts::Int
    n_sparse_writes::Int
    n_skipped_sparse_trailers::Int
    n_cse_float::Int
    n_cse_bool::Int
    n_env_symbols::Int
    n_setup_chunks::Int
    n_write_chunks::Int
    cse_chunk_size::Int
    write_chunk_size::Int
    cse_node_budget::Int
    write_node_budget::Int
end

"Build an internal shared-CSE in-place Jacobian function + sparse prototype."
function build_shared_cse_jac(sys; cse_chunk_size::Int=50, write_chunk_size::Int=200,
                              cse_node_budget::Int=80_000,
                              write_node_budget::Int=80_000,
                              return_stats::Bool=false)
    cse_chunk_size > 0 || throw(ArgumentError("cse_chunk_size must be positive"))
    write_chunk_size > 0 || throw(ArgumentError("write_chunk_size must be positive"))
    cse_node_budget > 0 || throw(ArgumentError("cse_node_budget must be positive"))
    write_node_budget > 0 || throw(ArgumentError("write_node_budget must be positive"))

    jac_exprs = ModelingToolkit.generate_jacobian(sys;
        sparse=true, expression=Val{true}, wrap_gfw=Val{false}, cse=false)
    _jac_oop, jac_iip = jac_exprs

    blocks = _all_blocks(jac_iip)
    sort!(blocks; by=b -> length(b.args), rev=true)
    if length(blocks) < 2
        error("build_shared_cse_jac: could not locate out-writes block in jac_iip " *
              "(found $(length(blocks)) :block expression(s))")
    end

    main_body = blocks[1]
    out_writes_block = blocks[2]
    wrap_idx = findfirst(s -> _contains_block(s, out_writes_block), main_body.args)
    wrap_idx === nothing &&
        error("build_shared_cse_jac: out-writes block not found in main body")

    setup_prefix = main_body.args[1:wrap_idx-1]
    orig_arg_tuple = jac_iip.args[1]
    out_arg = _first_arg_symbol(orig_arg_tuple)
    out_writes = out_writes_block.args
    setup_prefix = _needed_setup_prefix(setup_prefix, out_writes)

    cse_assignments = Vector{Tuple{Symbol,Any}}()
    env_symbols = Symbol[]
    for stmt in setup_prefix
        parts = _assignment_parts(stmt)
        parts === nothing && continue

        lhs, rhs = parts
        lhs isa Symbol || continue
        if _cse_symbol(lhs)
            push!(cse_assignments, (lhs, rhs))
        elseif !(lhs in env_symbols)
            push!(env_symbols, lhs)
        end
    end

    bool_cse_symbols = Symbol[sym for (sym, rhs) in cse_assignments if _boolean_cse_rhs(rhs)]
    bool_cse_set = Set(bool_cse_symbols)
    float_cse_symbols = Symbol[sym for (sym, _rhs) in cse_assignments if !(sym in bool_cse_set)]
    float_slotmap = Dict(sym => i for (i, sym) in enumerate(float_cse_symbols))
    bool_slotmap = Dict(sym => i for (i, sym) in enumerate(bool_cse_symbols))
    env_slotmap = Dict(sym => i for (i, sym) in enumerate(env_symbols))
    float_work_arg = gensym(:shared_cse_float_work)
    bool_work_arg = gensym(:shared_cse_bool_work)
    env_arg = gensym(:shared_cse_env)
    cse_refmap = merge(_slot_refmap(float_slotmap, float_work_arg),
                       _slot_refmap(bool_slotmap, bool_work_arg))
    env_refmap = _slot_refmap(env_slotmap, env_arg)
    workspace_args = (float_work_arg, bool_work_arg, env_arg)

    rewritten_setup_stmts = Any[
        _rewrite_setup_stmt(stmt, cse_refmap, env_refmap)
        for stmt in setup_prefix
    ]
    setup_chunks = _chunk_by_count_and_node_budget(
        rewritten_setup_stmts,
        cse_chunk_size,
        cse_node_budget;
        stmt_counts=_cse_assignment_counts(setup_prefix),
    )
    setup_fns = Function[
        _eval_shared_cse_chunk(orig_arg_tuple, workspace_args, chunk)
        for chunk in setup_chunks
    ]

    aliases = _sparse_write_aliases(setup_prefix, out_arg)
    rewritten_out_writes = Any[]
    skipped_out_writes = Any[]
    for stmt in out_writes
        parts = _assignment_parts(stmt)
        if parts === nothing
            _is_allowed_sparse_write_trailer(stmt, aliases) ? push!(skipped_out_writes, stmt) :
                error("build_shared_cse_jac: unsupported sparse Jacobian write statement: $(repr(stmt))")
            continue
        end
        if _direct_nzval_lhs(parts[1], aliases) === nothing
            _is_allowed_sparse_write_trailer(stmt, aliases) ? push!(skipped_out_writes, stmt) :
                error("build_shared_cse_jac: unsupported sparse Jacobian write lhs: $(repr(parts[1]))")
            continue
        end
        push!(rewritten_out_writes,
              _rewrite_sparse_write_refmaps(stmt, aliases, cse_refmap, env_refmap))
    end
    write_chunks = _chunk_by_count_and_node_budget(
        rewritten_out_writes,
        write_chunk_size,
        write_node_budget,
    )
    write_fns = Function[
        _eval_shared_cse_chunk(orig_arg_tuple, workspace_args, chunk)
        for chunk in write_chunks
    ]

    J_sym = ModelingToolkit.calculate_jacobian(sys; sparse=true)
    if length(rewritten_out_writes) != length(SparseArrays.nonzeros(J_sym))
        error("build_shared_cse_jac: rewrote $(length(rewritten_out_writes)) sparse writes, " *
              "but Jacobian prototype has $(length(SparseArrays.nonzeros(J_sym))) nonzeros " *
              "(skipped $(length(skipped_out_writes)) trailer statements)")
    end

    float_workspaces = [Vector{Float64}(undef, length(float_cse_symbols))
                        for _ in 1:Threads.nthreads()]
    bool_workspaces = [Vector{Bool}(undef, length(bool_cse_symbols))
                       for _ in 1:Threads.nthreads()]
    envspaces = [Vector{Any}(undef, length(env_symbols))
                 for _ in 1:Threads.nthreads()]
    nsetup_chunks = length(setup_fns)
    nwrite_chunks = length(write_fns)
    function jac!(J::SparseArrays.SparseMatrixCSC, u, p, t)
        tid = Threads.threadid()
        float_work = float_workspaces[tid]
        bool_work = bool_workspaces[tid]
        env = envspaces[tid]
        @inbounds for ci in 1:nsetup_chunks
            Base.invokelatest(setup_fns[ci], float_work, bool_work, env, J, u, p, t)
        end
        @inbounds for wi in 1:nwrite_chunks
            Base.invokelatest(write_fns[wi], float_work, bool_work, env, J, u, p, t)
        end
        return nothing
    end

    J_proto = SparseArrays.similar(J_sym, Float64)
    stats = SharedCSEJacStats(
        length(setup_prefix),
        length(rewritten_out_writes),
        length(skipped_out_writes),
        length(float_cse_symbols),
        length(bool_cse_symbols),
        length(env_symbols),
        nsetup_chunks,
        nwrite_chunks,
        cse_chunk_size,
        write_chunk_size,
        cse_node_budget,
        write_node_budget,
    )
    return return_stats ? (jac!, J_proto, stats) : (jac!, J_proto)
end
