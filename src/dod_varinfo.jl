# Data-Oriented Design VarInfo implementation.
#
# This file introduces a new `VarMeta` and `DODVarInfo` which store variable
# values grouped by their concrete Julia type to allow type-specialized storage.
# It provides a small compatibility surface so existing code that queries
# `getmetadata`, `getidx`, `getrange`, `getdist`, `getindex_internal`, `istrans`,
# and flag manipulation can work with `DODVarInfo` where appropriate.

# using DynamicPPL: VarName

struct VarMeta
    name::VarName
    type::DataType
    idx::Int # index within the type-vector
    len::Int # length (number of scalar entries)
    transformed::Bool
    dist::Any
    flags::Dict{Symbol,Bool}
    vec::Any  # cached reference to the backing Vector for this meta (or nothing)
end

# Small mutable holder for common accumulators used by the DOD fast-path.
# Make this parametric so the holder's fields have concrete types and
# `getproperty`/field access can be inlined by the compiler.

mutable struct MutableAccHolder{LPT, LJT, LLT}
    lp::LPT
    lj::LJT
    ll::LLT
end

# Typed callable updater to avoid closure overhead. Parameterized on which
# accumulate_*_inplace!! methods are available so the compiler can
# specialize and inline accordingly.

struct AccUpdater{LP_ASSUME, LJ_ASSUME, LL_ASSUME, LP_OBS, LJ_OBS, LL_OBS, LPT, LJT, LLT}
        holder::MutableAccHolder{LPT,LJT,LLT}
end

# Top-level specialized updater methods so Julia can compile them. These
# are parameterized on the AccUpdater type parameters so the compiler can
# emit optimized code for each combination.
function acc_updater_assume(u::AccUpdater{LP_ASSUME,LJ_ASSUME,LL_ASSUME,LP_OBS,LJ_OBS,LL_OBS,LPT,LJT,LLT}, vi::Any, val, logjac, vn, right) where {LP_ASSUME,LJ_ASSUME,LL_ASSUME,LP_OBS,LJ_OBS,LL_OBS,LPT,LJT,LLT}
    h = u.holder
    if LP_ASSUME
        accumulate_assume_inplace!!(h.lp, val, logjac, vn, right)
    else
        h.lp = accumulate_assume!!(h.lp, val, logjac, vn, right)
    end

    if LJ_ASSUME
        accumulate_assume_inplace!!(h.lj, val, logjac, vn, right)
    else
        h.lj = accumulate_assume!!(h.lj, val, logjac, vn, right)
    end

    if LL_ASSUME
        accumulate_assume_inplace!!(h.ll, val, logjac, vn, right)
    else
        h.ll = accumulate_assume!!(h.ll, val, logjac, vn, right)
    end
    return vi
end

function acc_updater_observe(u::AccUpdater{LP_ASSUME,LJ_ASSUME,LL_ASSUME,LP_OBS,LJ_OBS,LL_OBS,LPT,LJT,LLT}, vi::Any, right, left, vn) where {LP_ASSUME,LJ_ASSUME,LL_ASSUME,LP_OBS,LJ_OBS,LL_OBS,LPT,LJT,LLT}
    h = u.holder
    if LP_OBS
        accumulate_observe_inplace!!(h.lp, right, left, vn)
    else
        h.lp = accumulate_observe!!(h.lp, right, left, vn)
    end

    if LJ_OBS
        accumulate_observe_inplace!!(h.lj, right, left, vn)
    else
        h.lj = accumulate_observe!!(h.lj, right, left, vn)
    end

    if LL_OBS
        accumulate_observe_inplace!!(h.ll, right, left, vn)
    else
        h.ll = accumulate_observe!!(h.ll, right, left, vn)
    end
    return vi
end

# A lightweight view over an existing AccumulatorTuple that forwards the three
# common accumulator names to a MutableAccHolder. This avoids rebuilding
# a NamedTuple on every update when the holder is used by the DOD fast-path.
struct AccumulatorView
    base_nt::NamedTuple
    holder::MutableAccHolder
end

Base.haskey(av::AccumulatorView, ::Val{accname}) where {accname} = haskey(av.base_nt, accname)
function Base.getindex(av::AccumulatorView, ::Val{accname}) where {accname}
    # Three special-case names are served from the holder to reflect in-place
    # updates; fall back to the base NamedTuple otherwise.
    @inline if accname === :LogPrior
        return av.holder.lp
    elseif accname === :LogJacobian
        return av.holder.lj
    elseif accname === :LogLikelihood
        return av.holder.ll
    else
        return av.base_nt[Val(accname)]
    end
end

Base.keys(av::AccumulatorView) = keys(av.base_nt)
Base.length(av::AccumulatorView) = length(av.base_nt)
Base.iterate(av::AccumulatorView, s...) = iterate(av.base_nt, s...)
Base.copy(av::AccumulatorView) = begin
    # materialize a concrete AccumulatorTuple from the view
    nt = NamedTuple{Tuple(keys(av.base_nt))}(map(k->getindex(av, Val(k)), keys(av.base_nt)))
    return AccumulatorTuple(nt)
end

# Provide a `map` method so callers like `map(reset, getaccs(vi))` behave like
# `map` on `AccumulatorTuple` and return an `AccumulatorTuple`.
function Base.map(func::Function, av::AccumulatorView)
    names = Tuple(keys(av.base_nt))
    vals = Tuple(func(getindex(av, Val(k))) for k in names)
    nt = NamedTuple{names}(vals)
    return AccumulatorTuple(nt)
end

# API compatibility: provide getacc so callers that expect the AccumulatorTuple
# API can work with the view.
function getacc(av::AccumulatorView, ::Val{accname}) where {accname}
    return getindex(av, Val{accname}())
end


# Conversion helpers from existing Metadata/VarNamedVector shapes
using ..DynamicPPL: Metadata, VarNamedVector, getidx, getrange, getindex_internal, getdist, keys, accumulate_assume!!, accumulate_observe!!

function DODVarInfo_from_Metadata(md::Metadata, accs=nothing)
    vi = DODVarInfo(accs)
    # iterate over metadata.vns and push values
    for vn in md.vns
        val = getindex_internal(md, vn)
        meta = add_variable!(vi, vn, val; transformed=false, dist=getdist(md, vn))
        # copy flags
        for (k, bv) in md.flags
            if bv[getidx(md, vn)]
                meta.flags[k] = true
            end
        end
    end
    return vi
end

function DODVarInfo_from_VarNamedVector(vnv::VarNamedVector, accs=nothing)
    vi = DODVarInfo(accs)
    for vn in keys(vnv.idcs)
        val = getindex_internal(vnv, vn)
        add_variable!(vi, vn, val; transformed=vnv.is_unconstrained[getidx(vnv, vn)], dist=vnv.transforms[getidx(vnv, vn)])
    end
    return vi
end

"""A VarInfo which stores values grouped by concrete element type."""
# A no-op updater that implements the same API as `AccUpdater` but performs no actions.
struct NoopAccUpdater end

mutable struct DODVarInfo{AU} <: AbstractVarInfo
    values::Dict{DataType, Vector}           # type => vector of values of that type
    metas::Vector{VarMeta}                   # metadata for each VarName (order of discovery)
    idcs::Dict{VarName, Int}                 # maps VarName -> index into metas
    accumulators::Any                        # keep existing accumulators opaque (AccumulatorTuple expected)
    acc_cache::Any                           # small cache for accumulator presence checks (or nothing)
    acc_keys::Any                            # cached tuple of accumulator key symbols (or nothing)
    acc_default::Any                         # optional fast-path container for common accumulators
    acc_updater::AU                          # cached updater (always concrete type AU)
    meta_getters::Any                        # optional tuple of getter closures specialized per-meta
    meta_setters::Any                        # optional tuple of setter closures specialized per-meta
    typed_slots::Any                         # optional per-model concrete typed storage (Vector of Vectors)
    allocator_cursor::Any                    # per-meta next-free cursor (Vector{Int}) or nothing
    allocator_freelist::Any                  # per-meta free-list of (start indices) (Vector{Vector{Int}}) or nothing
end

function DODVarInfo(accs=nothing)
    accs === nothing && (accs = default_accumulators())
    # default to a NoopAccUpdater when constructed without a ModelDOD
    noop = NoopAccUpdater()
    vi = DODVarInfo{typeof(noop)}(Dict{DataType, Vector}(), VarMeta[], Dict{VarName, Int}(), accs, nothing, nothing, nothing, noop, nothing, nothing, nothing, nothing, nothing)
    vi = update_acc_cache!(vi)
    return vi
end

# Construct a DODVarInfo pre-seeded for a ModelDOD: create placeholder metas
# for each VarName mentioned in the model so generated code can index into
# the metas by literal integers.
function DODVarInfo(m::DynamicPPL.ModelDOD, accs=nothing)
    accs === nothing && (accs = default_accumulators())
    metas = Vector{VarMeta}(undef, length(m.meta_vns))
    idcs = Dict{VarName, Int}()
    for (i, vn) in enumerate(m.meta_vns)
    metas[i] = VarMeta(vn, Any, 0, 0, false, nothing, Dict{Symbol,Bool}(), nothing)
        idcs[vn] = i
    end
    # Initialize typed_slots from model-provided meta_types and meta_lens where available.
    n = length(metas)
    typed_slots = Vector{Any}(undef, n)
    allocator_cursor = Vector{Int}(undef, n)
    # per-meta freelist of (start_index, length) tuples
    allocator_freelist = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    mt = hasproperty(m, :meta_types) ? getfield(m, :meta_types) : nothing
    ml = hasproperty(m, :meta_lens) ? getfield(m, :meta_lens) : nothing
    for i in 1:n
        eltype_i = (mt !== nothing && length(mt) >= i) ? mt[i] : Any
        len_i = (ml !== nothing && length(ml) >= i) ? ml[i] : 1
        # create an empty Vector{eltype_i}
        try
            # If a length hint is available we can fully allocate the slot now.
            if len_i > 0
                typed_slots[i] = Vector{eltype_i}(undef, len_i)
            else
                typed_slots[i] = Vector{eltype_i}()
            end
        catch
            # fallback to Any if we cannot parametrize
            typed_slots[i] = Vector{Any}()
        end
        # reserve capacity if length hint available
        # If we preallocated the vector above, set the placeholder meta to point
        # at the preallocated backing vector with an initial idx of 1 so codegen
        # and generated model code can index into metas by integer literals.
        if len_i > 0 && length(typed_slots[i]) == len_i
            # leave contents uninitialized; set meta placeholder below
            nothing
        end
    end
    # Create a placeholder DODVarInfo with acc_updater set to `nothing` first,
    # then attempt to create a concrete acc_default and acc_updater and return
    # a parametric DODVarInfo typed on the updater when possible.
    # Initialize allocator helpers
    for i in 1:n
        allocator_cursor[i] = typed_slots[i] === nothing ? 0 : (length(typed_slots[i]) > 0 ? 1 : 0)
    allocator_freelist[i] = Vector{Tuple{Int,Int}}()
        if typed_slots[i] !== nothing && length(typed_slots[i]) > 0
            metas[i] = VarMeta(metas[i].name, eltype(typed_slots[i]), 0, 0, metas[i].transformed, metas[i].dist, metas[i].flags, typed_slots[i])
        end
    end

    tmp = DODVarInfo{Any}(Dict{DataType, Vector}(), metas, idcs, accs, nothing, nothing, nothing, nothing, nothing, nothing, typed_slots, allocator_cursor, allocator_freelist)
    # Build acc_default and acc_updater aggressively if accumulators present
    accs_nt = accs isa AccumulatorTuple ? accs.nt : nothing
    if accs_nt !== nothing
        lp_ok = haskey(accs_nt, :LogPrior)
        lj_ok = haskey(accs_nt, :LogJacobian)
        ll_ok = haskey(accs_nt, :LogLikelihood)
        if lp_ok || lj_ok || ll_ok
            holder = MutableAccHolder(
                haskey(accs_nt, :LogPrior) ? accs_nt.LogPrior : nothing,
                haskey(accs_nt, :LogJacobian) ? accs_nt.LogJacobian : nothing,
                haskey(accs_nt, :LogLikelihood) ? accs_nt.LogLikelihood : nothing,
            )
            # create concrete updater
            updater = make_accumulator_updater_from_holder(holder)
            # construct DODVarInfo parameterized on the updater type
            vi = DODVarInfo{typeof(updater)}(Dict{DataType, Vector}(), metas, idcs, AccumulatorView(accs_nt, holder), nothing, fieldnames(typeof(accs_nt)), holder, updater, nothing, nothing, typed_slots, allocator_cursor, allocator_freelist)
            # wire typed_slots into metas
            for i in 1:length(metas)
                if typed_slots[i] !== nothing && vi.metas[i].vec === nothing
                    vi.metas[i] = VarMeta(vi.metas[i].name, vi.metas[i].type, vi.metas[i].idx, vi.metas[i].len, vi.metas[i].transformed, vi.metas[i].dist, vi.metas[i].flags, typed_slots[i])
                end
            end
            return vi
        end
    end
    # Fallback: return tmp (with Any updater)
    vi = tmp
    # Build per-meta getter and setter closures specialized to each meta index.
    n = length(metas)
    if n > 0
            getters = ntuple(i -> begin
            # each getter only captures the integer `i` (no `vi` capture)
            function (vi::DODVarInfo{AU}) where {AU}
                meta = vi.metas[i]
                meta.idx == 0 && throw(KeyError(meta.name))
                vec = meta.vec === nothing ? vi.values[meta.type] : meta.vec
                return meta.len == 1 ? vec[meta.idx] : vec[meta.idx:(meta.idx + meta.len - 1)]
            end
        end, n)

            setters = ntuple(i -> begin
            function (vi::DODVarInfo{AU}, value) where {AU}
                meta = vi.metas[i]
                if meta.idx == 0
                    add_variable!(vi, meta.name, value; transformed=meta.transformed, dist=meta.dist)
                    return vi
                end
                vec = meta.vec === nothing ? vi.values[meta.type] : meta.vec
                if meta.len == 1
                    vec[meta.idx] = value
                else
                    vec[meta.idx:(meta.idx + meta.len - 1)] = value
                end
                return vi
            end
        end, n)
    vi.meta_getters = getters
    vi.meta_setters = setters
    else
        vi.meta_getters = nothing
        vi.meta_setters = nothing
    end
    vi = update_acc_cache!(vi)
    # Wire typed_slots as backing storage for placeholder metas when available.
    # (metas already wired above for preallocated typed_slots)
    return vi
end

# getaccs / setaccs!! expected by AbstractVarInfo
getaccs(vi::DODVarInfo{AU}) where {AU} = vi.accumulators

# Accept an AccumulatorTuple directly
function setaccs!!(vi::DODVarInfo{AU}, accs::AccumulatorTuple) where {AU}
    vi.accumulators = accs
    vi = update_acc_cache!(vi)
    return vi
end

# Accept a tuple of AbstractAccumulator (the high-level API used elsewhere)
function setaccs!!(vi::DODVarInfo{AU}, accs::NTuple{N,AbstractAccumulator}) where {AU,N}
    return setaccs!!(vi, AccumulatorTuple(accs))
end

# Optimized reset for DODVarInfo to avoid constructing intermediate
# NamedTuples when possible. Fast-paths the common case where
# `vi.accumulators` is an `AccumulatorView` whose only keys are the three
# common accumulators (LogPrior, LogJacobian, LogLikelihood) and those
# are held in the `MutableAccHolder` so they can be reset in-place.
function resetaccs!!(vi::DODVarInfo{AU}) where {AU}
    accs = vi.accumulators
    # Fast-path when we have an AccumulatorView backed by a holder.
    if accs isa AccumulatorView
        nt = accs.base_nt
        holder = accs.holder
        # Prefer a cached tuple of accumulator key symbols when available to
        # avoid calling `fieldnames(typeof(nt))` at runtime. Fall back to
        # `fieldnames` if the cache isn't present for some reason.
        names = vi.acc_keys === nothing ? fieldnames(typeof(nt)) : vi.acc_keys
        # common three-name fast-path predicate
        only_common = all(k -> (k === :LogPrior || k === :LogJacobian || k === :LogLikelihood), names)
        if only_common
            # reset holder fields in-place where present and avoid rebuilding
            # the NamedTuple since the base_nt already contains any non-holder
            # accumulators (there are none in this branch).
            if holder.lp !== nothing
                holder.lp = reset(holder.lp)
            end
            if holder.lj !== nothing
                holder.lj = reset(holder.lj)
            end
            if holder.ll !== nothing
                holder.ll = reset(holder.ll)
            end
            # keep the AccumulatorView wrapper (base_nt unchanged)
            vi.accumulators = AccumulatorView(nt, holder)
            return vi
        else
            # Mixed case: some accumulators are holder-backed, others are only
            # present in the NamedTuple. Build a new NamedTuple with reset
            # values for each key without allocating intermediate arrays via
            # `map`/`Tuple(generator)`. Do this by constructing a plain NT
            # value vector and materializing the NamedTuple directly.
            vals = Vector{Any}(undef, length(names))
            for (j, k) in enumerate(names)
                if k === :LogPrior
                    if holder.lp !== nothing
                        holder.lp = reset(holder.lp)
                    end
                    vals[j] = holder.lp
                elseif k === :LogJacobian
                    if holder.lj !== nothing
                        holder.lj = reset(holder.lj)
                    end
                    vals[j] = holder.lj
                elseif k === :LogLikelihood
                    if holder.ll !== nothing
                        holder.ll = reset(holder.ll)
                    end
                    vals[j] = holder.ll
                else
                    vals[j] = reset(nt[Val(k)])
                end
            end
            # materialize NamedTuple with the computed values
            nt2 = NamedTuple{names}(Tuple(vals))
            return setaccs!!(vi, AccumulatorTuple(nt2))
        end
    elseif accs isa AccumulatorTuple
        # Fallback: use existing generic behaviour but keep the fast-path at
        # this call site so DODVarInfo is handled efficiently.
        return setaccs!!(vi, map(reset, getaccs(vi)))
    else
        return setaccs!!(vi, map(reset, getaccs(vi)))
    end
end

# VarInfo-like transformation API
transformation(::DODVarInfo{AU}) where {AU} = DynamicTransformation()

# Transformation helpers similar to SimpleVarInfo
from_internal_transform(::DODVarInfo{AU}, ::VarName) where {AU} = identity
from_internal_transform(::DODVarInfo{AU}, ::VarName, dist) where {AU} = identity
from_linked_internal_transform(::DODVarInfo{AU}, ::VarName) where {AU} = identity
function from_linked_internal_transform(::DODVarInfo{AU}, ::VarName, dist) where {AU}
    return invlink_transform(dist)
end

function add_variable!(vi::DODVarInfo{AU}, vn::VarName, val; transformed=false, dist=nothing) where {AU}
    # If this VarName has a preallocated meta slot (from ModelDOD), use it
    if haskey(vi.idcs, vn)
        i = vi.idcs[vn]
        meta = vi.metas[i]
            if meta.idx == 0
                # If a concrete backing vector (meta.vec) exists and was preallocated by
                # ModelDOD, allocate a contiguous region for this value rather than
                # appending to the global type-vectors. This avoids type-key lookups and
                # repeated allocations in hot paths.
                if meta.vec !== nothing
                    len = isa(val, AbstractVector) ? length(val) : 1
                    idx = allocate_meta_slot!(vi, i, len)
                    vec = meta.vec
                    if len == 1
                        vec[idx] = val
                    else
                        vec[idx:(idx + len - 1)] = val
                    end
                    vi.metas[i] = VarMeta(vn, eltype(vec), idx, len, transformed, dist, Dict{Symbol,Bool}(), vec)
                    return vi.metas[i]
                end
            end
        # otherwise fall through to the generic path below which will create
        # a new typed bucket
    end
    # Fallback: create a new meta and append
    T = eltype(val) isa Type ? eltype(val) : typeof(val)
    # ensure a container for T
    if !haskey(vi.values, T)
        vi.values[T] = Vector{T}()
    end
    vec = vi.values[T]
    idx = length(vec) + 1
    # push value(s) - if val is AbstractVector, append its elements, otherwise push scalar
    if isa(val, AbstractVector)
        append!(vec, val)
        len = length(val)
    else
        push!(vec, val)
        len = 1
    end
    meta = VarMeta(vn, T, idx, len, transformed, dist, Dict{Symbol,Bool}(), vec)
    push!(vi.metas, meta)
    vi.idcs[vn] = length(vi.metas)
    return meta
end

# Allocate a contiguous region inside a preallocated backing vector for a meta.
# This uses per-meta freelists and a cursor stored on the DODVarInfo to
# reuse freed regions when possible and to advance the cursor otherwise.
function allocate_meta_slot!(vi::DODVarInfo, meta_i::Int, len::Int=1)
    meta = vi.metas[meta_i]
    vec = meta.vec
    if vec === nothing
        throw(ArgumentError("allocate_meta_slot! called for meta without preallocated vec"))
    end
    # initialize allocator helpers if missing
    if vi.allocator_freelist === nothing
        vi.allocator_freelist = [Vector{Tuple{Int,Int}}() for _ in 1:length(vi.metas)]
    end
    if vi.allocator_cursor === nothing
        # default cursor: next free index is 1 if vec has capacity > 0, else 0
        vi.allocator_cursor = [ (vi.metas[i].vec !== nothing && length(vi.metas[i].vec) > 0) ? 1 : 0 for i in 1:length(vi.metas) ]
    end

    fl = vi.allocator_freelist[meta_i]
    if fl !== nothing && !isempty(fl)
        # first-fit search for a block with sufficient length
        for j in 1:length(fl)
            s, blen = fl[j]
            if blen >= len
                # swap-remove the block to avoid O(n) shifting from deleteat!
                if j != length(fl)
                    fl[j] = fl[end]
                end
                pop!(fl)
                # if block larger than needed, push back remainder using O(1) append
                if blen > len
                    push!(fl, (s + len, blen - len))
                end
                return s
            end
        end
    end

    cur = vi.allocator_cursor[meta_i]
    if cur == 0
        cur = 1
    end
    needed = cur + len - 1
    if length(vec) < needed
        # exponential growth to reduce repeated resize! calls
        oldlen = length(vec)
        # choose growth factor; ensure we at least satisfy `needed`
        grow = oldlen > 0 ? max(needed, Int(ceil(oldlen * 1.5))) : max(needed, 16)
        resize!(vec, grow)
    end
    # advance cursor to next free position
    vi.allocator_cursor[meta_i] = cur + len
    return cur
end

function delete_variable!(vi::DODVarInfo{AU}, vn::VarName) where {AU}
    i = vi.idcs[vn]
    meta = vi.metas[i]
    # NOTE: we do not compact values in the backing vector for speed; mark as deleted
    # If this meta used a preallocated backing vector, record the freed start
    # index so future allocations can reuse the slot.
    if meta.vec !== nothing && meta.idx != 0
        # initialize freelist if missing
        if vi.allocator_freelist === nothing
            vi.allocator_freelist = [Vector{Tuple{Int,Int}}() for _ in 1:length(vi.metas)]
        end
        # insert freed block and merge with adjacent blocks to avoid fragmentation
        fl = vi.allocator_freelist[i]
        s = meta.idx
        l = meta.len
        # try to merge with any existing block that touches this one
        merged = false
        for j in 1:length(fl)
            s2, l2 = fl[j]
            # adjacent on left
            if s2 + l2 == s
                fl[j] = (s2, l2 + l)
                merged = true
                s, l = fl[j]
                break
            # adjacent on right
            elseif s + l == s2
                fl[j] = (s, l + l2)
                merged = true
                s, l = fl[j]
                break
            end
        end
        if !merged
            push!(fl, (s, l))
        end
    end
    delete!(vi.idcs, vn)
    # Preserve the backing vector pointer (meta.vec) so ModelDOD preallocated
    # storage remains available for future allocations.
    vi.metas[i] = VarMeta(meta.name, meta.type, 0, 0, false, nothing, Dict{Symbol,Bool}(), meta.vec)
    return meta
end

get_variable(vi::DODVarInfo{AU}, vn::VarName) where {AU} = begin
    i = vi.idcs[vn]
    meta = vi.metas[i]
    if meta.idx == 0
        throw(KeyError(vn))
    end
    vec = vi.values[meta.type]
    if meta.len == 1
        return vec[meta.idx]
    else
        return vec[meta.idx:(meta.idx + meta.len - 1)]
    end
end

# Use add_variable! for adding values; avoid extending external `push!!` here.

function haskey(vi::DODVarInfo{AU}, vn::VarName) where {AU}
    return haskey(vi.idcs, vn)
end

# getindex_internal compatible implementation: return stored value for vn or Colon
getindex_internal(vi::DODVarInfo{AU}, vn::VarName) where {AU} = get_variable(vi, vn)
function getindex_internal(vi::DODVarInfo{AU}, ::Colon) where {AU}
    # Preallocate total length to avoid intermediate allocations from vcat
    total = 0
    for m in vi.metas
        total += (m.idx != 0) ? m.len : 0
    end
    out = Vector{Any}(undef, total)
    pos = 1
    for m in vi.metas
        if m.idx != 0
            val = get_variable(vi, m.name)
            if m.len == 1
                out[pos] = val
                pos += 1
            else
                for x in val
                    out[pos] = x
                    pos += 1
                end
            end
        end
    end
    return out
end

function getdist(vi::DODVarInfo{AU}, vn::VarName) where {AU}
    i = vi.idcs[vn]
    return vi.metas[i].dist
end

function settrans!!(vi::DODVarInfo{AU}, trans::Bool, vn::VarName) where {AU}
    i = vi.idcs[vn]
    vi.metas[i].transformed = trans
    return vi
end

istrans(vi::DODVarInfo{AU}) where {AU} = any(m -> m.transformed, vi.metas)

Base.keys(vi::DODVarInfo{AU}) where {AU} = (m.name for m in vi.metas if m.idx != 0)

function vector_length(vi::DODVarInfo{AU}) where {AU}
    # sum of scalar lengths
    return sum(m.len for m in vi.metas if m.idx != 0)
end

# --- DOD-specific helpers and tilde handlers ---

# Helper: get VarMeta by meta-index (index into `vi.metas`). This allows
# codegen to refer to variables by their meta slot instead of VarName.
get_meta(vi::DODVarInfo{AU}, meta_i::Int) where {AU} = vi.metas[meta_i]

# Sampler-aware `assume` methods for DODVarInfo so `context_implementations.assume`
# will dispatch correctly when given a `DODVarInfo`.
function assume(
    rng::Random.AbstractRNG,
    sampler::Union{SampleFromPrior,SampleFromUniform},
    dist::Distribution,
    vn::VarName,
    vi::DODVarInfo,
)
    meta_i = get(vi.idcs, vn, 0)
    if meta_i == 0
        # fallback to the VarName-based DOD wrapper (handles lazy add)
        return tilde_assume_dod!!(rng, dist, vn, vi)
    else
        return tilde_assume_dod!!(rng, sampler, dist, meta_i, vi)
    end
end

function assume(sampler::Union{SampleFromPrior,SampleFromUniform}, dist::Distribution, vn::VarName, vi::DODVarInfo)
    meta_i = get(vi.idcs, vn, 0)
    if meta_i == 0
        return tilde_assume_dod!!(sampler, dist, vn, vi)
    else
        return tilde_assume_dod!!(sampler, dist, meta_i, vi)
    end
end

function assume(dist::Distribution, vn::VarName, vi::DODVarInfo)
    meta_i = get(vi.idcs, vn, 0)
    if meta_i == 0
        return tilde_assume_dod!!(dist, vn, vi)
    else
        return tilde_assume_dod!!(dist, meta_i, vi)
    end
end

# Helper: read stored (internal) value by meta index. Avoids dictionary lookups.
@inline function get_value_by_meta(vi::DODVarInfo, meta_i::Int)
    if vi.meta_getters !== nothing && meta_i <= length(vi.meta_getters)
        return vi.meta_getters[meta_i](vi)
    end
    meta = vi.metas[meta_i] 
    meta.idx == 0 && throw(KeyError(meta.name))
    vec = meta.vec === nothing ? vi.values[meta.type] : meta.vec
    return meta.len == 1 ? vec[meta.idx] : vec[meta.idx:(meta.idx + meta.len - 1)]
end

# Helper: write stored (internal) value by meta index.
@inline function set_value_by_meta!(vi::DODVarInfo, meta_i::Int, value)
    if vi.meta_setters !== nothing && meta_i <= length(vi.meta_setters)
        return vi.meta_setters[meta_i](vi, value)
    end
    meta = vi.metas[meta_i] 
    # If the variable has been deleted, fall back to add
    if meta.idx == 0
        add_variable!(vi, meta.name, value; transformed=meta.transformed, dist=meta.dist)
        return vi
    end
    vec = meta.vec === nothing ? vi.values[meta.type] : meta.vec
    if meta.len == 1
        vec[meta.idx] = value
    else
        vec[meta.idx:(meta.idx + meta.len - 1)] = value
    end
    return vi
end

# Primary fast assume/observe entrypoints that accept a meta-index. The idea is
# that `@model_dod` can generate calls with this integer meta index so that the
# inner loop avoids VarName dictionary lookups.
@inline function tilde_assume_dod!!(::Any, right, meta_i::Int, vi::DODVarInfo{AU}) where {AU}
    meta = get_meta(vi, meta_i)
    vn = meta.name
    # If this meta is a placeholder (idx == 0) perform the lazy-add path and
    # when the meta placeholder has no backing storage yet (idx == 0) we need
    # to materialize a concrete value. Previously this returned `nothing` and
    # deferred the add which caused downstream uses of the variable to see
    # `nothing` (e.g. `Normal(nothing, 1)`). Instead, sample from the prior
    # (equivalent to the default non-sampler behaviour) and add the variable.
    if meta.idx == 0
        # delegate to the sampler-aware variant using the default RNG and
        # `SampleFromPrior()` so we produce a concrete value and update `vi`.
        return tilde_assume_dod!!(Random.default_rng(), SampleFromPrior(), right, meta_i, vi)
    end
    # Fallback assume (no sampler provided) - replicate semantics from `assume(dist, vn, vi)`
    y = get_value_by_meta(vi, meta_i)
    f = from_maybe_linked_internal_transform(vi, vn, right)
    x, inv_logjac = with_logabsdet_jacobian(f, y)
        vi = accumulate_assume_dod!!(vi, x, -inv_logjac, vn, right)
    return x, vi
end

# Allow calling tilde_assume_dod!! with just (dist, meta_i, vi) for compatibility
function tilde_assume_dod!!(right::Distribution, meta_i::Int, vi::DODVarInfo{AU}) where {AU}
    return tilde_assume_dod!!(nothing, right, meta_i, vi)
end



# Sampler-aware variants mirror the behaviour in `context_implementations.jl` but
# operate via meta indices to avoid repeated lookups.
function tilde_assume_dod!!(
    rng::Random.AbstractRNG,
    sampler::Union{SampleFromPrior, SampleFromUniform},
    dist::Distribution,
    meta_i::Int,
    vi::DODVarInfo{AU},
) where {AU}
    meta = get_meta(vi, meta_i)
    vn = meta.name
    # If the meta placeholder has no backing storage yet (idx == 0), treat it
    # as not present and create a new entry. Otherwise use the existing value.
    if meta.idx == 0
        # not present: create a new entry
        r = init(rng, dist, sampler)
        if istrans(vi)
            f = to_linked_internal_transform(vi, vn, dist)
            # store linked/internal representation and mark as transformed
            add_variable!(vi, vn, f(r); transformed=true, dist=dist)
        else
            add_variable!(vi, vn, r, dist=dist)
        end
    else
        # Always overwrite for SampleFromUniform or when marked for deletion
        if sampler isa SampleFromUniform || is_flagged(vi, vn, :del)
            unset_flag!(vi, vn, :del)
            r = init(rng, dist, sampler)
            f = to_maybe_linked_internal_transform(vi, vn, dist)
            # store the *internal* representation
            vi = set_value_by_meta!(vi, meta_i, f(r))
        else
            # otherwise just extract existing internal value and reconstruct
            r = get_value_by_meta(vi, meta_i)
        end
    end

    logjac = logabsdetjac(istrans(vi, vn) ? link_transform(dist) : identity, r)
        vi = accumulate_assume_dod!!(vi, r, logjac, vn, dist)
    return r, vi
end

# Provide a non-rng sampler entry (mirrors `assume` without rng)
function tilde_assume_dod!!(
    sampler::Union{SampleFromPrior,SampleFromUniform},
    dist::Distribution,
    meta_i::Int,
    vi::DODVarInfo{AU},
) where {AU}
    return tilde_assume_dod!!(Random.default_rng(), sampler, dist, meta_i, vi)
end

# Observe variant that accepts meta index
function tilde_observe_dod!!(::Any, right, left, meta_i::Int, vi::DODVarInfo{AU}) where {AU}
    meta = get_meta(vi, meta_i)
    vn = meta.name
        vi = accumulate_observe_dod!!(vi, right, left, vn)
    return left, vi
end

# Backwards-compatible wrappers that accept a VarName but use the faster meta-path
function tilde_assume_dod!!(::Any, right, vn::VarName, vi::DODVarInfo{AU}) where {AU}
    meta_i = get(vi.idcs, vn, 0)
    if meta_i == 0
    # Fall back to previous behaviour: read the stored internal value via
    # VarName access (this will throw `KeyError` if the variable is missing)
    # and then perform the usual accumulation. This mirrors the non-DOD
    # `assume(dist, vn, vi)` behavior.
    y = getindex_internal(vi, vn)
    f = from_maybe_linked_internal_transform(vi, vn, right)
    x, inv_logjac = with_logabsdet_jacobian(f, y)
    vi = accumulate_assume_dod!!(vi, x, -inv_logjac, vn, right)
    return x, vi
    else
        return tilde_assume_dod!!(right, meta_i, vi)
    end
end

function tilde_assume_dod!!(right::Distribution, vn::VarName, vi::DODVarInfo{AU}) where {AU}
    return tilde_assume_dod!!(nothing, right, vn, vi)
end

function tilde_observe_dod!!(::Any, right, left, vn::VarName, vi::DODVarInfo{AU}) where {AU}
    meta_i = get(vi.idcs, vn, 0)
    if meta_i == 0
        vi = accumulate_observe_dod!!(vi, right, left, vn)
        return left, vi
    else
        return tilde_observe_dod!!(right, left, meta_i, vi)
    end
end

istrans(vi::DODVarInfo{AU}, vn::VarName) where {AU} = begin
    i = get(vi.idcs, vn, 0)
    i == 0 && return false
    return vi.metas[i].transformed
end

function set_flag!(vi::DODVarInfo{AU}, vn::VarName, flag::Union{String,Symbol}) where {AU}
    i = vi.idcs[vn]
    vi.metas[i].flags[Symbol(flag)] = true
end

function unset_flag!(vi::DODVarInfo{AU}, vn::VarName, flag::Union{String,Symbol}) where {AU}
    i = vi.idcs[vn]
    delete!(vi.metas[i].flags, Symbol(flag))
end

function is_flagged(vi::DODVarInfo{AU}, vn::VarName, flag::Union{String,Symbol}) where {AU}
    i = get(vi.idcs, vn, 0)
    i == 0 && return false
    return get(vi.metas[i].flags, Symbol(flag), false)
end

# Small helper: precompute presence of common accumulators to avoid repeated
# haskey calls on the `AccumulatorTuple`/NamedTuple in hot inner loops.
function update_acc_cache!(vi::DODVarInfo{AU}) where {AU}
    accs = vi.accumulators
    # Helper to construct a new DODVarInfo with specified updater type/values
    make_new_vi = function(new_updater, acc_default, accumulators_view)
    return DODVarInfo{typeof(new_updater)}(vi.values, vi.metas, vi.idcs, accumulators_view === nothing ? vi.accumulators : accumulators_view, nothing, nothing, acc_default, new_updater, vi.meta_getters, vi.meta_setters, vi.typed_slots, vi.allocator_cursor, vi.allocator_freelist)
    end

    if accs isa AccumulatorTuple
        nt = accs.nt
        lp_ok = haskey(nt, :LogPrior)
        lj_ok = haskey(nt, :LogJacobian)
        ll_ok = haskey(nt, :LogLikelihood)
        acc_cache = (lp_ok, lj_ok, ll_ok)
        if lp_ok || lj_ok || ll_ok
            # create holder and updater
            holder = MutableAccHolder(
                lp_ok ? nt.LogPrior : nothing,
                lj_ok ? nt.LogJacobian : nothing,
                ll_ok ? nt.LogLikelihood : nothing,
            )
            updater = make_accumulator_updater_from_holder(holder)
            # If current vi already has same updater type, mutate in-place
            if typeof(vi.acc_updater) === typeof(updater)
                vi.acc_cache = acc_cache
                vi.acc_keys = fieldnames(typeof(nt))
                vi.acc_default = holder
                vi.accumulators = AccumulatorView(nt, holder)
                vi.acc_updater = updater
                return vi
            else
                # Return a new DODVarInfo parameterized on updater type
                new_vi = make_new_vi(updater, holder, AccumulatorView(nt, holder))
                new_vi.acc_cache = acc_cache
                new_vi.acc_keys = fieldnames(typeof(nt))
                return new_vi
            end
        else
            # no relevant accumulators: use NoopAccUpdater
            if vi.acc_updater isa NoopAccUpdater
                vi.acc_cache = (false,false,false)
                vi.acc_keys = nothing
                vi.acc_default = nothing
                vi.accumulators = accs
                vi.acc_updater = NoopAccUpdater()
                return vi
            else
                new_vi = make_new_vi(NoopAccUpdater(), nothing, nothing)
                new_vi.acc_cache = nothing
                new_vi.acc_keys = nothing
                return new_vi
            end
        end
    else
        # accs is not an AccumulatorTuple: always return a NoopAccUpdater-typed vi
        if vi.acc_updater isa NoopAccUpdater
            vi.acc_cache = nothing
            vi.acc_keys = nothing
            vi.acc_default = nothing
            vi.accumulators = accs
            vi.acc_updater = NoopAccUpdater()
            return vi
        else
            new_vi = make_new_vi(NoopAccUpdater(), nothing, nothing)
            new_vi.acc_cache = nothing
            new_vi.acc_keys = nothing
            return new_vi
        end
    end
end

# Helper to detect if the updater is a real AccUpdater (not a Noop)
has_updater(vi::DODVarInfo) = !(vi.acc_updater isa NoopAccUpdater)

"""Return a small updater closure that updates the accumulators for `vi`.
The closure signature is (op::Symbol, vi::DODVarInfo, args...) where op is
:assume or :observe and args are forwarded to the appropriate accumulate_*!!.
This is a light-weight specialization scaffold; codegen (option 3) can emit
an inlined version of this per-model to eliminate the closure overhead.
"""
function make_accumulator_updater(vi::DODVarInfo)
    # Capture the types of the holder accumulators and precompute whether
    # the in-place accumulate methods exist for each accumulator type. Doing
    # this once here removes repeated `hasmethod` checks from the hot path.
    holder = vi.acc_default
    lp_t = typeof(holder.lp)
    lj_t = typeof(holder.lj)
    ll_t = typeof(holder.ll)

    # Assume the default accumulator types implement in-place APIs; avoid
    # repeated runtime `hasmethod` checks in the hot path by setting these to
    # `true` so the updater will use the in-place variants.
    lp_assume_inplace = true
    lj_assume_inplace = true
    ll_assume_inplace = true

    lp_observe_inplace = true
    lj_observe_inplace = true
    ll_observe_inplace = true
        lp_assume_inplace = holder.lp !== nothing && hasmethod(accumulate_assume_inplace!!, Tuple{typeof(holder.lp),Any,Any,VarName,Any})
        lj_assume_inplace = holder.lj !== nothing && hasmethod(accumulate_assume_inplace!!, Tuple{typeof(holder.lj),Any,Any,VarName,Any})
        ll_assume_inplace = holder.ll !== nothing && hasmethod(accumulate_assume_inplace!!, Tuple{typeof(holder.ll),Any,Any,VarName,Any})

        lp_observe_inplace = holder.lp !== nothing && hasmethod(accumulate_observe_inplace!!, Tuple{typeof(holder.lp),Any,Any,VarName})
        lj_observe_inplace = holder.lj !== nothing && hasmethod(accumulate_observe_inplace!!, Tuple{typeof(holder.lj),Any,Any,VarName})
        ll_observe_inplace = holder.ll !== nothing && hasmethod(accumulate_observe_inplace!!, Tuple{typeof(holder.ll),Any,Any,VarName})
    # instantiate a specialized updater type using the booleans we computed
    # Top-level functions `acc_updater_assume` and `acc_updater_observe` will
    # be used to perform the updates for the returned AccUpdater instance.
    return AccUpdater{lp_assume_inplace, lj_assume_inplace, ll_assume_inplace, lp_observe_inplace, lj_observe_inplace, ll_observe_inplace, lp_t, lj_t, ll_t}(holder)
end

# Build an AccUpdater instance directly from a MutableAccHolder, returning a
# concrete AccUpdater{...} specialized on the holder field types and whether
# in-place methods exist. This is used when constructing a parametric
# `DODVarInfo{AU}` from a `ModelDOD` so the updater field can be concrete.
function make_accumulator_updater_from_holder(holder::MutableAccHolder)
    lp_t = typeof(holder.lp)
    lj_t = typeof(holder.lj)
    ll_t = typeof(holder.ll)

    lp_assume_inplace = holder.lp !== nothing && hasmethod(accumulate_assume_inplace!!, Tuple{typeof(holder.lp),Any,Any,VarName,Any})
    lj_assume_inplace = holder.lj !== nothing && hasmethod(accumulate_assume_inplace!!, Tuple{typeof(holder.lj),Any,Any,VarName,Any})
    ll_assume_inplace = holder.ll !== nothing && hasmethod(accumulate_assume_inplace!!, Tuple{typeof(holder.ll),Any,Any,VarName,Any})

    lp_observe_inplace = holder.lp !== nothing && hasmethod(accumulate_observe_inplace!!, Tuple{typeof(holder.lp),Any,Any,VarName})
    lj_observe_inplace = holder.lj !== nothing && hasmethod(accumulate_observe_inplace!!, Tuple{typeof(holder.lj),Any,Any,VarName})
    ll_observe_inplace = holder.ll !== nothing && hasmethod(accumulate_observe_inplace!!, Tuple{typeof(holder.ll),Any,Any,VarName})

    return AccUpdater{lp_assume_inplace, lj_assume_inplace, ll_assume_inplace, lp_observe_inplace, lj_observe_inplace, ll_observe_inplace, lp_t, lj_t, ll_t}(holder)
end

# Optimized accumulator fast-paths for DODVarInfo. These avoid the generic
# `map_accumulators!!` machinery (which maps over a `NamedTuple`) by updating
# the common default accumulators directly when present.
@inline function accumulate_assume_dod!!(vi::DODVarInfo, val, logjac, vn, right)
    accs = vi.accumulators
    # If the value is `nothing` this is the lazy-add path; we don't have a
    # concrete value to score, so skip accumulator updates entirely and
    # return the VarInfo unchanged. The actual variable will be added later
    # when a concrete value is provided.
    if val === nothing
        return vi
    end

    # fast-path: consult the cached presence flags when available
    if has_updater(vi)
    @debug "DOD fast-path: acc_updater_assume called" holder_ptr=pointer_from_objref(vi.acc_updater.holder) acc_holder_ptr=(vi.accumulators isa AccumulatorView ? pointer_from_objref(vi.accumulators.holder) : "no-holder")
        # use the cached typed AccUpdater for minimal-dispatch updates
        acc_updater_assume(vi.acc_updater, vi, val, logjac, vn, right)
        return vi
    elseif vi.acc_cache !== nothing && vi.acc_cache isa Tuple && (vi.acc_cache[1] || vi.acc_cache[2] || vi.acc_cache[3])
        # use acc_default holder when we have presence flags indicating accumulators
        holder = vi.acc_default
        if holder !== nothing
            @debug "DOD fast-path: acc_default used" holder_ptr=pointer_from_objref(holder) acc_holder_ptr=(vi.accumulators isa AccumulatorView ? pointer_from_objref(vi.accumulators.holder) : "no-holder")
            if holder.lp !== nothing
                holder.lp = accumulate_assume!!(holder.lp, val, logjac, vn, right)
            end
            if holder.lj !== nothing
                holder.lj = accumulate_assume!!(holder.lj, val, logjac, vn, right)
            end
            if holder.ll !== nothing
                holder.ll = accumulate_assume!!(holder.ll, val, logjac, vn, right)
            end
            return vi
        end
    elseif vi.acc_default !== nothing
        # backward-compatible path: acc_default is a MutableAccHolder
        holder = vi.acc_default
        if holder !== nothing
            if holder.lp !== nothing
                holder.lp = accumulate_assume!!(holder.lp, val, logjac, vn, right)
            end
            if holder.lj !== nothing
                holder.lj = accumulate_assume!!(holder.lj, val, logjac, vn, right)
            end
            if holder.ll !== nothing
                holder.ll = accumulate_assume!!(holder.ll, val, logjac, vn, right)
            end
            return vi
        end
    end
    # Fallback to generic mapping
    return map_accumulators!!(acc -> accumulate_assume!!(acc, val, logjac, vn, right), vi)
end

@inline function accumulate_observe_dod!!(vi::DODVarInfo, right, left, vn)
    accs = vi.accumulators
    # fast-path using acc_cache
    if has_updater(vi)
        acc_updater_observe(vi.acc_updater, vi, right, left, vn)
        return vi
    elseif vi.acc_cache !== nothing && vi.acc_cache isa Tuple && (vi.acc_cache[1] || vi.acc_cache[2] || vi.acc_cache[3])
        holder = vi.acc_default
        if holder !== nothing
            if holder.lp !== nothing
                holder.lp = accumulate_observe!!(holder.lp, right, left, vn)
            end
            if holder.lj !== nothing
                holder.lj = accumulate_observe!!(holder.lj, right, left, vn)
            end
            if holder.ll !== nothing
                holder.ll = accumulate_observe!!(holder.ll, right, left, vn)
            end
            return vi
        end
    elseif vi.acc_default !== nothing
        holder = vi.acc_default
        if holder.lp !== nothing
            holder.lp = accumulate_observe!!(holder.lp, right, left, vn)
        end
        if holder.lj !== nothing
            holder.lj = accumulate_observe!!(holder.lj, right, left, vn)
        end
        if holder.ll !== nothing
            holder.ll = accumulate_observe!!(holder.ll, right, left, vn)
        end
        return vi
    end
    return map_accumulators!!(acc -> accumulate_observe!!(acc, right, left, vn), vi)
end

# Ensure the generic accumulator API dispatches to the DOD-optimized
# implementations when a `DODVarInfo` is provided so generic `assume` and
# `observe` code paths (which call `accumulate_*!!`) will use the fast-path.
function accumulate_assume!!(vi::DODVarInfo, val, logjac, vn, right)
    # DEBUG: trace DOD accumulate dispatch (logged at debug level)
    @debug :accumulate_assume_called typeof(vi) val_is_nothing=(val === nothing)
    return accumulate_assume_dod!!(vi, val, logjac, vn, right)
end

function accumulate_observe!!(vi::DODVarInfo, right, left, vn)
    @debug :accumulate_observe_called typeof(vi)
    return accumulate_observe_dod!!(vi, right, left, vn)
end

# end of file - content defined in parent DynamicPPL module

# Fast path helpers that avoid the ThreadSafeVarInfo wrapper when evaluating
# a model with a `DODVarInfo`. These are defined here to avoid circular
# include order issues; they rely only on `Model` and `evaluate_threadunsafe!!`.
function DynamicPPL.logprior(m::DynamicPPL.ModelDOD, vi::DODVarInfo{AU}) where {AU}
    result, vi_new = DynamicPPL.evaluate_threadunsafe!!(DynamicPPL.Model(m.f, m.args, m.defaults, m.context), vi)
    return DynamicPPL.getlogprior(last((result, vi_new)))
end

function Distributions.loglikelihood(m::DynamicPPL.ModelDOD, vi::DODVarInfo{AU}) where {AU}
    result, vi_new = DynamicPPL.evaluate_threadunsafe!!(DynamicPPL.Model(m.f, m.args, m.defaults, m.context), vi)
    return DynamicPPL.getloglikelihood(last((result, vi_new)))
end

# Prefer the thread-unsafe evaluation path whenever a `DODVarInfo` is
# supplied. Dispatching on the `varinfo` type (rather than the ModelDOD
# wrapper) lets any `Model` be evaluated via the fast, thread-unsafe path
# when a `DODVarInfo` is provided, avoiding the `ThreadSafeVarInfo` wrapper.
function AbstractPPL.evaluate!!(m::DynamicPPL.Model, vi::DODVarInfo{AU}) where {AU}
    return evaluate_threadunsafe!!(m, vi)
end

# Also accept ModelDOD directly and route to the same thread-unsafe evaluator by
# wrapping the `ModelDOD` into a regular `Model`. This keeps the DOD-specific
# emission semantics while reusing the same hot evaluator path.
function AbstractPPL.evaluate!!(m::DynamicPPL.ModelDOD, vi::DODVarInfo{AU}) where {AU}
    # Some `ModelDOD` instances may encode missing/default keyword arguments as
    # `nothing` in the `defaults` NamedTuple which can leak into generated
    # evaluators and be used as constructor arguments (e.g. `Normal(nothing, 1)`).
    # Filter out `nothing` entries so the wrapped `Model` only carries real
    # defaults into the evaluator.
    filtered_defaults = (; [k => v for (k, v) in pairs(m.defaults) if v !== nothing]...)
    return evaluate_threadunsafe!!(DynamicPPL.Model(m.f, m.args, filtered_defaults, m.context), vi)
end
