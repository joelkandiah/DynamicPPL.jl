

"""
    DictVarInfo(
        rng::Random.AbstractRNG,
        model::Model,
        init_strategy::AbstractInitStrategy=InitFromPrior(),
    )

Generate a `DictVarInfo` object for the given `model`.
"""
# Constructor for the specific type alias
function (::Type{DictVarInfo})(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
)
    return typed_dict_varinfo(untyped_varinfo(rng,model, init_strategy))
end

function (::Type{DictVarInfo})(model::Model, init_strategy::AbstractInitStrategy=InitFromPrior())
    return DictVarInfo(Random.default_rng(), model, init_strategy)
end

"""
    typed_dict_varinfo(vi::UntypedVarInfo)

Converts an `UntypedVarInfo` (metadata is a single generic `Metadata`) into a `DictVarInfo` (metadata is a `Dictionary` of typed `Metadata`).
"""
function typed_dict_varinfo(vi::UntypedVarInfo)
    meta = vi.metadata
    # Group Metadata by symbol
    # This logic matches typed_varinfo but constructs a Dictionary instead of NamedTuple
    new_metas = Metadata[]
    
    # Symbols of all instances of `VarName{sym}` in `vi.vns`
    syms_found = unique(map(getsym, meta.vns))
    
    for s in syms_found
        # Find all indices in `vns` with symbol `s`
        inds = findall(vn -> getsym(vn) === s, meta.vns)
        n = length(inds)
        # New `vns`
        sym_vns = getindex.((meta.vns,), inds)
        # New idcs
        sym_idcs = Dict(a => i for (i, a) in enumerate(sym_vns))
        # New dists
        sym_dists = getindex.((meta.dists,), inds)
        # New is_transformed
        sym_is_transformed = meta.is_transformed[inds]

        # Extract new ranges and vals
        _ranges = getindex.((meta.ranges,), inds)
        # `copy.()` is a workaround to reduce the eltype from Real to Int or Float64
        _vals = [copy.(meta.vals[_ranges[i]]) for i in 1:n]
        # Recalculate ranges for the new contiguous storage
        sym_ranges = Vector{eltype(_ranges)}(undef, n)
        start = 0
        for i in 1:n
            len = length(_vals[i])
            sym_ranges[i] = (start + 1):(start + len)
            start += len
        end
        sym_vals = foldl(vcat, _vals)

        push!(
            new_metas,
            Metadata(
                sym_idcs, sym_vns, sym_ranges, sym_vals, sym_dists, sym_is_transformed
            ),
        )
    end
    
    # Create the Dictionary
    # keys are the symbols, values are the Metadata
    dict_meta = dictionary(zip(syms_found, new_metas))
    
    return VarInfo(dict_meta, copy(vi.accs))
end

function typed_dict_varinfo(vi::DictVarInfo)
    return vi
end

function typed_dict_varinfo(
    rng::Random.AbstractRNG,
    model::Model,
    init_strategy::AbstractInitStrategy=InitFromPrior(),
)
    return typed_dict_varinfo(untyped_varinfo(rng, model, init_strategy))
end
function typed_dict_varinfo(model::Model, init_strategy::AbstractInitStrategy=InitFromPrior())
    return typed_dict_varinfo(Random.default_rng(), model, init_strategy)
end


# === Overloads for VarInfo with Dictionary metadata ===

# keys
Base.keys(vi::DictVarInfo) = mapreduce(md -> md.vns, vcat, values(vi.metadata))

# values_as
function values_as(vi::DictVarInfo, ::Type{Dictionary})
    d = Dictionary{VarName, Any}()
    for md in values(vi.metadata)
        for vn in md.vns
            insert!(d, vn, vi[vn])
        end
    end
    return d
end
values_as(vi::DictVarInfo) = values_as(vi, Dictionary)

# getmetadata
getmetadata(vi::DictVarInfo, vn::VarName) = vi.metadata[getsym(vn)]

# vector_length
vector_length(vi::DictVarInfo) = sum(vector_length, vi.metadata)

# unflatten needs careful handling for Dictionary order
# Dictionaries.jl usually preserves insertion order, or we can iterate
function unflatten_metadata(dict::Dictionary, x::AbstractVector)
    # We iterate over the dictionary. 
    # NOTE: This assumes that the iteration order matches the flattening order.
    # Dictionaries.jl `Dictionary` preserves insertion order if constructed that way?
    # Or rather, `map` produces a new Dictionary with same keys and order.
    
    offset = 0
    new_values = map(dict) do md
        len = vector_length(md)
        new_md = unflatten_metadata(md, x[(offset + 1):(offset + len)])
        offset += len
        return new_md
    end
    return new_values
end

# subset
function subset(metadata::Dictionary, vns::AbstractVector{<:VarName})
    # Identify which symbols are relevant
    vns_syms = Set(unique(map(getsym, vns)))
    
    # Filter the dictionary to contain only these symbols
    # Then subset each metadata within
    
    # We can use `filter` on the dictionary, then map
    relevant_meta = filter(kv -> (keys(kv) in vns_syms), metadata)
    
    # Now for each metadata, subset it
    sub_meta = map(pairs(relevant_meta)) do (sym, md)
        subset(md, filter(==(sym) ∘ getsym, vns))
    end
    
    return sub_meta
end

# merge_metadata
function merge_metadata(left::Dictionary, right::Dictionary)
    # Dictionaries.jl merge equivalent?
    # We need to merge recursively if keys overlap
    
    # Strategy: 
    # 1. Identify union of keys
    # 2. For each key, if present in both, call merge_metadata on values
    # 3. Else take the one present
    
    # Note: `merge` in Dictionaries.jl might prefer right?
    # But we need deep merge for Metadata
    
    all_keys = union(keys(left), keys(right))
    
    new_vals = map(all_keys) do k
        if haskey(left, k) && haskey(right, k)
            return merge_metadata(left[k], right[k])
        elseif haskey(left, k)
            return left[k]
        else
            return right[k]
        end
    end
    
    return dictionary(zip(all_keys, new_vals))
end

# getindex_internal for Colon
function getindex_internal(vi::DictVarInfo, ::Colon)
    # Concatenate values from all metadata
    return mapreduce(
        Base.Fix2(getindex_internal, Colon()), 
        vcat, 
        vi.metadata
    )
end

# vector_getrange
function vector_getrange(vi::DictVarInfo, vn::VarName)
    offset = 0
    sym = getsym(vn)
    
    # Iterate to find offset
    for (k, md) in pairs(vi.metadata)
        if k === sym
            return getrange(md, vn) .+ offset
        end
        offset += sum(length, md.ranges)
    end
    throw(KeyError(vn))
end

# vector_getranges
function vector_getranges(vi::DictVarInfo, vns::Vector{<:VarName})
    # Similar strategy to NTVarInfo but iterating dictionary
    # TODO: optimize if Dictionary allows efficient lookup
    
    ranges = Vector{UnitRange{Int}}(undef, length(vns))
    not_seen = fill(true, length(vns))
    
    offset = 0
    for (k, md) in pairs(vi.metadata)
        # Check which vns belong to this metadata (symbol match)
        # Optimization: pre-group vns by symbol?
        
        current_md_len = sum(length, md.ranges)
        
        # Naive: filter vns for this symbol
        # But we need to fill the correct indices in `ranges`
        
        for (i, vn) in enumerate(vns)
            if not_seen[i] && getsym(vn) === k
                 ranges[i] = getrange(md, vn) .+ offset
                 not_seen[i] = false
            end
        end
        
        offset += current_md_len
    end
    
    if any(not_seen)
        throw(KeyError(vns[findall(not_seen)]))
    end
    return ranges
end

# set_transformed!!
function set_transformed!!(vi::DictVarInfo, val::Bool, vn::VarName)
    sym = getsym(vn)
    md = vi.metadata[sym]
    new_md = set_transformed!!(md, val, vn)
    
    # Update the dictionary
    # Dictionaries are immutable-ish by default? 
    # Use insert/set for Dictionary?
    # Dictionaries.jl provides mechanism for "set" producing new dictionary?
    # `set` from Dictionaries.jl
    
    new_dict = set(vi.metadata, sym, new_md)
    return Accessors.@set vi.metadata = new_dict
end

# Linking and InvLinking for DictVarInfo

function _link!!(vi::DictVarInfo, vns)
    vns_by_sym = Dict{Symbol, Vector{VarName}}()
    for vn in vns
        sym = getsym(vn)
        if !haskey(vns_by_sym, sym)
             vns_by_sym[sym] = VarName[]
        end
        push!(vns_by_sym[sym], vn)
    end
    
    cumulative_logjac = 0.0
    
    for (sym, md) in pairs(vi.metadata)
        relevant_vns = get(vns_by_sym, sym, VarName[])
        
        for vn in relevant_vns
             if !is_transformed(vi, vn)
                 f = internal_to_linked_internal_transform(vi, vn)
                 
                 # Manual _inner_transform! logic to accumulate logjac
                 yvec, logjac = with_logabsdet_jacobian(f, getindex_internal(md, vn))
                 start = first(getrange(md, vn))
                 setrange!(md, vn, start:(start + length(yvec) - 1))
                 setval!(md, yvec, vn)
                 
                 cumulative_logjac += logjac
                 set_transformed!!(md, true, vn)
             end
        end
    end
    
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, cumulative_logjac)
    end
    return vi
end

function _invlink!!(vi::DictVarInfo, vns)
    vns_by_sym = Dict{Symbol, Vector{VarName}}()
    for vn in vns
        sym = getsym(vn)
        if !haskey(vns_by_sym, sym); vns_by_sym[sym] = VarName[]; end
        push!(vns_by_sym[sym], vn)
    end
    
    cumulative_inv_logjac = 0.0
    
    for (sym, md) in pairs(vi.metadata)
        relevant_vns = get(vns_by_sym, sym, VarName[])
        
        for vn in relevant_vns
            if is_transformed(vi, vn)
                f = linked_internal_to_internal_transform(vi, vn)
                
                # Manual logic
                y, inv_logjac = with_logabsdet_jacobian(f, getindex_internal(md, vn))
                yvec = tovec(y)
                
                start = first(getrange(md, vn))
                setrange!(md, vn, start:(start + length(yvec) - 1))
                setval!(md, yvec, vn)
                
                cumulative_inv_logjac += inv_logjac
                set_transformed!!(md, false, vn)
            end
        end
    end
    
    if hasacc(vi, Val(:LogJacobian))
        vi = acclogjac!!(vi, cumulative_inv_logjac)
    end
    return vi
end
