const INTERNALNAMES = (:__model__, :__varinfo__)

"""
    need_concretize(expr)

Return `true` if `expr` needs to be concretized, i.e., if it contains a colon `:` or
requires a dynamic optic.

# Examples

```jldoctest; setup=:(using Accessors)
julia> DynamicPPL.need_concretize(:(x[1, :]))
true

julia> DynamicPPL.need_concretize(:(x[1, end]))
true

julia> DynamicPPL.need_concretize(:(x[1, 1]))
false
"""
function need_concretize(expr)
    return Accessors.need_dynamic_optic(expr) || begin
        flag = false
        MacroTools.postwalk(expr) do ex
            # Concretise colon by default
            ex == :(:) && (flag = true) && return ex
        end
        flag
    end
end

"""
    make_varname_expression(expr)

Return a `VarName` based on `expr`, concretizing it if necessary.
"""
function make_varname_expression(expr)
    # HACK: Usage of `drop_escape` is unfortunate. It's a consequence of the fact
    # that in DynamicPPL we the entire function body. Instead we should be
    # more selective with our escape. Until that's the case, we remove them all.
    return AbstractPPL.drop_escape(varname(expr, need_concretize(expr)))
end

"""
    isassumption(expr[, vn])

Return an expression that can be evaluated to check if `expr` is an assumption in the
model.

Let `expr` be `:(x[1])`. It is an assumption in the following cases:
    1. `x` is not among the input data to the model,
    2. `x` is among the input data to the model but with a value `missing`, or
    3. `x` is among the input data to the model with a value other than missing,
       but `x[1] === missing`.

When `expr` is not an expression or symbol (i.e., a literal), this expands to `false`.

If `vn` is specified, it will be assumed to refer to a expression which
evaluates to a `VarName`, and this will be used in the subsequent checks.
If `vn` is not specified, `AbstractPPL.varname(expr, need_concretize(expr))` will be
used in its place.
"""
function isassumption(expr::Union{Expr,Symbol}, vn=make_varname_expression(expr))
    return quote
        if $(DynamicPPL.contextual_isassumption)(
            __model__.context, $(DynamicPPL.prefix)(__model__.context, $vn)
        )
            # Considered an assumption by `__model__.context` which means either:
            # 1. We hit the default implementation, e.g. using `DefaultContext`,
            #    which in turn means that we haven't considered if it's one of
            #    the model arguments, hence we need to check this.
            # 2. We are working with a `ConditionContext` _and_ it's NOT in the model arguments,
            #    i.e. we're trying to condition one of the latent variables.
            #    In this case, the below will return `true` since the first branch
            #    will be hit.
            # 3. We are working with a `ConditionContext` _and_ it's in the model arguments,
            #    i.e. we're trying to override the value. This is currently NOT supported.
            #    TODO: Support by adding context to model, and use `model.args`
            #    as the default conditioning. Then we no longer need to check `inargnames`
            #    since it will all be handled by `contextual_isassumption`.
            if !($(DynamicPPL.inargnames)($vn, __model__)) ||
                $(DynamicPPL.inmissings)($vn, __model__)
                true
            else
                $(maybe_view(expr)) === missing
            end
        else
            false
        end
    end
end

# failsafe: a literal is never an assumption
isassumption(expr, vn) = :(false)
isassumption(expr) = :(false)

"""
    contextual_isassumption(context, vn)

Return `true` if `vn` is considered an assumption by `context`.
"""
function contextual_isassumption(context::AbstractContext, vn)
    if hasconditioned_nested(context, vn)
        val = getconditioned_nested(context, vn)
        # TODO: Do we even need the `>: Missing`, i.e. does it even help the compiler?
        if eltype(val) >: Missing && val === missing
            return true
        else
            return false
        end
    else
        return true
    end
end

isfixed(expr, vn) = false
function isfixed(::Union{Symbol,Expr}, vn)
    return :($(DynamicPPL.contextual_isfixed)(
        __model__.context, $(DynamicPPL.prefix)(__model__.context, $vn)
    ))
end

"""
    contextual_isfixed(context, vn)

Return `true` if `vn` is considered fixed by `context`.
"""
function contextual_isfixed(context::AbstractContext, vn)
    if hasfixed_nested(context, vn)
        val = getfixed_nested(context, vn)
        # TODO: Do we even need the `>: Missing`, i.e. does it even help the compiler?
        if eltype(val) >: Missing && val === missing
            return false
        else
            return true
        end
    else
        return false
    end
end

# If we're working with, say, a `Symbol`, then we're not going to `view`.
maybe_view(x) = x
maybe_view(x::Expr) = :(@views($x))

"""
    isliteral(expr)

Return `true` if `expr` is a literal, e.g. `1.0` or `[1.0, ]`, and `false` otherwise.
"""
isliteral(e) = false
isliteral(::Number) = true
function isliteral(e::Expr)
    # In the special case that the expression is of the form `abc[blahblah]`, we consider it
    # to be a literal if `abc` is a literal. This is necessary for cases like
    # [1.0, 2.0][idx...] ~ Normal()
    # which are generated when turning `.~` expressions into loops over `~` expressions.
    if e.head == :ref
        return isliteral(e.args[1])
    end
    return !isempty(e.args) && all(isliteral, e.args)
end

"""
    check_tilde_rhs(x)

Check if the right-hand side `x` of a `~` is a `Distribution` or an array of
`Distributions`, then return `x`.
"""
function check_tilde_rhs(@nospecialize(x))
    return throw(
        ArgumentError(
            "the right-hand side of a `~` must be a `Distribution`, an array of `Distribution`s, or a submodel",
        ),
    )
end
check_tilde_rhs(x::Distribution) = x
check_tilde_rhs(x::AbstractArray{<:Distribution}) = x
check_tilde_rhs(x::Submodel{M,AutoPrefix}) where {M,AutoPrefix} = x

"""
    check_dot_tilde_rhs(x)

Check if the right-hand side `x` of a `.~` is a `UnivariateDistribution`, then return `x`.
"""
function check_dot_tilde_rhs(@nospecialize(x))
    return throw(
        ArgumentError("the right-hand side of a `.~` must be a `UnivariateDistribution`")
    )
end
function check_dot_tilde_rhs(::AbstractArray{<:Distribution})
    msg = """
        As of v0.35, DynamicPPL does not allow arrays of distributions in `.~`. \
        Please use `product_distribution` instead, or write a loop if necessary. \
        See https://github.com/TuringLang/DynamicPPL.jl/releases/tag/v0.35.0 for more \
        details.\
    """
    return throw(ArgumentError(msg))
end
check_dot_tilde_rhs(x::UnivariateDistribution) = x

"""
    unwrap_right_vn(right, vn)

Return the unwrapped distribution on the right-hand side and variable name on the left-hand
side of a `~` expression such as `x ~ Normal()`.

This is used mainly to unwrap `NamedDist` distributions.
"""
unwrap_right_vn(right, vn) = right, vn
unwrap_right_vn(right::NamedDist, vn) = unwrap_right_vn(right.dist, right.name)

"""
    unwrap_right_left_vns(right, left, vns)

Return the unwrapped distributions on the right-hand side and values and variable names on the
left-hand side of a `.~` expression such as `x .~ Normal()`.

This is used mainly to unwrap `NamedDist` distributions and adjust the indices of the
variables.

# Example
```jldoctest; setup=:(using Distributions, LinearAlgebra)
julia> _, _, vns = DynamicPPL.unwrap_right_left_vns(MvNormal(ones(2), I), randn(2, 2), @varname(x)); vns[end]
x[:, 2]

julia> _, _, vns = DynamicPPL.unwrap_right_left_vns(Normal(), randn(1, 2), @varname(x)); vns[end]
x[1, 2]

julia> _, _, vns = DynamicPPL.unwrap_right_left_vns(Normal(), randn(1, 2), @varname(x[:])); vns[end]
x[:][1, 2]

julia> _, _, vns = DynamicPPL.unwrap_right_left_vns(Normal(), randn(3), @varname(x[1])); vns[end]
x[1][3]
```
"""
unwrap_right_left_vns(right, left, vns) = right, left, vns
function unwrap_right_left_vns(right::NamedDist, left::AbstractArray, ::VarName)
    return unwrap_right_left_vns(right.dist, left, right.name)
end
function unwrap_right_left_vns(right::NamedDist, left::AbstractMatrix, ::VarName)
    return unwrap_right_left_vns(right.dist, left, right.name)
end
function unwrap_right_left_vns(
    right::MultivariateDistribution, left::AbstractMatrix, vn::VarName
)
    # This an expression such as `x .~ MvNormal()` which we interpret as
    #     x[:, i] ~ MvNormal()
    # for `i = size(left, 2)`. Hence the symbol should be `x[:, i]`,
    # and we therefore add the `Colon()` below.
    vns = map(axes(left, 2)) do i
        return AbstractPPL.concretize(Accessors.IndexLens((Colon(), i)) ∘ vn, left)
    end
    return unwrap_right_left_vns(right, left, vns)
end
function unwrap_right_left_vns(
    right::Union{Distribution,AbstractArray{<:Distribution}},
    left::AbstractArray,
    vn::VarName,
)
    vns = map(CartesianIndices(left)) do i
        return Accessors.IndexLens(Tuple(i)) ∘ vn
    end
    return unwrap_right_left_vns(right, left, vns)
end

resolve_varnames(vn::VarName, _) = vn
resolve_varnames(vn::VarName, dist::NamedDist) = dist.name

#################
# Main Compiler #
#################

"""
    @model(expr[, warn = false])

Macro to specify a probabilistic model.

If `warn` is `true`, a warning is displayed if internal variable names are used in the model
definition.

# Examples

Model definition:

```julia
@model function model(x, y = 42)
    ...
end
```

To generate a `Model`, call `model(xvalue)` or `model(xvalue, yvalue)`.
"""
macro model(expr, warn=false)
    # include `LineNumberNode` with information about the call site in the
    # generated function for easier debugging and interpretation of error messages
    return esc(model(__module__, __source__, expr, warn))
end

macro model_dod(expr, warn=false)
    # DOD-specific model generator.
    return esc(model_dod(__module__, __source__, expr, warn))
end

function model_dod(mod, linenumbernode, expr, warn)
    modeldef = build_model_definition(expr)

    # Keep the original body so we can collect VarNames used in the model.
    orig_body = modeldef[:body]

    # Generate main body (reuse existing logic)
    # Enable DOD-aware code generation so tilde sites can emit the fast-paths
    generated_body = generate_mainbody(mod, modeldef[:body], warn; dod=true)

    # Collect VarName literals used in the original body so we can precompute
    # meta-index locals at evaluator entry and make tilde sites use those
    # locals (avoids per-site dictionary lookups).
    vns = collect_model_varnames(orig_body)
    # Attempt to infer simple concrete element types per-meta from RHSs.
    meta_types = infer_meta_types(orig_body, vns)
    # Also attempt to infer per-meta element lengths (1 for scalars, n for vectors)
    meta_lens = infer_meta_lens(orig_body, vns)
    # Also collect tilde pairs to attempt to infer element types for each meta.
    tilde_pairs = collect_model_tilde_pairs(orig_body)

    # Build precomputed meta bindings and per-meta typed-slot locals
    if !isempty(vns)
        assigns = map(enumerate(vns)) do (i, vn_expr)
            meta_sym = meta_sym_for_vn(vn_expr)
            Expr(:(=), meta_sym, QuoteNode(i))
        end
        len_assigns = map(enumerate(vns)) do (i, vn_expr)
            len_sym = meta_len_sym_for_vn(vn_expr)
            Expr(:(=), len_sym, QuoteNode(meta_lens[i]))
        end
        typed_assigns = map(enumerate(vns)) do (i, vn_expr)
            typed_sym = meta_typed_sym_for_vn(vn_expr)
            # Local binding: typed slot if available, otherwise `nothing`.
            Expr(:(=), typed_sym, :( (__varinfo__.typed_slots !== nothing && length(__varinfo__.typed_slots) >= $(QuoteNode(i)) && __varinfo__.typed_slots[$(QuoteNode(i))] !== nothing) ? __varinfo__.typed_slots[$(QuoteNode(i))] : nothing ))
        end
    # Prebind acc_updater to a local so the generated evaluator can
    # specialize on the concrete updater type when available.
    acc_updater_assign = Expr(:(=), :(__acc_updater__), :( __varinfo__.acc_updater ))
        # Prepend the assignments to the generated body (meta index, meta len, typed slots)
    modeldef[:body] = Expr(:block, assigns..., len_assigns..., typed_assigns..., acc_updater_assign, generated_body)
    else
        modeldef[:body] = generated_body
    end

    # Store the model VarName list on the modeldef so it can be included in the
    # emitted `ModelDOD` object. This allows `ModelDOD` to carry a per-model
    # mapping of VarName -> meta index used by the DOD fast-path.
    modeldef[:meta_vns] = vns
    modeldef[:meta_types] = meta_types
    # Build a meta_types and meta_lens vector by matching the collected tilde RHS to vns
    meta_types = Vector{Any}(undef, length(vns))
    meta_lens = Vector{Int}(undef, length(vns))
    for (i, vn) in enumerate(vns)
        # find a tilde pair matching this VarName (by string form)
        found = false
        for (vn_expr, rhs) in tilde_pairs
            if string(vn_expr) == string(vn)
                meta_types[i] = infer_meta_type_from_rhs(rhs)
                meta_lens[i] = infer_meta_lens_from_rhs(rhs)
                found = true
                break
            end
        end
        if !found
            meta_types[i] = :(Any)
            meta_lens[i] = 1
        end
    end
    modeldef[:meta_types] = meta_types
    modeldef[:meta_lens] = meta_lens

    return build_output_dod(modeldef, linenumbernode)
end


# Deterministic meta symbol for a VarName expression so that we can emit a
# consistent local variable name for each VarName used in a model. We use the
# hash of the VarName expression to avoid collisions in the common case.
meta_sym_for_vn(vn_expr) = Symbol("meta_", abs(hash(vn_expr)))
meta_typed_sym_for_vn(vn_expr) = Symbol("meta_typed_", abs(hash(vn_expr)))
meta_len_sym_for_vn(vn_expr) = Symbol("meta_len_", abs(hash(vn_expr)))


# Walk the original model body and collect the unique VarName literal
# expressions corresponding to the left-hand side of `~` / `.~` sites.
function collect_model_varnames(expr)
    seen = Set{Any}()
    out = Any[]
    MacroTools.postwalk(expr) do ex
        args = getargs_tilde(ex)
        if args !== nothing
            L, R = args
            vn = make_varname_expression(L)
            if vn ∉ seen
                push!(out, vn)
                push!(seen, vn)
            end
        end
        args2 = getargs_dottilde(ex)
        if args2 !== nothing
            L, R = args2
            vn = make_varname_expression(L)
            if vn ∉ seen
                push!(out, vn)
                push!(seen, vn)
            end
        end
        return ex
    end
    return out
end

# Heuristic: look at RHS AST and try to classify a likely element type.
function infer_type_from_rhs(rhs)
    if rhs isa Expr && rhs.head == :call
        f = rhs.args[1]
        fname = string(f)
        # common multivariate distributions
        if occursin("MvNormal", fname) || occursin("Multivariate", fname)
            return Vector{Float64}
        end
        # common univariate distributions -> scalar Float64
        if occursin("Normal", fname) || occursin("Beta", fname) || occursin("Gamma", fname) || occursin("Exponential", fname) || occursin("Uniform", fname) || occursin("InverseGamma", fname)
            return Float64
        end
    end
    # fallback
    return Any
end

# Collect pairs of (VarName expression, RHS expression) for tilde sites so
# codegen can attempt to infer per-meta element types.
function collect_model_tilde_pairs(expr)
    pairs = Vector{Tuple{Any,Any}}()
    MacroTools.postwalk(expr) do ex
        args = getargs_tilde(ex)
        if args !== nothing
            L, R = args
            push!(pairs, (make_varname_expression(L), R))
        end
        args2 = getargs_dottilde(ex)
        if args2 !== nothing
            L, R = args2
            # For dotted-tilde sites we record the per-element RHS expression
            push!(pairs, (make_varname_expression(L), R))
        end
        return ex
    end
    return pairs
end

# Infer per-meta length (number of scalar elements) from RHS expressions.
function infer_meta_lens(body, vns)
    pairs = collect_model_tilde_pairs(body)
    inferred = Dict{Any,Int}()
    for (vn_expr, rhs) in pairs
        inferred[vn_expr] = infer_meta_lens_from_rhs(rhs)
    end
    return [get(inferred, vn, 1) for vn in vns]
end

function infer_meta_lens_from_rhs(rhs)
    # If RHS is a call to a multivariate distribution, guess length 2 (conservative)
    if rhs isa Expr && rhs.head == :call
        f = rhs.args[1]
        fname = string(f)
        if occursin("MvNormal", fname) || occursin("Multivariate", fname) || occursin("Dirichlet", fname)
            return 2
        end
    end
    # If RHS is a vector literal, use its length
    if rhs isa Expr && rhs.head == :vect
        return length(rhs.args)
    end
    return 1
end

# Build a simple meta_types vector for `@model_dod` by scanning tilde sites.
function infer_meta_types(body, vns)
    pairs = collect_model_tilde_pairs(body)
    inferred = Dict{Any,Any}()
    for (vn_expr, rhs) in pairs
        inferred[vn_expr] = infer_type_from_rhs(rhs)
    end
    return [get(inferred, vn, Any) for vn in vns]
end

# Very small heuristic inference of meta element types from RHS expressions.
# This is intentionally conservative: unrecognized RHS -> Any.
function infer_meta_type_from_rhs(rhs)
    # If RHS is a call expression, attempt to extract the called name.
    if Meta.isexpr(rhs, :call)
        fn = rhs.args[1]
        # fn may be a symbol or a dotted access like Distributions.Normal
        name = if fn isa Symbol
            String(fn)
        elseif Meta.isexpr(fn, :.)
            # take the last part e.g. Distributions.Normal -> Normal
            last(fn.args) isa Symbol ? String(last(fn.args)) : ""
        else
            ""
        end
        if name in ("Normal", "Uniform", "Beta", "Gamma", "Exponential", "InverseGamma", "Bernoulli", "Binomial", "Poisson")
            return :(Float64)
        elseif occursin("Mv", name) || name in ("MvNormal", "MvNormalDiag", "Dirichlet")
            return Expr(:curly, :Vector, :Float64)
        else
            return :(Any)
        end
    else
        return :(Any)
    end
end

function build_output_dod(modeldef, linenumbernode)
    args = transform_args(modeldef[:args])
    kwargs = transform_args(modeldef[:kwargs])

    # Need to update `args` and `kwargs` since we might have added `TypeWrap` to the types.
    modeldef[:args] = args
    modeldef[:kwargs] = kwargs

    ## Build the anonymous evaluator from the user-provided model definition.
    evaluatordef = copy(modeldef)

    # Add the internal arguments to the user-specified arguments (positional + keywords).
    # Use abstract types so the evaluator accepts either `Model` or `ModelDOD` and
    # `AbstractVarInfo` (wrappers may convert between ModelDOD and Model).
    # For DOD models, accept a concrete `DODVarInfo` so hot-path field accesses
    # (e.g., `acc_updater`, `metas`, `typed_slots`) can be type-specialized by
    # the compiler and avoid boxing/dispatch overhead.
    evaluatordef[:args] = vcat(
        [:(__model__::$(DynamicPPL.AbstractProbabilisticProgram)), :(__varinfo__::($(DynamicPPL.DODVarInfo){AU} where AU))],
        args,
    )

    evaluatordef[:body] = MacroTools.@q begin
        $(linenumbernode)
        $(replace_returns(add_return_to_last_statment(modeldef[:body])))
    end

    ## Build the model function.

    if MacroTools.@capture(modeldef[:name], ::T_)
        name = gensym(:f)
        modeldef[:name] = Expr(:(::), name, T)
    elseif MacroTools.@capture(modeldef[:name], (name_::_ | name_))
    else
        throw(ArgumentError("unsupported format of model function"))
    end

    args_split = map(MacroTools.splitarg, args)
    kwargs_split = map(MacroTools.splitarg, kwargs)
    args_nt = namedtuple_from_splitargs(args_split)
    kwargs_inclusion = map(splitarg_to_expr, kwargs_split)

    # Build expressions that construct the vector of meta VarNames, types and lengths
    meta_vec_expr = Expr(:vect, (modeldef[:meta_vns])...)
    # meta types -> use inferred types if available, default to Any
    if haskey(modeldef, :meta_types)
        meta_types_expr = Expr(:vect, (modeldef[:meta_types])...)
    else
        meta_types_expr = Expr(:vect, (fill(:(Any), length(modeldef[:meta_vns]))...))
    end
    meta_lens_expr = Expr(:vect, (fill(:(1), length(modeldef[:meta_vns]))...))
    modeldef[:body] = MacroTools.@q begin
        $(linenumbernode)
        return $(DynamicPPL.ModelDOD)($name, $args_nt, (; $(kwargs_inclusion...)); meta_vns=$(meta_vec_expr), meta_types=$(meta_types_expr), meta_lens=$(meta_lens_expr))
    end

    return MacroTools.@q begin
        $(MacroTools.combinedef(evaluatordef))
        $(Base).@__doc__ $(MacroTools.combinedef(modeldef))
    end
end

function model(mod, linenumbernode, expr, warn)
    modeldef = build_model_definition(expr)

    # Generate main body
    modeldef[:body] = generate_mainbody(mod, modeldef[:body], warn)

    return build_output(modeldef, linenumbernode)
end

"""
    build_model_definition(input_expr)

Builds the `modeldef` dictionary from the model's expression, where
`modeldef` is a dictionary compatible with `MacroTools.combinedef`.
"""
function build_model_definition(input_expr)
    # Break up the model definition and extract its name, arguments, and function body
    modeldef = MacroTools.splitdef(input_expr)

    # Check that the function has a name
    # https://github.com/TuringLang/DynamicPPL.jl/issues/260
    haskey(modeldef, :name) ||
        throw(ArgumentError("anonymous functions without name are not supported"))

    # Print a warning if function body of the model is empty
    warn_empty(modeldef[:body])

    ## Construct model_info dictionary

    # Shortcut if the model does not have any arguments
    if !haskey(modeldef, :args) && !haskey(modeldef, :kwargs)
        return modeldef
    end

    # Ensure that all arguments have a name, i.e., are of the form `name` or `name::T`
    addargnames!(modeldef[:args])

    return modeldef
end

"""
    generate_mainbody(mod, expr, warn)

Generate the body of the main evaluation function from expression `expr` and arguments
`args`.

If `warn` is true, a warning is displayed if internal variables are used in the model
definition.
"""
generate_mainbody(mod, expr, warn; dod=false) = generate_mainbody!(mod, Symbol[], expr, warn; dod=dod)

generate_mainbody!(mod, found, x, warn; dod=false) = x
function generate_mainbody!(mod, found, sym::Symbol, warn; dod=false)
    if warn && sym in INTERNALNAMES && sym ∉ found
        @warn "you are using the internal variable `$sym`"
        push!(found, sym)
    end

    return sym
end
function generate_mainbody!(mod, found, expr::Expr, warn; dod=false)
    # Do not touch interpolated expressions
    expr.head === :$ && return expr.args[1]

    # Do we don't want escaped expressions because we unfortunately
    # escape the entire body afterwards.
    Meta.isexpr(expr, :escape) && return generate_mainbody(mod, found, expr.args[1], warn)

    # If it's a macro, we expand it
    if Meta.isexpr(expr, :macrocall)
        return generate_mainbody!(mod, found, macroexpand(mod, expr; recursive=true), warn)
    end

    # Modify dotted tilde operators.
    args_dottilde = getargs_dottilde(expr)
    if args_dottilde !== nothing
        L, R = args_dottilde
        return generate_mainbody!(
            mod, found, Base.remove_linenums!(generate_dot_tilde(L, R)), warn; dod=dod
        )
    end

    # Modify tilde operators.
    args_tilde = getargs_tilde(expr)
    if args_tilde !== nothing
        L, R = args_tilde
        return Base.remove_linenums!(
            generate_tilde(
                generate_mainbody!(mod, found, L, warn; dod=dod),
                generate_mainbody!(mod, found, R, warn; dod=dod),
                dod=dod,
            ),
        )
    end

    # Modify the assignment operators.
    args_assign = getargs_coloneq(expr)
    if args_assign !== nothing
        L, R = args_assign
        return Base.remove_linenums!(
            generate_assign(
                generate_mainbody!(mod, found, L, warn; dod=dod),
                generate_mainbody!(mod, found, R, warn; dod=dod),
            ),
        )
    end

    return Expr(expr.head, map(x -> generate_mainbody!(mod, found, x, warn), expr.args)...)
end

function generate_assign(left, right)
    # A statement `x := y` reduces to `x = y`, but if __varinfo__ has an accumulator for
    # ValuesAsInModel then in addition we push! the pair of `x` and `y` to the accumulator.
    @gensym acc right_val vn
    return quote
        $right_val = $right
        if $(DynamicPPL.is_extracting_values)(__varinfo__)
            $vn = $(DynamicPPL.prefix)(__model__.context, $(make_varname_expression(left)))
            __varinfo__ = $(map_accumulator!!)(
                $acc -> push!($acc, $vn, $right_val), __varinfo__, Val(:ValuesAsInModel)
            )
        end
        $left = $right_val
    end
end

function generate_tilde_literal(left, right)
    # If the LHS is a literal, it is always an observation
    @gensym value
    return quote
        $value, __varinfo__ = $(DynamicPPL.tilde_observe!!)(
            __model__.context,
            $(DynamicPPL.check_tilde_rhs)($right),
            $left,
            nothing,
            __varinfo__,
        )
        $value
    end
end

"""
    generate_tilde(left, right)

Generate an `observe` expression for data variables and `assume` expression for parameter
variables.
"""
function generate_tilde(left, right; dod=false)
    isliteral(left) && return generate_tilde_literal(left, right)

    # Otherwise it is determined by the model or its value,
    # if the LHS represents an observation
    @gensym vn isassumption value dist

    return quote
        $dist = $right
        $vn = $(DynamicPPL.resolve_varnames)($(make_varname_expression(left)), $dist)
        $isassumption = $(DynamicPPL.isassumption(left, vn))
        if $(DynamicPPL.isfixed(left, vn))
            $left = $(DynamicPPL.getfixed_nested)(
                __model__.context, $(DynamicPPL.prefix)(__model__.context, $vn)
            )
        elseif $isassumption
            $(generate_tilde_assume(left, dist, vn; dod=dod))
        else
            # If `vn` is not in `argnames`, we need to make sure that the variable is defined.
            if !$(DynamicPPL.inargnames)($vn, __model__)
                $left = $(DynamicPPL.getconditioned_nested)(
                    __model__.context, $(DynamicPPL.prefix)(__model__.context, $vn)
                )
            end

            $value, __varinfo__ = $(dod ? :(DynamicPPL.tilde_observe_dod!!) : :(DynamicPPL.tilde_observe!!))(
                __model__.context,
                $(DynamicPPL.check_tilde_rhs)($dist),
                $(maybe_view(left)),
                $vn,
                __varinfo__,
            )
            $value
        end
    end
end

function generate_tilde_assume(left, right, vn; dod=false)
    # HACK: Because the Setfield.jl macro does not support assignment
    # with multiple arguments on the LHS, we need to capture the return-values
    # and then update the LHS variables one by one.
    @gensym value
    expr = :($left = $value)
    if left isa Expr
        expr = AbstractPPL.drop_escape(
            Accessors.setmacro(BangBang.prefermutation, expr; overwrite=true)
        )
    end

    if !dod
        return quote
            $value, __varinfo__ = $(DynamicPPL.tilde_assume!!)(
                __model__.context,
                $(DynamicPPL.unwrap_right_vn)($(DynamicPPL.check_tilde_rhs)($right), $vn)...,
                __varinfo__,
            )
            $expr
            $value
        end
    else
        # DOD path: use the precomputed local meta_<hash> binding created by
        # `@model_dod` at evaluator entry. Emit an inlined fast-path that uses
        # the integer meta index to avoid per-site closures and small-dispatches.
        # Fall back to the VarName-based wrapper when the meta binding is zero.
        meta_sym = meta_sym_for_vn(vn)
        return quote
            # Fallback to non-meta path if the meta local is not set
            if $(meta_sym) == 0
                $value, __varinfo__ = $(DynamicPPL.tilde_assume_dod!!)(
                    __model__.context,
                    $(DynamicPPL.unwrap_right_vn)($(DynamicPPL.check_tilde_rhs)($right), $vn)...,
                    $vn,
                    __varinfo__,
                )
            else
                # Fast-path: inline the DOD meta-indexed assume handling.
                meta_i = $(meta_sym)
                # access meta and backing vector directly to avoid function call
                meta = __varinfo__.metas[meta_i]
                if meta.idx == 0
                    # variable not present: fall back to generic wrapper to preserve semantics
                    $value, __varinfo__ = $(DynamicPPL.tilde_assume_dod!!)(
                        __model__.context,
                        $(DynamicPPL.unwrap_right_vn)($(DynamicPPL.check_tilde_rhs)($right), $vn)...,
                        $vn,
                        __varinfo__,
                    )
                else
                    # Prefer per-model typed slot local (prebound at entry) if available
                    typed_local = $(meta_typed_sym_for_vn(vn))
                    vec = typed_local === nothing ? (meta.vec === nothing ? __varinfo__.values[meta.type] : meta.vec) : typed_local
                    # read internal value without creating slices. Use the
                    # prebound per-meta length local (meta_len_<hash>) so the
                    # compiler can constant-propagate small fixed lengths when
                    # available. We keep the `view` for reads to preserve the
                    # expected input shape for transforms.
                    if $(meta_len_sym_for_vn(vn)) == 1
                        y = vec[meta.idx]
                    else
                        # for multivariate, use a non-allocating view into the backing
                        # storage to avoid creating temporary arrays when calling transforms.
                        y = view(vec, meta.idx:(meta.idx + $(meta_len_sym_for_vn(vn)) - 1))
                    end
                    f = $(DynamicPPL.from_maybe_linked_internal_transform)(__varinfo__, $vn, $right)
                    x, inv_logjac = $(DynamicPPL.with_logabsdet_jacobian)(f, y)

                    # update accumulators: prefer a cached typed updater to avoid
                    # exception-driven fallbacks and small-dispatch overhead.
                        if __acc_updater__ !== nothing
                            $(DynamicPPL.acc_updater_assume)(__acc_updater__, __varinfo__, x, -inv_logjac, $vn, $right)
                        elseif __varinfo__.acc_default !== nothing
                        # backward-compatible path: update holder via the generic API
                        h = __varinfo__.acc_default
                        h.lp = $(DynamicPPL.accumulate_assume!!)(h.lp, x, -inv_logjac, $vn, $right)
                        h.lj = $(DynamicPPL.accumulate_assume!!)(h.lj, x, -inv_logjac, $vn, $right)
                        h.ll = $(DynamicPPL.accumulate_assume!!)(h.ll, x, -inv_logjac, $vn, $right)
                    else
                        __varinfo__ = $(DynamicPPL.map_accumulators!!)(a->$(DynamicPPL.accumulate_assume!!)(a, x, -inv_logjac, $vn, $right), __varinfo__)
                    end

                    # write back the internal value without range allocations.
                    # For very small fixed lengths we emit unrolled element
                    # assignments to avoid the temporary `SubArray` allocation
                    # and bounds-check overhead of `copyto!` on the hot path.
                    if $(meta_len_sym_for_vn(vn)) == 1
                        vec[meta.idx] = x
                    elseif $(meta_len_sym_for_vn(vn)) == 2
                        vec[meta.idx] = x[1]
                        vec[meta.idx + 1] = x[2]
                    elseif $(meta_len_sym_for_vn(vn)) == 3
                        vec[meta.idx] = x[1]
                        vec[meta.idx + 1] = x[2]
                        vec[meta.idx + 2] = x[3]
                    else
                        # fallback: generic copy for larger or unknown lengths
                        copyto!(view(vec, meta.idx:(meta.idx + meta.len - 1)), x)
                    end

                    $value = x
                end
            end
            $expr
            $value
        end
    end
end

"""
    generate_dot_tilde(left, right)

Generate the expression that replaces `left .~ right` in the model body.
"""
function generate_dot_tilde(left, right)
    @gensym dist left_axes idx
    return quote
        $dist = $(DynamicPPL.check_dot_tilde_rhs)($right)
        $left_axes = axes($left)
        for $idx in Iterators.product($left_axes...)
            $left[$idx...] ~ $dist
        end
    end
end

# Note that we cannot use `MacroTools.isdef` because
# of https://github.com/FluxML/MacroTools.jl/issues/154.
"""
    isfuncdef(expr)

Return `true` if `expr` is any form of function definition, and `false` otherwise.
"""
function isfuncdef(e::Expr)
    return if Meta.isexpr(e, :function)
        # Classic `function f(...)`
        true
    elseif Meta.isexpr(e, :->)
        # Anonymous functions/lambdas, e.g. `do` blocks or `->` defs.
        true
    elseif Meta.isexpr(e, :(=)) && Meta.isexpr(e.args[1], :call)
        # Short function defs, e.g. `f(args...) = ...`.
        true
    else
        false
    end
end

"""
    replace_returns(expr)

Return `Expr` with all `return ...` statements replaced with
`return ..., DynamicPPL.return_values(__varinfo__)`.

Note that this method will _not_ replace `return` statements within function
definitions. This is checked using [`isfuncdef`](@ref).
"""
replace_returns(e) = e
function replace_returns(e::Expr)
    isfuncdef(e) && return e

    if Meta.isexpr(e, :return)
        # We capture the original return-value in `retval` and return
        # a `Tuple{typeof(retval),typeof(__varinfo__)}`.
        # If we don't capture the return-value separately, cases such as
        # `return x = 1` will result in `(x = 1, __varinfo__)` which will
        # mistakenly attempt to construct a `NamedTuple` (which fails on Julia 1.3
        # and is not our intent).
        @gensym retval
        return quote
            $retval = $(map(replace_returns, e.args)...)
            return $retval, __varinfo__
        end
    end

    return Expr(e.head, map(replace_returns, e.args)...)
end

# If it's just a symbol, e.g. `f(x) = 1`, then we make it `f(x) = return 1`.
add_return_to_last_statment(body) = Expr(:return, body)
function add_return_to_last_statment(body::Expr)
    # If the last statement is a return-statement, we don't do anything.
    # Otherwise we replace the last statement with a `return` statement.
    Meta.isexpr(body.args[end], :return) && return body
    # We need to copy the arguments since we are modifying them.
    new_args = copy(body.args)
    new_args[end] = Expr(:return, body.args[end])
    return Expr(body.head, new_args...)
end

const FloatOrArrayType = Type{<:Union{AbstractFloat,AbstractArray}}
hasmissing(::Type) = false
hasmissing(::Type{>:Missing}) = true
hasmissing(::Type{<:AbstractArray{TA}}) where {TA} = hasmissing(TA)
hasmissing(::Type{Union{}}) = false # issue #368

"""
    TypeWrap{T}

A wrapper type used internally to make expressions such as `::Type{TV}` in the model arguments
not ending up as a `DataType`.
"""
struct TypeWrap{T} end

function arg_type_is_type(e)
    return Meta.isexpr(e, :curly) && length(e.args) > 1 && e.args[1] === :Type
end

function splitarg_to_expr((arg_name, arg_type, is_splat, default))
    return is_splat ? :($arg_name...) : arg_name
end

"""
    transform_args(args)

Return transformed `args` used in both the model constructor and evaluator.

Specifically, this replaces expressions of the form `::Type{TV}=Vector{Float64}`
with `::TypeWrap{TV}=TypeWrap{Vector{Float64}}()` to avoid introducing `DataType`.
"""
function transform_args(args)
    splitargs = map(args) do arg
        arg_name, arg_type, is_splat, default = MacroTools.splitarg(arg)
        return if arg_type_is_type(arg_type)
            arg_name, :($TypeWrap{$(arg_type.args[2])}), is_splat, :($TypeWrap{$default}())
        else
            arg_name, arg_type, is_splat, default
        end
    end
    return map(Base.splat(MacroTools.combinearg), splitargs)
end

function namedtuple_from_splitargs(splitargs)
    names = map(splitargs) do (arg_name, arg_type, is_splat, default)
        is_splat ? Symbol("#splat#$(arg_name)") : arg_name
    end
    names_expr = Expr(:tuple, map(QuoteNode, names)...)
    vals = Expr(:tuple, map(first, splitargs)...)
    return :(NamedTuple{$names_expr}($vals))
end

"""
    build_output(modeldef, linenumbernode)

Builds the output expression.
"""
function build_output(modeldef, linenumbernode)
    args = transform_args(modeldef[:args])
    kwargs = transform_args(modeldef[:kwargs])

    # Need to update `args` and `kwargs` since we might have added `TypeWrap` to the types.
    modeldef[:args] = args
    modeldef[:kwargs] = kwargs

    ## Build the anonymous evaluator from the user-provided model definition.
    evaluatordef = copy(modeldef)

    # Add the internal arguments to the user-specified arguments (positional + keywords).
    evaluatordef[:args] = vcat(
        [:(__model__::$(DynamicPPL.Model)), :(__varinfo__::$(DynamicPPL.AbstractVarInfo))],
        args,
    )

    # Replace the user-provided function body with the version created by DynamicPPL.
    # We use `MacroTools.@q begin ... end` instead of regular `quote ... end` to ensure
    # that no new `LineNumberNode`s are added apart from the reference `linenumbernode`
    # to the call site.
    # NOTE: We need to replace statements of the form `return ...` with
    # `return (..., __varinfo__)` to ensure that the second
    # element in the returned value is always the most up-to-date `__varinfo__`.
    # See the docstrings of `replace_returns` for more info.
    evaluatordef[:body] = MacroTools.@q begin
        $(linenumbernode)
        $(replace_returns(add_return_to_last_statment(modeldef[:body])))
    end

    ## Build the model function.

    # Obtain or generate the name of the model to support functors:
    # https://github.com/TuringLang/DynamicPPL.jl/issues/367
    if MacroTools.@capture(modeldef[:name], ::T_)
        name = gensym(:f)
        modeldef[:name] = Expr(:(::), name, T)
    elseif MacroTools.@capture(modeldef[:name], (name_::_ | name_))
    else
        throw(ArgumentError("unsupported format of model function"))
    end

    args_split = map(MacroTools.splitarg, args)
    kwargs_split = map(MacroTools.splitarg, kwargs)
    args_nt = namedtuple_from_splitargs(args_split)
    kwargs_inclusion = map(splitarg_to_expr, kwargs_split)

    # Update the function body of the user-specified model.
    # We use `MacroTools.@q begin ... end` instead of regular `quote ... end` to ensure
    # that no new `LineNumberNode`s are added apart from the reference `linenumbernode`
    # to the call site
    modeldef[:body] = MacroTools.@q begin
        $(linenumbernode)
        return $(DynamicPPL.Model)($name, $args_nt; $(kwargs_inclusion...))
    end

    return MacroTools.@q begin
        $(MacroTools.combinedef(evaluatordef))
        $(Base).@__doc__ $(MacroTools.combinedef(modeldef))
    end
end

function warn_empty(body)
    if all(l -> isa(l, LineNumberNode), body.args)
        @warn("Model definition seems empty, still continue.")
    end
    return nothing
end

# TODO(mhauru) matchingvalue has methods that can accept both types and values. Why?
# TODO(mhauru) This function needs a more comprehensive docstring.
"""
    matchingvalue(vi, value)

Convert the `value` to the correct type for the `vi` object.
"""
function matchingvalue(vi, value)
    T = typeof(value)
    if hasmissing(T)
        _value = convert(get_matching_type(vi, T), value)
        # TODO(mhauru) Why do we make a deepcopy, even though in the !hasmissing branch we
        # are happy to return `value` as-is?
        if _value === value
            return deepcopy(_value)
        else
            return _value
        end
    else
        return value
    end
end

function matchingvalue(vi, value::FloatOrArrayType)
    return get_matching_type(vi, value)
end
function matchingvalue(vi, ::TypeWrap{T}) where {T}
    return TypeWrap{get_matching_type(vi, T)}()
end

# TODO(mhauru) This function needs a more comprehensive docstring. What is it for?
"""
    get_matching_type(vi, ::TypeWrap{T}) where {T}

Get the specialized version of type `T` for `vi`.
"""
get_matching_type(_, ::Type{T}) where {T} = T
function get_matching_type(vi, ::Type{<:Union{Missing,AbstractFloat}})
    return Union{Missing,float_type_with_fallback(eltype(vi))}
end
function get_matching_type(vi, ::Type{<:AbstractFloat})
    return float_type_with_fallback(eltype(vi))
end
function get_matching_type(vi, ::Type{<:Array{T,N}}) where {T,N}
    return Array{get_matching_type(vi, T),N}
end
function get_matching_type(vi, ::Type{<:Array{T}}) where {T}
    return Array{get_matching_type(vi, T)}
end
