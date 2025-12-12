module DynamicPPL

using AbstractMCMC: AbstractSampler, AbstractChains
using AbstractPPL
using Bijectors
using Compat
using Distributions
using OrderedCollections: OrderedCollections, OrderedDict
using Dictionaries: Dictionaries, Dictionary, dictionary, set
using Printf: Printf

using AbstractMCMC: AbstractMCMC
using ADTypes: ADTypes
using BangBang: BangBang, push!!, empty!!, setindex!!
using MacroTools: MacroTools
using ConstructionBase: ConstructionBase
using Accessors: Accessors
using LogDensityProblems: LogDensityProblems

using LinearAlgebra: LinearAlgebra, Cholesky

using DocStringExtensions

using Random: Random

# For extending
import AbstractPPL: predict, hasvalue, getvalue

# TODO: Remove these when it's possible.
import Bijectors: link, invlink

import Base:
    Symbol,
    ==,
    hash,
    getindex,
    setindex!,
    push!,
    show,
    isempty,
    empty!,
    getproperty,
    setproperty!,
    keys,
    haskey

# VarInfo
export AbstractVarInfo,
    VarInfo,
    SimpleVarInfo,
    AbstractAccumulator,
    LogLikelihoodAccumulator,
    LogPriorAccumulator,
    LogJacobianAccumulator,
    DictVarInfo,
    UntypedVarInfo,
    typed_varinfo,
    typed_dict_varinfo,
    getlogp,
    setlogp!!,
    acclogp!!,
    resetlogp!!,
    get_num_produce,
    set_num_produce!,
    reset_num_produce!,
    increment_num_produce!,
    set_retained_vns_del_by_spl!,
    is_flagged,
    set_flag!,
    unset_flag!,
    set_gid!,
    updategid!,
    setorder!,
    istrans,
    link!,
    invlink!,
    tonamedtuple,
    values_as,
    # VarName
    VarName,
    inspace,
    subsumes,
    @varname,
    # Compiler
    @model,
    # Utilities
    OrderedDict,
    vectorize,
    reconstruct,
    reconstruct,
    Sample,
    Chain,
    init,
    solve,
    MCMCThreads,
    MCMCDistributed,
    MCMCSerial,
    # Model
    Model,
    getmissings,
    getargnames,
    setthreadsafe,
    requires_threadsafe,
    extract_priors,
    values_as_in_model,
    # evaluation
    evaluate!!,
    init!!,
    # LogDensityFunction
    LogDensityFunction,
    OnlyAccsVarInfo,
    # Leaf contexts
    AbstractContext,
    contextualize,
    DefaultContext,
    InitContext,
    # Parent contexts
    AbstractParentContext,
    childcontext,
    setchildcontext,
    leafcontext,
    setleafcontext,
    # Tilde pipeline
    tilde_assume!!,
    tilde_observe!!,
    # Probabilistic
    LikelihoodContext,
    PriorContext,
    MiniBatchContext,
    PrefixContext,
    ConditionContext,
    assume,
    dot_assume,
    observe,
    dot_observe,
    tilde_assume,
    tilde_observe,
    dot_tilde_assume,
    dot_tilde_observe,
    # Pseudo-marginal
    PseudoMarginalContext,
    # Initialization
    AbstractInitStrategy,
    InitFromPrior,
    InitFromUniform,
    InitFromParams,
    get_param_eltype,
    # Pseudo distributions
    NamedDist,
    NoDist,
    # Convenience functions
    logjoint,
    logprior,
    loglikelihood,
    pointwise_prior_logdensities,
    pointwise_logdensities,
    pointwise_loglikelihoods,
    condition,
    decondition,
    fix,
    unfix,
    predict,
    marginalize,
    prefix,
    returned,
    to_submodel,
    # Selection
    @submodel,
    # Struct to hold model outputs
    ParamsWithStats,
    # Convenience macros
    @addlogprob!,
    value_iterator_from_chain,
    check_model,
    check_model_and_trace,
    # Deprecated.
    @logprob_str,
    @prob_str,
    generated_quantities

# Reexport
using Distributions: loglikelihood
export loglikelihood

# TODO: Remove once we feel comfortable people aren't using it anymore.
macro logprob_str(str)
    return :(error(
        "The `@logprob_str` macro is no longer supported. See https://turinglang.org/dev/docs/using-turing/guide/#querying-probabilities-from-model-or-chain for information on how to query probabilities, and https://github.com/TuringLang/DynamicPPL.jl/issues/356 for information regarding its removal.",
    ))
end

macro prob_str(str)
    return :(error(
        "The `@prob_str` macro is no longer supported. See https://turinglang.org/dev/docs/using-turing/guide/#querying-probabilities-from-model-or-chain for information on how to query probabilities, and https://github.com/TuringLang/DynamicPPL.jl/issues/356 for information regarding its removal.",
    ))
end

# TODO(mhauru) We should write down the list of methods that any subtype of AbstractVarInfo
# has to implement. Not sure what the full list is for parameters values, but for
# accumulators we only need `getaccs` and `setaccs!!`.
"""
    AbstractVarInfo

Abstract supertype for data structures that capture random variables when executing a
probabilistic model and accumulate log densities such as the log likelihood or the
log joint probability of the model.

See also: [`VarInfo`](@ref), [`SimpleVarInfo`](@ref).
"""
abstract type AbstractVarInfo <: AbstractModelTrace end

# Necessary forward declarations
include("utils.jl")
include("contexts.jl")
include("contexts/default.jl")
include("contexts/init.jl")
include("contexts/transformation.jl")
include("contexts/prefix.jl")
include("contexts/conditionfix.jl")  # Must come after contexts/prefix.jl
include("model.jl")
include("varname.jl")
include("distribution_wrappers.jl")
include("submodel.jl")
include("varnamedvector.jl")
include("accumulators.jl")
include("default_accumulators.jl")
include("abstract_varinfo.jl")
include("threadsafe.jl")
include("varinfo.jl")
include("dict_varinfo.jl")
include("simple_varinfo.jl")
include("onlyaccs.jl")
include("compiler.jl")
include("pointwise_logdensities.jl")
include("logdensityfunction.jl")
include("model_utils.jl")
include("extract_priors.jl")
include("values_as_in_model.jl")
include("experimental.jl")
include("chains.jl")
include("bijector.jl")

include("debug_utils.jl")
using .DebugUtils
include("test_utils.jl")

include("deprecated.jl")

if isdefined(Base.Experimental, :register_error_hint)
    function __init__()
        # Better error message if users forget to load JET.jl
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, _
            requires_jet =
                exc.f === DynamicPPL.Experimental._determine_varinfo_jet &&
                length(argtypes) >= 2 &&
                argtypes[1] <: Model &&
                argtypes[2] <: AbstractContext
            requires_jet |=
                exc.f === DynamicPPL.Experimental.is_suitable_varinfo &&
                length(argtypes) >= 3 &&
                argtypes[1] <: Model &&
                argtypes[2] <: AbstractContext &&
                argtypes[3] <: AbstractVarInfo
            if requires_jet
                print(
                    io,
                    "\n$(exc.f) requires JET.jl to be loaded. Please run `using JET` before calling $(exc.f).",
                )
            end
        end

        # Same for MarginalLogDensities.jl
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, _
            requires_mld =
                exc.f === DynamicPPL.marginalize &&
                length(argtypes) == 2 &&
                argtypes[1] <: Model &&
                argtypes[2] <: AbstractVector{<:Union{Symbol,<:VarName}}
            if requires_mld
                printstyled(
                    io,
                    "\n\n    `$(exc.f)` requires MarginalLogDensities.jl to be loaded.\n    Please run `using MarginalLogDensities` before calling `$(exc.f)`.\n";
                    color=:cyan,
                    bold=true,
                )
            end
        end

        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, _
            is_evaluate_three_arg =
                exc.f === AbstractPPL.evaluate!! &&
                length(argtypes) == 3 &&
                argtypes[1] <: Model &&
                argtypes[2] <: AbstractVarInfo &&
                argtypes[3] <: AbstractContext
            if is_evaluate_three_arg
                print(
                    io,
                    "\n\nThe method `evaluate!!(model, varinfo, new_ctx)` has been removed. Instead, you should store the `new_ctx` in the `model.context` field using `new_model = contextualize(model, new_ctx)`, and then call `evaluate!!(new_model, varinfo)` on the new model. (Note that, if the model already contained a non-default context, you will need to wrap the existing context.)",
                )
            end
        end
    end
end

# Standard tag: Improves stacktraces
# Ref: https://www.stochasticlifestyle.com/improved-forwarddiff-jl-stacktraces-with-package-tags/
struct DynamicPPLTag end

# Extended in MarginalLogDensitiesExt
function marginalize end

end # module
