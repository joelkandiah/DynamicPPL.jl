using DynamicPPL, Distributions
using Random
using LinearAlgebra

@model function simple_model()
    m ~ Normal(0, 1)
    x ~ Normal(m, 1)
    return (m = m, x = x)
end

@model_dod function simple_model_dod()
    m ~ Normal(0, 1)
    x ~ Normal(m, 1)
    return (m = m, x = x)
end

function smoke()
    try
        println("SMOKE: starting")
        # baseline: simple model with SimpleVarInfo and @model
        println("SMOKE: constructing SimpleVarInfo for @model")
        vi_simple = SimpleVarInfo((m = 0.5, x = 0.0))
        println("SMOKE: computing baseline accumulators (@model)")
        r1 = logprior(simple_model(), vi_simple)
        println("SMOKE: baseline logprior=", r1)
        r2 = loglikelihood(simple_model(), vi_simple)
        println("SMOKE: baseline loglikelihood=", r2)

        # construct a DODVarInfo directly (avoid Metadata misuse in the test)
        println("SMOKE: constructing DODVarInfo for @model_dod")
        vi_dod = DODVarInfo()
        add_variable!(vi_dod, @varname(m), 0.5, dist = Normal(0, 1))
        add_variable!(vi_dod, @varname(x), 0.0, dist = Normal(0.5, 1))
        println("SMOKE: calling DOD logprior/loglikelihood (@model_dod)")
        r1_dod = logprior(simple_model_dod(), vi_dod)
        println("SMOKE: dod logprior=", r1_dod)
        r2_dod = loglikelihood(simple_model_dod(), vi_dod)
        println("SMOKE: dod loglikelihood=", r2_dod)

        println("SMOKE: finished successfully")
        return (r1, r2, r1_dod, r2_dod)
    catch e
        println("SMOKE ERROR: ", e)
        Base.show_backtrace(stderr, catch_backtrace())
        rethrow()
    end
end

println("RUN: ", try smoke() catch e; "failed" end)

# Additional smoke tests requested by developer
function smoke_more()
    println("SMOKE_MORE: starting extended checks")
    # 1) Sampling from the prior using SampleFromPrior
    println("SMOKE_MORE: sampling from prior")
    vi_sp = DODVarInfo()
    add_variable!(vi_sp, @varname(m), 0.0, dist = Normal(0,1))
    # sample via tilde_assume_dod!! sampler-aware entry
    r, vi_sp2 = tilde_assume_dod!!(Random.default_rng(), SampleFromPrior(), Normal(0,1), get(vi_sp.idcs, @varname(m), 0), vi_sp)
    println("SMOKE_MORE: sampled value for m = ", r)

    # 1b) Sampling via top-level `sample()` API (mirror lkj.jl usage)
    println("SMOKE_MORE: sampling via sample() API")
    s_base = sample(simple_model(), SampleFromPrior(), 200; progress=false)
    s_dod = sample(simple_model_dod(), SampleFromPrior(), 200; progress=false)
    println("SMOKE_MORE: sample counts base=", length(s_base), ", dod=", length(s_dod))

    # 2) Multivariate distribution
    println("SMOKE_MORE: multivariate distribution test")
    @model function mv_model()
        v ~ MvNormal([0.0, 0.0], Matrix{Float64}(I, 2, 2))
        return (v = v,)
    end
    @model_dod function mv_model_dod()
        v ~ MvNormal([0.0, 0.0], Matrix{Float64}(I, 2, 2))
        return (v = v,)
    end
    vi_mv = SimpleVarInfo((v = [0.0, 0.0],))
    vi_mv_dod = DODVarInfo()
    add_variable!(vi_mv_dod, @varname(v), [0.0, 0.0], dist = MvNormal([0.0, 0.0], Matrix{Float64}(I,2,2)))
    lp_base = logprior(mv_model(), vi_mv)
    lp_dod = logprior(mv_model_dod(), vi_mv_dod)
    println("SMOKE_MORE: mv logprior base=", lp_base, " dod=", lp_dod)

    # 3) Constrained distribution (Beta on (0,1)) with link handling
    println("SMOKE_MORE: constrained distribution (Beta) test")
    @model function beta_model()
        p ~ Beta(2.0, 5.0)
        return (p = p,)
    end
    @model_dod function beta_model_dod()
        p ~ Beta(2.0, 5.0)
        return (p = p,)
    end
    vi_b = SimpleVarInfo((p = 0.2,))
    vi_b_dod = DODVarInfo()
    add_variable!(vi_b_dod, @varname(p), 0.2, dist = Beta(2.0, 5.0))
    lp_b = logprior(beta_model(), vi_b)
    lp_b_dod = logprior(beta_model_dod(), vi_b_dod)
    println("SMOKE_MORE: beta logprior base=", lp_b, " dod=", lp_b_dod)

    # 4) Non-zero loglikelihood: observe data
    println("SMOKE_MORE: non-zero loglikelihood test")
    @model function obs_model(y)
        μ ~ Normal(0,1)
        y ~ Normal(μ, 1)
        return (μ = μ,)
    end
    @model_dod function obs_model_dod(y)
        μ ~ Normal(0,1)
        y ~ Normal(μ, 1)
        return (μ = μ,)
    end
    y = 2.0
    # construct VarInfo with observed y via passing as argument to evaluate
    lp_obs_base = begin
        vi = SimpleVarInfo((μ = 0.0, y = y))
        loglikelihood(obs_model(y), vi)
    end
    lp_obs_dod = begin
        vi = DODVarInfo()
        add_variable!(vi, @varname(μ), 0.0, dist=Normal(0,1))
        add_variable!(vi, @varname(y), y, dist=Normal(0.0,1.0))
        loglikelihood(obs_model_dod(y), vi)
    end
    println("SMOKE_MORE: obs loglik base=", lp_obs_base, " dod=", lp_obs_dod)

    println("SMOKE_MORE: finished")
    return true
end

println("RUN_MORE: ", try smoke_more() catch e; "failed" end)
