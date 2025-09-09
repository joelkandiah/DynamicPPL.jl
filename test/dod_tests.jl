using Test
using DynamicPPL
using Distributions

@testset "DOD model parity" begin
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

    # simple equality checks for logprior/loglikelihood between baseline and DOD
    vi_base = SimpleVarInfo((m = 0.2, x = 0.0))
    vi_dod = DODVarInfo()
    add_variable!(vi_dod, @varname(m), 0.2, dist = Normal(0,1))
    add_variable!(vi_dod, @varname(x), 0.0, dist = Normal(0.2,1))

    @test isapprox(logprior(simple_model(), vi_base), logprior(simple_model_dod(), vi_dod), atol=1e-12)
    @test isapprox(loglikelihood(simple_model(), vi_base), loglikelihood(simple_model_dod(), vi_dod), atol=1e-12)

    # sampling parity (small sample sizes to keep tests fast)
    s_base = sample(simple_model(), SampleFromPrior(), 50; progress=false)
    s_dod = sample(simple_model_dod(), SampleFromPrior(), 50; progress=false)
    @test length(s_base) == length(s_dod) == 50
end
