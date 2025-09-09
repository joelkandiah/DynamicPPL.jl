using DynamicPPL
using Distributions
using Test

@testset "compiler_tests: ModelDOD emission parity" begin
    @model function demo(x)
        m ~ Normal(0.0, 1.0)
        x ~ Normal(m, 1)
        return nothing
    end

    m_plain = demo(1.0)
    @model_dod function demo_dod(x)
        m ~ Normal(0.0, 1.0)
        x ~ Normal(m, 1)
        return nothing
    end
    m_dod = demo_dod(1.0)

    # ensure sampler returns DODVarInfo for emitted models when used
    vi_plain = VarInfo(m_plain)
    vi_dod = DODVarInfo(m_dod)

    # evaluate once to populate
    _, vi_plain_eval = DynamicPPL.evaluate!!(m_plain, deepcopy(vi_plain))
    # Wrap `ModelDOD` into a `Model` but avoid using `m_dod.defaults` which may
    # contain `nothing` placeholders; use an empty `NamedTuple` for defaults to
    # prevent `nothing` being used as distribution parameters in the emitted code.
    m_dod_model = DynamicPPL.Model(m_dod.f, m_dod.args, NamedTuple(), m_dod.context)
    _, vi_dod_eval = DynamicPPL.evaluate!!(m_dod_model, deepcopy(vi_dod))

    @test typeof(vi_dod) <: DynamicPPL.AbstractVarInfo
    @test Set(keys(vi_plain_eval)) == Set(keys(vi_dod_eval))
end
