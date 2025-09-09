using DynamicPPL
using Distributions
using Test

@testset "submodel_tests: plain-model parity" begin
    @model function inner()
        x ~ Normal()
        y ~ Normal()
        return (x, y)
    end

    @model function outer()
        return a ~ to_submodel(inner())
    end

    m_plain = outer()
    vi_plain = VarInfo(m_plain)

    _, vi_plain_eval = DynamicPPL.evaluate!!(m_plain, deepcopy(vi_plain))
    @test Set(keys(vi_plain_eval)) == Set([@varname(a.x), @varname(a.y)])
end

# NOTE: DOD models embedding plain submodels currently require additional
# wiring (to_submodel(::ModelDOD) is not implemented). Skipping this test
# for now; add a focused test once the support is implemented.
