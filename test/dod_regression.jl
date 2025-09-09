using DynamicPPL, Test, Distributions, InteractiveUtils

@model_dod function simple_model_dod_rt()
    m ~ Normal(0,1)
    x ~ Normal(m,1)
    return (m=m, x=x)
end

m = simple_model_dod_rt()
vi = DODVarInfo(m)

@test (vi.acc_updater isa DynamicPPL.AccUpdater) || (vi.acc_updater isa DynamicPPL.NoopAccUpdater)

# Ensure evaluator is specialized on parametric DODVarInfo and that the __acc_updater__ local
# is the expected Union containing the concrete AccUpdater when present.
msig = Base.unwrap_unionall(typeof(m.f)).name.wrapper
# run via the public evaluate entrypoint: convert `ModelDOD` to `Model` and
# call the existing `evaluate_and_sample!!(model::Model, ...)` path.
m_as_model = DynamicPPL.Model(m.f, m.args, m.defaults, m.context)
val, vi2 = DynamicPPL.evaluate_and_sample!!(m_as_model, vi)
@test typeof(vi2) <: DODVarInfo

# micro-benchmark-ish sanity check: call the typed updater path and ensure it runs and returns a DODVarInfo
if DynamicPPL.has_updater(vi)
    res = DynamicPPL.acc_updater_assume(vi.acc_updater, vi, 0.0, 0.0, @varname(m), Normal(0,1))
    @test res isa DODVarInfo
end

println("dod_regression: OK")
