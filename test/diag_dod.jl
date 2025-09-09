using DynamicPPL, Distributions, InteractiveUtils, BenchmarkTools

@model_dod function simple_model_dod()
    m ~ Normal(0, 1)
    x ~ Normal(m, 1)
    return (m = m, x = x)
end

m = simple_model_dod()
vi = DODVarInfo(m)
println("typeof(vi.acc_updater): ", typeof(vi.acc_updater))
println("typeof(vi.acc_default): ", typeof(vi.acc_default))
println("acc_updater===nothing: ", vi.acc_updater === nothing)
vn = @varname(m)

# Use a typed wrapper to model the evaluator-local case where `__varinfo__`
# is a `DODVarInfo` argument. Pass a real `right` distribution so accumulator
# implementations that call `logpdf` have valid inputs.
function test(vi::DODVarInfo{AU}) where {AU}
    accup = vi.acc_updater
    right = Normal(0, 1)
    return DynamicPPL.acc_updater_assume(accup, vi, 0.0, 0.0, vn, right)
end

println("@code_warntype for typed-updater call-site (vi::DODVarInfo):")
@code_warntype test(vi)

println("micro-benchmark of typed-updater call-site:")
@btime test(vi)

# create a DODVarInfo without updater to benchmark fallback paths
vi2 = DODVarInfo()
println("\nFallback path (no acc_updater) benchmark - map_accumulators path):")
# Benchmark the generic accumulator-mapping path directly.
g() = DynamicPPL.map_accumulators!!(acc -> DynamicPPL.accumulate_assume!!(acc, 0.0, 0.0, vn, Normal(0,1)), vi2)
@btime g()

println("done")
