using DynamicPPL, Distributions, BenchmarkTools, Random, Profile

@model_dod function simple_model_dod()
    m ~ Normal(0, 1)
    x ~ Normal(m, 1)
    return (m = m, x = x)
end

function make_dod()
    vi = DODVarInfo()
    add_variable!(vi, @varname(m), 0.5, dist = Normal(0,1))
    add_variable!(vi, @varname(x), 0.0, dist = Normal(0.5,1))
    return vi
end

vi = make_dod()
meta_m = vi.idcs[@varname(m)]
meta_x = vi.idcs[@varname(x)]

println("Warmup: one call")
logprior(simple_model_dod(), vi)

println("Benchmarking tilde_assume_dod!! (sampler-aware) with BenchmarkTools")
@btime tilde_assume_dod!!(Random.default_rng(), SampleFromPrior(), Normal(0,1), $meta_m, $vi)

println("Micro-loop profile (Profile.@profile) - 1e5 iterations")
Profile.clear()
rng = Random.MersenneTwister(1234)
Profile.@profile for i in 1:100000
    tilde_assume_dod!!(rng, SampleFromPrior(), Normal(0,1), meta_m, vi)
    tilde_observe_dod!!(nothing, Normal(0.5,1), 0.0, meta_x, vi)
end

println("Profile samples collected. Use `Profile.print()` to inspect.")
Profile.print()

println("Done")
