using DynamicPPL, Distributions, BenchmarkTools, Profile, Random, Logging

# small model used in previous benchmarks
@model function simple_model()
    m ~ Normal(0, 1)
    x ~ Normal(m, 1)
end
m = simple_model()

@model_dod function simple_model_dod()
    m ~ Normal(0, 1)
    x ~ Normal(m, 1)
end
mdod = simple_model_dod()

# deterministic RNG for reproducible profiles
Random.seed!(1234)

println("Warmup: one call each")
sample(m, SampleFromPrior(), 10; progress=false)
sample(mdod, SampleFromPrior(), 10; progress=false)

println("Benchmarking sample(m, SampleFromPrior(), 1000) - baseline")
@btime sample($m, SampleFromPrior(), 1000; progress=false)

println("Benchmarking sample(mdod, SampleFromPrior(), 1000) - DOD")
@btime sample($mdod, SampleFromPrior(), 1000; progress=false)

# Profile each repeatedly to gather samples
nreps = 20
println("Profiling baseline sample, $nreps repetitions")
Profile.clear()
with_logger(Logging.NullLogger()) do
    Profile.@profile begin
        for i in 1:nreps
            sample(m, SampleFromPrior(), 1000; progress=false)
        end
    end
end
println("Profile summary: baseline")
Profile.print()

println("Profiling DOD sample, $nreps repetitions")
Profile.clear()
with_logger(Logging.NullLogger()) do
    Profile.@profile begin
        for i in 1:nreps
            sample(mdod, SampleFromPrior(), 1000; progress=false)
        end
    end
end
println("Profile summary: DOD")
Profile.print()
