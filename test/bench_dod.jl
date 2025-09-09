using BenchmarkTools, DynamicPPL, Distributions

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

function make_setups()
    vi_simple = SimpleVarInfo((m = 0.5, x = 0.0))
    vi_dod = DODVarInfo()
    add_variable!(vi_dod, @varname(m), 0.5, dist = Normal(0, 1))
    add_variable!(vi_dod, @varname(x), 0.0, dist = Normal(0.5, 1))
    return vi_simple, vi_dod
end

vi_simple, vi_dod = make_setups()

println("Benchmark: logprior(@model, SimpleVarInfo)")
@btime logprior(simple_model(), vi_simple)

println("Benchmark: loglikelihood(@model, SimpleVarInfo)")
@btime loglikelihood(simple_model(), vi_simple)

println("Benchmark: logprior(@model_dod, DODVarInfo)")
@btime logprior(simple_model_dod(), vi_dod)

println("Benchmark: loglikelihood(@model_dod, DODVarInfo)")
@btime loglikelihood(simple_model_dod(), vi_dod)

# Benchmark sampling via AbstractMCMC.sample using SampleFromPrior
const SAMPLE_N = 1000
println("Benchmark: sample(@model, SampleFromPrior(), $SAMPLE_N)")
@btime sample(simple_model(), SampleFromPrior(), SAMPLE_N; progress=false)

println("Benchmark: sample(@model_dod, SampleFromPrior(), $SAMPLE_N)")
@btime sample(simple_model_dod(), SampleFromPrior(), SAMPLE_N; progress=false)
