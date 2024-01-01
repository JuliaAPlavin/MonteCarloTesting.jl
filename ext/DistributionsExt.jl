module DistributionsExt
using MonteCarloTesting
using Distributions: fit, cdf, ccdf, quantile, Poisson, Normal

function MonteCarloTesting.pvalue(mc::MonteCarloTesting.MCSamples, mode::Type{Poisson}; alt)
    @assert sampletype(mc) <: Real
    dist = fit(Poisson, randomvals(mc))
    return alt == (>=) ? ccdf(dist, realval(mc) - 1) :
           alt == (<=) ?  cdf(dist, realval(mc)) :
           @assert false
end

# signatures must be more specific than in MonteCarloTesting itself
# otherwise - method overwritten warnings
MonteCarloTesting.nσ(p::PValue) = quantile(Normal(0, 1), 1-p.p/2)
MonteCarloTesting._nσ_str(p::PValue) = "$(round(MonteCarloTesting.nσ(p), digits=1))σ"

end
