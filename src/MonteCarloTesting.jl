module MonteCarloTesting

export MonteCarloSamples, map_w_params, map_w_batch_params, pvalues_individual, PValueModes, nsamples, nparams, filter_params

using EllipsisNotation
using SplitApplyCombine
using Statistics
using Distributions


Base.@kwdef struct MonteCarloSamples{T, AT1 <: AbstractArray{T}, AT2 <: AbstractArray{T}, P, AP <: AbstractArray{P}}
    params::AP = fill((;), ())
    real::AT1
    random::AT2
end

Base.:(==)(mcs_a::MonteCarloSamples, mcs_b::MonteCarloSamples) = mcs_a.params == mcs_b.params && mcs_a.real == mcs_b.real && mcs_a.random == mcs_b.random

nsamples(mcs::MonteCarloSamples) = size(mcs.random)[end]
nparams(mcs::MonteCarloSamples) = length(mcs.params)
sizeparams(mcs::MonteCarloSamples) = size(mcs.params)
Base.first(mcs::MonteCarloSamples, n::Integer) = MonteCarloSamples(; mcs.params, mcs.real, random=mcs.random[.., 1:n])


function Base.map(f, mcs::MonteCarloSamples; mapfunc=map)
    return MonteCarloSamples(;
        mcs.params,
        real=mapfunc(f, mcs.real),
        random=mapfunc(f, mcs.random),
    )
end

function map_w_params(f, mcs::MonteCarloSamples, params...; mapfunc=map)
    new_params_merged = map(pars -> merge(pars...), Iterators.product(params...))
    return MonteCarloSamples(;
        params=map(pars -> merge(pars...), Iterators.product(params..., mcs.params)),
        real=mapfunc(((pars, sample),) -> f(sample, pars), Iterators.product(new_params_merged, mcs.real)),
        random=mapfunc(((pars, sample),) -> f(sample, pars), Iterators.product(new_params_merged, mcs.random)),
    )
end

function map_w_batch_params(f, mcs::MonteCarloSamples, params...; mapfunc=map)
    new_params_merged = map(pars -> merge(pars...), Iterators.product(params...))
    return MonteCarloSamples(;
        # why map(identity, ...) ?
        params=map(pars -> merge(pars...), Iterators.product(params..., mcs.params)),
        real=map(identity, mapfunc(sample -> f(sample, new_params_merged), mcs.real)) |> combinedims,
        random=map(identity, mapfunc(sample -> f(sample, new_params_merged), mcs.random)) |> combinedims,
    )
end

function filter_params(f::Function, mcs::MonteCarloSamples)
    ix = findall(f, mcs.params)
    return MonteCarloSamples(; params=mcs.params[ix], real=mcs.real[ix], random=mcs.random[ix, :])
end

function filter_params(mcs::MonteCarloSamples; specs...)
    return filter_params(mcs) do p
        all(v(getfield(p, k)) for (k, v) in pairs(specs))
    end
end


module PValueModes
abstract type PValueMode end
struct Fraction <: PValueMode end
struct Poisson <: PValueMode end
end

function pvalues_individual(mcs::MonteCarloSamples, mode::PValueModes.Fraction=PValueModes.Fraction(); mapfunc=map, cmp_expected=(<))
    mapfunc(pairs(mcs.params) |> collect) do (ix, pars)
        real = mcs.real[ix, 1]
        random = @view mcs.random[ix, :]
        (; pars, pvalue=(sum(r -> !cmp_expected(r, real), random) + 1) / (length(random) + 1))
    end
end

function pvalues_individual(mcs::MonteCarloSamples, mode::PValueModes.Poisson; mapfunc=map)
    mapfunc(pairs(mcs.params) |> collect) do (ix, pars)
        real = mcs.real[ix, 1]
        random_avg = mean(@view(mcs.random[ix, :]))
        (; pars, pvalue=1 - cdf(Poisson(random_avg), real - 1))
    end
end

end
