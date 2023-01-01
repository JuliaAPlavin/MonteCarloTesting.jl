module MonteCarloTesting


using RectiGrids
using Statistics: mean
using Distributions: fit, cdf, ccdf, Poisson
using Parameters
using SplitApplyCombine
using StatsBase: competerank
import AxisKeys
using LazyStack: stack  # AxisKeys supports stacking with LazyStack, and not with SplitApplyCombine
import Random

export montecarlo, realval, randomvals, realrandomvals, nrandom, sampletype, Poisson, Fraction, pvalue, pvalues_all, pvalue_wtrials, mapsamples, map_w_params, swap_realval


""" Stores the real/actual/true value together with its Monte-Carlo realizations. """
Base.@kwdef struct MCSamples{T, AT <: AbstractArray{T}}
	real::T
	random::AT
end

"    realval(mc::MCSamples{T})::T
The real value of `MCSamples`. "
realval(mc::MCSamples) = mc.real

"    nrandom(mc::MCSamples)::Int
Number of random realizations within an `MCSamples` object, or within each of the same-sized `MCSamples` in a container. "
nrandom(mc::MCSamples) = length(randomvals(mc))

"    randomvals(mc::MCSamples{T})::AbstractVector{T}
Array of random realizations in `MCSamples`. Can be eager or lazily computed. "
randomvals(mc::MCSamples) = mc.random

"    realrandomvals(mc::MCSamples{T})::AbstractVector{T}
Array containing both the real value and random realizations from `MCSamples`. "
realrandomvals(mc::MCSamples) = [[realval(mc)]; randomvals(mc)]

"""
    montecarlo(; real::T, random::Vector{T})
    montecarlo(; real::T, randomfunc::(RNG -> T), nrandom::Int[, rng::AbstractRNG])

Create an `MCSamples` struct containing the real value and its random Monte-Carlo realizations.

Realizations can be specified as a `random::Vector`, or generated lazily on-demand using `randomfunc` function.
Reproduceability in the latter case is ensured by creating an array of copied and seeded `rng`s.
"""
function montecarlo(; real, random=nothing, randomfunc=nothing, nrandom=nothing, rng=Random.default_rng())
	@assert !isnothing(random) != (!isnothing(randomfunc) && !isnothing(nrandom))
	if !isnothing(random)
		MCSamples(; real, random)
	else
		rngs = map(seed -> Random.seed!(copy(rng), seed), 1:nrandom)
		MCSamples(; real, random=mapview(rng -> randomfunc(copy(rng)), rngs))
	end
end


@with_kw struct MCSamplesMulti{A <: KeyedArray{<:MCSamples}}
    arr::A
    
    @assert length(unique(nrandom.(arr))) == 1
end

nrandom(mcm::MCSamplesMulti) = nrandom(first(mcm.arr))

Base.broadcastable(mcm::MCSamplesMulti) = mcm.arr
Base.size(mcm::MCSamplesMulti) = size(mcm.arr)
Base.getindex(mcm::MCSamplesMulti, I::Int...) = mcm.arr[I...]
Base.getindex(mcm::MCSamplesMulti, I...) = MCSamplesMulti(mcm.arr[I...])
(mcm::MCSamplesMulti)(args...; kwargs...) = MCSamplesMulti(mcm.arr(args...; kwargs...))
AxisKeys.axiskeys(mcm::MCSamplesMulti) = axiskeys(mcm.arr)
AxisKeys.named_axiskeys(mcm::MCSamplesMulti) = named_axiskeys(mcm.arr)

sampletype(::Type{<:MCSamples{T}}) where {T} = T
sampletype(::Type{<:MCSamplesMulti{A}}) where {A} = sampletype(eltype(A))
sampletype(mc::Union{<:MCSamples, <:MCSamplesMulti}) = sampletype(typeof(mc))
Base.eltype(mc::MCSamples) = sampletype(mc)


"""    mapsamples(f, mc::Union{MCSamples,MCSamplesMulti} [; mapfunc=map])

Transform `mc` by applying `f` to each sample, both real value and random realizations.

`mapfunc` argument can be used for parallelization: eg, `mapfunc = ThreadsX.map`.
"""
function mapsamples(f, mcs::MCSamples; mapfunc=map)
	return MCSamples(;
		real=f(realval(mcs)),
		random=mapfunc(f, randomvals(mcs)),
	)
end

function mapsamples(f, mcm::MCSamplesMulti; mapfunc=map)
	return MCSamplesMulti(
		map(mcm.arr) do mc
			mapsamples(f, mc; mapfunc)
		end
	)
end

"""    map_w_params(f::( (T, P) -> U ), mc::MCSamples{T}, params::RectiGrid{P} [; mapfunc=map])::MCSamplesMulti{U}

Add deterministic parameters to Monte-Carlo realizations.

Applies `f(sample, param)` to each combination of existing samples (both real and random) and deterministic parameters. Parameters already present in `mc` are also included when calling `f`.

`mapfunc` argument can be used for parallelization: eg, `mapfunc = ThreadsX.map`.
"""
function map_w_params(f, mcs::MCSamples, params; mapfunc=map)
	return MCSamplesMulti(
		map(params) do pars
			mapsamples(mcs; mapfunc) do sample
				f(sample, pars)
			end
		end
	)
end

function map_w_params(f, mcm::MCSamplesMulti, params; mapfunc=map)
	prev_grid = grid(; named_axiskeys(mcm.arr)...)
	return MCSamplesMulti(
		map(prev_grid, mcm.arr) do prev_pars, mc
			map(params) do pars
				mapsamples(mc; mapfunc) do sample
					f(sample, merge(prev_pars, pars))
				end
			end
		end |> stack
	)
end

"""    swap_realval(mc, randix::Int)

Swap the real value and the random realization at index `randix`.
Keeps the original `MCSamples` object intact.

Can be useful for sanity checks: ensure that the analysis doesn't yield significant results for a random realization.
"""
function swap_realval(mc::MCSamples, randix::Int)
	MCSamples(;
		real=mc.random[randix],
		random=[mc.random[begin:randix-1]; [mc.real]; mc.random[randix+1:end]],
	)
end

struct Fraction end

"""
    pvalue(mc::MCSamples, [mode::Type=Fraction]; alt)
    pvalue(mc::MCSamples, mode::Type{Distribution}; alt)

Estimate p-value, the probability of a random realization to be above/below the real value.
The direction of the alternative hypothesis is speicifed by the `alt` parameter:
- `alt= <=` -- probability of `random <= real`,
- `alt= >=` -- probability of `random >= real`.

`mode = Fraction`: estimate as a simple fraction, `p = (#{alt(random, real)} + 1) / (#random + 1)`.

`mode = Poisson`: fit the Poisson distribution to random realizations and compute `p = cdf(real)` or `p = ccdf(real - 1)` depending on `alt`.
"""
function pvalue end

function pvalue(mc::MCSamples, mode::Type{Fraction}=Fraction; alt)
    @assert eltype(mc) <: Real
    @assert alt(realval(mc), realval(mc))
    np = sum(alt.(randomvals(mc), realval(mc)))
    n = nrandom(mc)
    return (np + 1) / (n + 1)
end

function pvalue(mc::MCSamples, mode::Type{Poisson}; alt)
    @assert eltype(mc) <: Real
    dist = fit(Poisson, randomvals(mc))
    return alt == (>=) ? ccdf(dist, realval(mc) - 1) :
           alt == (<=) ?  cdf(dist, realval(mc)) :
           @assert false
end

"""    pvalues_all(mc::MCSamples, [mode::Type=Fraction]; alt)::MCSamples{Real}

Estimate the p-value for the real value, and "p-values" for all random realizations. The latter correspond to swapping each realization with the real value and computing the regular p-value. Time: `O(n log(n))` where `n = nrandom(mc)`.

See `pvalue()` docs for more details.
"""
function pvalues_all(mc::MCSamples, mode::Type{Fraction}=Fraction; alt)
	@assert eltype(mc) <: Real
	@assert alt(realval(mc), realval(mc))
	ranks = competerank(realrandomvals(mc); lt=!alt)
	nps = length(ranks) + 1 .- ranks
	pvals = nps ./ length(ranks)
	return MCSamples(real=pvals[1], random=pvals[2:end])
end
"""    pvalue_wtrials(mc::MCSamplesMulti; alt)

Compute so-called _pre-trial_ and _post-trial_ p-values.
- Pre-trial: minimum of individual p-values for each combination of deterministic parameters. Affected by the multiple comparisons issue.
- Post-trial: estimate of the probability to obtan the pre-trial p-value as low as it is in random realizations.

`alt`: specification of the alternative hypothesis, passed as-is to `pvalue()`.
"""
function pvalue_wtrials(mcm::MCSamplesMulti; alt)
	# pvalue for each realization and parameter value:
	ps_all = map(mcm.arr) do mc
		pvalues_all(mc; alt)
	end

	# pvalue for each realization:
	pretrials = MCSamples(
		real=minimum(ps -> realval(ps), ps_all),
		random=map(1:nrandom(mcm)) do i
			minimum(ps -> randomvals(ps)[i], ps_all)
		end
	)

	pretrial = realval(pretrials)  # actual pretrial pvalue
	posttrial = pvalue(pretrials; alt=<=)

	(; pretrial, posttrial)
end

end
