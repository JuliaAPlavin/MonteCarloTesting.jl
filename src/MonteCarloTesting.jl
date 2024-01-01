module MonteCarloTesting

using FlexiMaps: mapview
using StatsBase: competerank
using IntervalSets
using Accessors
using ConstructionBaseExtras  # for IntervalSets
import Random
using Printf

export
    montecarlo,
    realval, randomvals, realrandomvals, nrandom, sampletype,
    alt,
    PValue,
    Fraction, pvalue, pvalues_all, pvalue_post,
    pvalue_tiesinterval, pvalue_mcinterval,
    mapsamples, map_w_params,
    swap_realval

struct PValue
    p::Float64
end

# signatures must be less specific than in DistributionsExt
# otherwise - method overwritten warnings
nσ(p) = error("Load Distributions")
_nσ_str(p) = "XXσ"

function Base.show(io::IO, p::PValue)
    # print(io, lpad(round(100*p.p, sigdigits=2), 5), "%")
    s = @sprintf "%.1e" p.p
    print(io, replace(s, r"(?<=[+-])0" => ""))
    print(io, " (", _nσ_str(p), ")")
end

struct _NoAlt end
(::_NoAlt)(args...) = error("No alternative hypothesis specified. Use `@set alt(mc) = ...`, or provide the `alt` argument to p-value function.")

""" Stores the real/actual/true value together with its Monte-Carlo realizations. """
struct MCSamples{T, AT <: AbstractArray{T}, ALT}
    real::T
    random::AT
    alt::ALT
end
MCSamples{T, AT}(real, random) where {T, AT}= MCSamples{T, AT, _NoAlt}(real, random, _NoAlt())
MCSamples{T}(real, random) where {T} = MCSamples{T, typeof(random)}(real, random)
MCSamples{T}(; real, random) where {T} = MCSamples{T}(real, random)
MCSamples(; real, random) = MCSamples(real, random, _NoAlt())

@accessor alt(mc::MCSamples) = mc.alt

"    realval(mc::MCSamples{T})::T
The real value of `MCSamples`. "
@accessor realval(mc::MCSamples) = mc.real

"    nrandom(mc::MCSamples)::Int
Number of random realizations within an `MCSamples` object, or within each of the same-sized `MCSamples` in a container. "
nrandom(mc::MCSamples) = length(randomvals(mc))

"    randomvals(mc::MCSamples{T})::AbstractVector{T}
Array of random realizations in `MCSamples`. Can be eager or lazily computed. "
@accessor randomvals(mc::MCSamples) = mc.random

"    realrandomvals(mc::MCSamples{T})::AbstractVector{T}
Array containing both the real value and random realizations from `MCSamples`. "
realrandomvals(mc::MCSamples) = [[realval(mc)]; randomvals(mc)]

Base.:(==)(a::MCSamples, b::MCSamples) = a.real == b.real && a.random == b.random

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
        MCSamples{typeof(real)}(; real, random)
    else
        check_randomfunc(randomfunc, copy(rng))
        rngs = map(seed -> Random.seed!(copy(rng), seed), rand(rng, UInt, nrandom))
        MCSamples{typeof(real)}(; real, random=mapview(rng -> randomfunc(copy(rng)), rngs))
    end
end

function check_randomfunc(randomfunc, rng)
    crng = copy(rng)
    @assert crng == rng
    x = randomfunc(crng)
    @assert crng != rng  "Provided `randomfunc(rng)` doesn't use its `rng` argument. This can't be right!"
    xx = try
        deepcopy(x)
    catch e
        @warn "Cannot check whether `randomfunc(rng)` returns the same result given the same `rng`"  e
        return
    end
    if x == xx
        y = randomfunc(rng)
        @assert x == y  "Provided `randomfunc(rng)` returns different values when called with the same `rng`."
    else
        @warn "Cannot check whether `randomfunc(rng)` returns the same result given the same `rng`: its return value `x != deepcopy(x)`."
    end
end



sampletype(::Type{<:MCSamples{T}}) where {T} = T
sampletype(mc::MCSamples) = sampletype(typeof(mc))
Accessors.set(mc::MCSamples, ::typeof(sampletype), ::Type{T}) where {T} = MCSamples{T}(convert(T, mc.real), convert.(T, mc.random))


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

pvalue(::Type{PValue}, args...; kwargs...) = PValue(pvalue(args...; kwargs...))

function pvalue(mc::MCSamples, mode::Type{Fraction}=Fraction; alt=alt(mc))
    @assert sampletype(mc) <: Real
    @assert alt(realval(mc), realval(mc))
    nalt = count(r -> alt(r, realval(mc)), randomvals(mc))
    n = nrandom(mc)
    return (nalt + 1) / (n + 1)
end

function pvalue_mcinterval(mc::MCSamples; alt=alt(mc), nσ=2)
    @assert sampletype(mc) <: Real
    @assert alt(realval(mc), realval(mc))
    nalt = count(r -> alt(r, realval(mc)), randomvals(mc))
    n = nrandom(mc)
    return ci_wilson((; x=nalt, n=n); nσ)
end

function pvalue_tiesinterval(mc::MCSamples; alt=alt(mc))
    @assert sampletype(mc) <: Real
    @assert alt(realval(mc), realval(mc))
    nalt = count(r -> alt(r, realval(mc)), randomvals(mc))
    nalt_lo = count(r -> alt(r, realval(mc)) && !alt(realval(mc), r), randomvals(mc))
    n = nrandom(mc)
    nalt_int = nalt_lo..nalt
    @modify(x -> (x + 1) / (n + 1), nalt_int |> Properties())
end

# adapted from https://github.com/JuliaStats/HypothesisTests.jl/blob/master/src/binomial.jl
# Wilson score interval
function ci_wilson(x; nσ)
    q = nσ # quantile(Normal(), 1-alpha/2)
    p = x.x / x.n
    denominator = 1 + q^2/x.n
    μ = p + q^2/(2*x.n)
    μ /= denominator
    σ = sqrt(p*(1-p)/x.n + q^2/(4x.n^2))
    σ /= denominator
    μ ± q*σ
end

"""    pvalues_all(mc::Union{MCSamples, MCSamplesMulti}, [mode::Type=Fraction]; alt)::MCSamples{Real}

Estimate the p-value for the real value, and "p-values" for all random realizations.
The latter correspond to swapping each realization with the real value and computing the regular p-value.

For MCSamplesMulti, maps over all parameters.

See also `pvalue()` docs.
"""
function pvalues_all(mc::MCSamples, mode::Type{Fraction}=Fraction; alt=alt(mc))
    @assert sampletype(mc) <: Real
    @assert alt(realval(mc), realval(mc))
    ranks = competerank(realrandomvals(mc); lt=!alt)
    nps = length(ranks) + 1 .- ranks
    pvals = nps ./ length(ranks)
    return MCSamples(real=pvals[1], random=pvals[2:end])
end

# defined only in RectiGridsExt:
function map_w_params end
function map_whole_realizations end
function pvalue_post end

end
