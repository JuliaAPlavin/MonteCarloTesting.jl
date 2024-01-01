module RectiGridsExt
using Accessors
using FlexiMaps: mapview
using AxisKeys
using RectiGrids
using MonteCarloTesting
import MonteCarloTesting:
    MCSamples, nrandom, sampletype, alt,
    mapsamples, map_w_params, map_whole_realizations,
    pvalues_all, pvalue_post


const MCSamplesMulti{T} = KeyedArray{T} where {T<:MCSamples}

function nrandom(mcm::MCSamplesMulti)
    @assert allequal(nrandom.(mcm))
    nrandom(first(mcm))
end

sampletype(::Type{<:MCSamplesMulti{A}}) where {A} = sampletype(A)
sampletype(mc::MCSamplesMulti) = sampletype(eltype(mc))
Accessors.set(mc::MCSamplesMulti, ::typeof(sampletype), ::Type{T}) where {T} = @set mc |> Elements() |> sampletype = T

function alt(mcm::MCSamplesMulti)
    @assert allequal(alt.(mcm))
    alt(first(mcm))
end
Accessors.set(mcm::MCSamplesMulti, ::typeof(alt), v) = @set mcm |> Elements() |> alt = v

function mapsamples(f, mcm::MCSamplesMulti; mapfunc=map)
    return map(mcm) do mc
        mapsamples(f, mc; mapfunc)
    end
end

function map_whole_realizations(f, mcs::MCSamplesMulti)
    return MCSamples(
        real=f(mapview(realval, mcs)),
        random=map(1:nrandom(mcs)) do i
            f(mapview(ps -> randomvals(ps)[i], mcs))
        end
    )
end

"""    map_w_params(f::( (T, P) -> U ), mc::MCSamples{T}, params::RectiGrid{P} [; mapfunc=map])::MCSamplesMulti{U}

Add deterministic parameters to Monte-Carlo realizations.

Applies `f(sample, param)` to each combination of existing samples (both real and random) and deterministic parameters. Parameters already present in `mc` are also included when calling `f`.

`mapfunc` argument can be used for parallelization: eg, `mapfunc = ThreadsX.map`.
"""
function map_w_params(f, mcs::MCSamples, params; mapfunc=map)
    return map(params) do pars
        mapsamples(mcs; mapfunc) do sample
            f(sample, pars)
        end
    end
end

function map_w_params(f, mcm::MCSamplesMulti, params; mapfunc=map)
    prev_grid = grid(; named_axiskeys(mcm)...)
    return map(prev_grid, mcm) do prev_pars, mc
        map(params) do pars
            mapsamples(mc; mapfunc) do sample
                f(sample, merge(prev_pars, pars))
            end
        end
    end |> stack
end

function map_w_params(f, mcs::MCSamples; mapfunc=map)
    mc_tmp = mapsamples(mcs; mapfunc) do sample
        f(sample, (;))
    end
    axks = named_axiskeys(realval(mc_tmp))
    @assert all(A -> named_axiskeys(A) == axks, randomvals(mc_tmp))
    map(grid(;axks...)) do pars
        mapsamples(mc_tmp) do ss
            ss(;pars...)
        end
    end
end

function map_w_params(f, mcm::MCSamplesMulti; mapfunc=map)
    prev_grid = grid(; named_axiskeys(mcm)...)
    return map(prev_grid, mcm) do prev_pars, mc
        mc_tmp = mapsamples(mc; mapfunc) do sample
            f(sample, prev_pars)
        end
        axks = named_axiskeys(realval(mc_tmp))
        @assert all(A -> named_axiskeys(A) == axks, randomvals(mc_tmp))
        map(grid(;axks...)) do pars
            mapsamples(mc_tmp) do ss
                ss(;pars...)
            end
        end
    end |> stack
end

function pvalues_all(mcm::MCSamplesMulti, mode=Fraction; alt=alt(mcm))
    map(mcm) do mc
        pvalues_all(mc, mode; alt)
    end
end

"""    pvalue_post(mc::MCSamplesMulti; alt, combine=minimum)

Compute the so-called _post-trial_ p-value. That's an estimate of the probability to obtain the pre-trial p-value as low as it is in random realizations.

`alt`: specification of the alternative hypothesis, passed as-is to `pvalue()`.
`combine`: experimental.
"""
function pvalue_post(mcm::MCSamplesMulti; alt=alt(mcm), combine=minimum)
    # pvalue for each realization and parameter value:
    ps_all = pvalues_all(mcm; alt)
    # test statistics for each realization:
    test_stats = map_whole_realizations(combine, ps_all)
    return pvalue(test_stats; alt=<=)
end

end