using MonteCarloTesting
using IntervalSets
using Accessors
using Statistics: mean
using StableRNGs
using RectiGrids
using AxisKeys
using Distributions: Poisson
using Test

@testset "basic hardcoded" begin
    mc = @inferred montecarlo(real=0, random=[-1, 0, 0, 0, 0, 0, 0, 1, 1])
    @test sampletype(mc) == Int
    @test realval(mc) == 0
    @test nrandom(mc) == 9
    @test randomvals(mc) == [-1, 0, 0, 0, 0, 0, 0, 1, 1]
    @test realrandomvals(mc) == [0, -1, 0, 0, 0, 0, 0, 0, 1, 1]

    @test realval(swap_realval(mc, 1)) == -1
    @test randomvals(swap_realval(mc, 1)) == [0, 0, 0, 0, 0, 0, 0, 1, 1]

    @test pvalue(mc; alt= >=) == 0.9
    @test pvalue(mc; alt= <=) == 0.8
    @test pvalue_mcinterval(mc; alt= >=) == 0.5577894550965302..0.9806720833650083

    Accessors.test_getset_laws(sampletype, mc, Float64, Int)
    Accessors.test_getset_laws(realval, mc, 5, 2)
    Accessors.test_getset_laws(first ∘ randomvals, mc, 1, 2)

    let mcf = @set sampletype(mc) = Float64

        @test pvalue(@set(realval(mcf) = -0.1); alt= >=) == 0.9
        @test pvalue(@set(realval(mcf) = 0.1); alt= >=) == 0.3
        @test pvalue(@set(realval(mcf) = -0.1); alt= <=) == 0.2
        @test pvalue(@set(realval(mcf) = 0.1); alt= <=) == 0.8

        @test pvalue_tiesinterval(mc; alt = >=) == 0.3..0.9
        @test pvalue_tiesinterval(mc; alt = <=) == 0.2..0.8

        @test pvalue_tiesinterval(@set(realval(mcf) = -0.1); alt = >=) == 0.9..0.9
        @test pvalue_tiesinterval(@set(realval(mcf) = 0.1); alt = >=) == 0.3..0.3
    end

    @test realval(pvalues_all(mc; alt= >=)) == 0.9
    @test realval(pvalues_all(mc; alt= <=)) == 0.8
    @test randomvals(pvalues_all(mc; alt= >=)) == [1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.2]

    mc1 = mapsamples(x -> x + 1, mc)
    @test realrandomvals(mc1) == [1, 0, 1, 1, 1, 1, 1, 1, 2, 2]
    @test pvalues_all(mc1; alt= >=) == pvalues_all(mc; alt= >=)

    mc2 = map_w_params(mc, grid(a=-1:2)) do x, p
        x + p.a
    end
    @test sampletype(mc2) == Int
    @test realval.(mc2) == [-1, 0, 1, 2]
    @test realval.(mc2)(a=1) == 1
    @test realval.(mc2(a=0:2)) == [0, 1, 2]
    @test realval(mc2(a=1)) == 1

    @test_throws TypeError montecarlo(real=1, random=Any[2, 3, 4])
end

@testset "PValue" begin
    @test sprint(show, PValue(0.1234)) == "1.2e-1 (1.5σ)"
    @test sprint(show, PValue(1.234e-5)) == "1.2e-5 (4.4σ)"

    mc = montecarlo(real=0, random=[-1, 0, 0, 0, 0, 0, 0, 1, 1])
    @test PValue(pvalue(mc; alt= >=)) == pvalue(PValue, mc; alt= >=)
end

@testset "randomization" begin
    mc1 = @inferred montecarlo(
        real=0.,
        randomfunc=rng -> rand(rng),
        nrandom=1000,
        rng=StableRNG(123),
    )
    @test sampletype(mc1) === Float64
    @test collect(randomvals(mc1)) == collect(randomvals(mc1))  # repeatable
    @test allunique(randomvals(mc1))
    @test sum(randomvals(mc1)) ≈ 489.158  rtol=1e-4

    mc2 = montecarlo(
        real=0.,
        randomfunc=rng -> rand(rng),
        nrandom=1000,
        # don't pass rng
    )
    @test collect(randomvals(mc2)) == collect(randomvals(mc2))  # repeatable
    @test allunique(randomvals(mc2))
    @test sum(randomvals(mc2)) != sum(randomvals(mc1))

    mc3 = montecarlo(
        real=0.,
        randomfunc=rng -> rand(rng),
        nrandom=1000,
        # don't pass rng
    )
    @test sum(randomvals(mc3)) != sum(randomvals(mc2))

    @test_throws AssertionError montecarlo(real=0., randomfunc=_ -> rand(), nrandom=100)
    @test_throws AssertionError montecarlo(real=0., randomfunc=rng -> (rand(rng); rand()), nrandom=100)
end

@testset "common usage" begin
    mc1 = @inferred montecarlo(
        real=range(0.5, 0.6, length=100) |> collect,
        randomfunc=rng -> rand(rng, 100),
        nrandom=1000,
        rng=StableRNG(123),
    )
    @test sampletype(mc1) === Vector{Float64}
    @test nrandom(mc1) == 1000
    @test collect(randomvals(mc1)) == collect(randomvals(mc1))  # repeatable

    mc2 = mapsamples(mc1) do xs
        mean(xs)
    end

    @test pvalue(mc2, alt= >=) ≈ 0.039960  rtol=1e-4
    @test pvalue(mc2, alt= <=) ≈ 0.961039  rtol=1e-4
    @test pvalue(mc2, alt= >=) + pvalue(mc2, alt= <=) > 1
    @test pvalue(mc2, alt= >=) == pvalue(mc2, Fraction, alt= >=)
    @test pvalue(swap_realval(mc2, 4), alt= >=) ≈ 0.513486  rtol=1e-4
    @test pvalue(swap_realval(mc2, 4), alt= >=) == randomvals(pvalues_all(mc2; alt= >=))[4]

    for alt in [>=, <=]
        int = pvalue_mcinterval(mc2; alt)
        p = pvalue(mc2; alt)
        @test p ∈ int
        @test p ≈ mean(int)  rtol=0.1
        @test width(int) ≈ 0.025  rtol=0.1
    end

    mc3 = map_w_params(mc1, grid(n=10:100)) do xs, ps
        mean(xs[1:ps.n])
    end
    @test realval.(mc3)(n=10) ≈ 0.504545  rtol=1e-4
    @test realval.(mc3)(n=95) ≈ 0.547474  rtol=1e-4
    @test pvalue.(mc3, alt= >=)(n=10) ≈ 0.456543  rtol=1e-4
    @test pvalue.(mc3, alt= >=)(n=95) ≈ 0.048951  rtol=1e-4
    @test realval(mc3(n=10)) == realval.(mc3)(n=10)
    @test mc3(n=10:20)[1] == mc3(n=10)

    mc4 = mapsamples(x -> x^2, mc3)
    @test realval.(mc4)(n=10) ≈ 0.504545^2  rtol=1e-4
    @test pvalue.(mc4, alt= >=) == pvalue.(mc3, alt= >=)

    mc5 = map_w_params(mc3, grid(p=2:4, mul=0.1:0.1:2)) do x, ps
        @assert 10 <= ps.n <= 100
        ps.mul * x^ps.p
    end
    @test nrandom(mc5) == 1000
    @test minimum(pvalue.(mc5; alt= >=)) ≈ 0.039960  rtol=1e-4
    @test pvalue_post(mc5; alt= >=) ≈ 0.206793  rtol=1e-4
    @test minimum(pvalue.(mc5(n= >=(90)); alt= >=)) ≈ 0.039960  rtol=1e-4
    @test pvalue_post(mc5(n= >=(90)); alt= >=) ≈ 0.060939  rtol=1e-4
end

@testset "different pvalues" begin
    mcp = montecarlo(real=10, random=rand(StableRNG(123), Poisson(4), 1000))
    @test pvalue(mcp, alt= >=) ≈ 0.008991  rtol=1e-4
    @test pvalue(mcp, alt= <=) ≈ 0.997002  rtol=1e-4
    @test pvalue(mcp, Poisson, alt= >=) ≈ 0.00874459  rtol=1e-4
    @test pvalue(mcp, Poisson, alt= <=) ≈ 0.996913  rtol=1e-4
end


import Aqua
import CompatHelperLocal as CHL
@testset begin
    CHL.@check()
    Aqua.test_ambiguities(MonteCarloTesting, recursive=false)
    Aqua.test_unbound_args(MonteCarloTesting)
    Aqua.test_undefined_exports(MonteCarloTesting)
    Aqua.test_stale_deps(MonteCarloTesting)
end
