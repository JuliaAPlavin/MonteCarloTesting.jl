using MonteCarloTesting
using Test

import Aqua
import CompatHelperLocal as CHL
@testset begin
    CHL.@check()
    Aqua.test_ambiguities(MonteCarloTesting, recursive=false)
    Aqua.test_unbound_args(MonteCarloTesting)
    Aqua.test_undefined_exports(MonteCarloTesting)
    Aqua.test_stale_deps(MonteCarloTesting)
end

# @testset begin
#     @test_throws MethodError MonteCarloSamples(real=0, random=1:5)
# 	mcits = MonteCarloSamples(real=fill(0), random=1:5)
#     @test mcits.real == fill(0)
#     @test mcits.random == 1:5
#     @test nsamples(mcits) == 5
#     @test nparams(mcits) == 1
#     samples = map(it -> it, first(mcits, 5))
#     @test samples == mcits
#     @test nsamples(first(mcits, 3)) == 3
#     @test nparams(first(mcits, 3)) == 1

#     @test nsamples(filter_params(p -> true, mcits)) == 5
#     @test nparams(filter_params(p -> true, mcits)) == 1

#     samples_par1 = map_w_params(mcits, [(;a) for a in 1:3]) do it, params
#         (; it, params...)
#     end
#     @test nsamples(samples_par1) == 5
#     @test nparams(samples_par1) == 3
#     @test samples_par1.params == [(a=1,), (a=2,), (a=3,)]
#     @test samples_par1.real == [(it=0, a=1), (it=0, a=2), (it=0, a=3)]
#     @test samples_par1.random == [(it=1, a=1) (it=2, a=1) (it=3, a=1) (it=4, a=1) (it=5, a=1); (it=1, a=2) (it=2, a=2) (it=3, a=2) (it=4, a=2) (it=5, a=2); (it=1, a=3) (it=2, a=3) (it=3, a=3) (it=4, a=3) (it=5, a=3)]

#     samples_par2 = map_w_params(mcits, [(;b, a) for b in [:x, :y], a in 1:3]) do it, params
#         (; it, params...)
#     end
#     @test nsamples(samples_par2) == 5
#     @test nparams(samples_par2) == 6
#     @test samples_par2.params == [(b = :x, a = 1) (b = :x, a = 2) (b = :x, a = 3); (b = :y, a = 1) (b = :y, a = 2) (b = :y, a = 3)]
#     @test samples_par2.real == [(it = 0, b = :x, a = 1) (it = 0, b = :x, a = 2) (it = 0, b = :x, a = 3); (it = 0, b = :y, a = 1) (it = 0, b = :y, a = 2) (it = 0, b = :y, a = 3)]
#     @test size(samples_par2.random) == (2, 3, 5)

#     @test nsamples(first(samples_par2, 3)) == 3
#     @test nparams(first(samples_par2, 3)) == 6
#     @test size(first(samples_par2, 3).random) == (2, 3, 3)

#     @test nsamples(filter_params(samples_par2; a=(==)(1))) == 5
#     @test nparams(filter_params(samples_par2; a=(==)(1))) == 2
#     @test filter_params(p -> p.a == 1, samples_par2) == filter_params(samples_par2; a=(==)(1))

#     samples_par3 = map_w_params(samples_par1, [(; b) for b in [:x, :y]]) do it, params
#         (; it.it, params.b, it.a)
#     end
#     @test nsamples(samples_par3) == 5
#     @test nparams(samples_par3) == 6
#     @test samples_par3.params == samples_par2.params
#     @test samples_par3.real == samples_par2.real
#     @test samples_par3.random == samples_par2.random

#     samples_par4 = map_w_batch_params(mcits, [(;b, a) for b in [:x, :y], a in 1:3]) do it, paramss
#         [(; it, params...) for params in paramss]
#     end
#     @test nsamples(samples_par4) == 5
#     @test nparams(samples_par4) == 6
#     @test samples_par4.params == samples_par2.params
#     @test samples_par4.real == samples_par2.real
#     @test samples_par4.random == samples_par2.random
# end

# @testset begin
# 	mcits = MonteCarloSamples(real=[0], random=1:1000)
# 	samples = map(mcits) do it
#         rand()
# 	end
#     @test nsamples(samples) == 1000
#     @test nparams(samples) == 1
#     pvals = pvalues_individual(samples)
#     @test size(pvals) == ()
#     @test 0 < only(pvals).pvalue < 1
# end
