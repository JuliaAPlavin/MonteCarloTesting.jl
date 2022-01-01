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
