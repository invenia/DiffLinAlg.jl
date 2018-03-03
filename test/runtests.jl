using DiffLinearAlgebra, Compat.Test

import FDM: assert_approx_equal, central_fdm
import DiffLinearAlgebra: AA, AM, AVM, AS, ASVM, Arg1, Arg2
using Compat.Random

@testset "DiffLinearAlgebra" begin
    include("test_util.jl")
    include("util.jl")
    # include("generic.jl")
    include("blas.jl")
    # include("diagonal.jl")
    # include("triangular.jl")
    # include("uniformscaling.jl")
    # include("factorization/cholesky.jl")
    include("test_imports.jl")
end
