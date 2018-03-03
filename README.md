# DiffLinearAlgebra
Note: the current version of this package is not intended for general consumption.

[![Build Status](https://travis-ci.org/invenia/DiffLinearAlgebra.jl.svg?branch=master)](https://travis-ci.org/invenia/DiffLinearAlgebra.jl) [![Windows Build status](https://ci.appveyor.com/api/projects/status/g0gun5dxbkt631am/branch/master?svg=true)](https://ci.appveyor.com/project/invenia/difflinearalgebra-jl/branch/master) [![codecov.io](http://codecov.io/github/invenia/DiffLinearAlgebra.jl/coverage.svg?branch=master)](http://codecov.io/github/invenia/DiffLinearAlgebra.jl?branch=master) [![DiffLinearAlgebra](http://pkg.julialang.org/badges/DiffLinearAlgebra_0.6.svg)](http://pkg.julialang.org/detail/DiffLinearAlgebra) [![DiffLinearAlgebra](http://pkg.julialang.org/badges/DiffLinearAlgebra_0.7.svg)](http://pkg.julialang.org/detail/DiffLinearAlgebra)

DiffLinearAlgebra can be (_very_ loosely) thought of as [DiffRules.jl](https://github.com/JuliaDiff/DiffRules.jl) for linear algebra. For every sensitivity, we provide a function which, when provided with the input and output from the forward pass and the reverse-mode sensitvity w.r.t the output from the forward pass, computes the sensitivity of the specified argument.

```julia
  A, B = randn(5, 3), randn(3, 4)
  C, C̄ = A * B, randn(5, 4)
  Ā = ∇(*, Val{1}, (), C, C̄, A, B)
  B̄ = ∇(*, Val{2}, (), C, C̄, A, B)
```

In the above example, the sensitivities of `A` and `B` are computed from `C` and a random seeding of `C̄`. (Note that the third argument is currently redundant; see [this issue](https://github.com/invenia/DiffLinearAlgebra.jl/issues/1) for motivation for its inclusion.)

We also expose some "metadata" for each implemented sensitivity. This is done via a set called `ops` contains [DiffOp structs](https://github.com/invenia/DiffLinearAlgebra.jl/blob/master/src/util.jl#L9). These structs contain information regarding the arguments types supported by each sensitivity, and which arguments are differentiable.

