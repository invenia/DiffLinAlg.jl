@testset "Triangular" begin

    for f in [LowerTriangular, UpperTriangular]
        let P = 10, rng = MersenneTwister(123456), N = 10

            sc = ()->randn(rng)
            mPP = ()->abs.(randn(rng, P, P))
            dPP = ()->Diagonal(abs.(randn(rng, P)))
            tPP = ()->f(abs.(randn(rng, P, P)))
            m0 = fill!(similar(mPP()), zero(eltype(mPP())))

            # Construction.
            @test check_errs(N, unary_ȲD(f)..., tPP, mPP, mPP)
            @test check_errs(N, unary_ȲD_inplace(f, m0)..., tPP, mPP, mPP)

            # Determinant.
            @test check_errs(N, unary_ȲD(det)..., sc, tPP, tPP)
            @test check_errs(N, unary_ȲD_inplace(det, zero(tPP()))..., sc, tPP, tPP)
            @test check_errs(N, unary_ȲD_inplace(det, zero(dPP()))..., sc, tPP, tPP)

            # Log Determinant.
            @test check_errs(N, unary_ȲD(logdet)..., sc, tPP, tPP)
            @test check_errs(N, unary_ȲD_inplace(logdet, zero(tPP()))..., sc, tPP, tPP)
            @test check_errs(N, unary_ȲD_inplace(logdet, zero(dPP()))..., sc, tPP, tPP)
        end
    end
end
