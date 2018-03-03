import Base: -, trace, inv, det, logdet, transpose, vecnorm
import Compat.Base: ctranspose

############################# Unary sensitivities #############################
push!(ops, DiffOp(:(Base.:-), :(Tuple{DLA.AA}), [true]))
∇(::typeof(-), ::Arg1, p, Y::AA, Ȳ::AA, X::AA) = map(-, Ȳ)

push!(ops, DiffOp(:(Base.trace), :(Tuple{DLA.AM}), [true]))
∇(::typeof(trace), ::Arg1, p, Y::Real, Ȳ::Real, X::AM) = Diagonal(fill!(similar(X), Ȳ))

push!(ops, DiffOp(:(Base.inv), :(Tuple{DLA.AM}), [true]))
∇(::typeof(inv), ::Arg1, p, Y::AM, Ȳ::AM, X::AM) = -transpose(Y) * Ȳ * transpose(Y)

push!(ops, DiffOp(:(Base.det), :(Tuple{DLA.AM}), [true]))
∇(::typeof(det), ::Arg1, p, Y::Real, Ȳ::Real, X::AM) = Y * Ȳ * transpose(inv(X))

push!(ops, DiffOp(:(Base.logdet), :(Tuple{DLA.AM}), [true]))
∇(::typeof(logdet), ::Arg1, p, Y::Real, Ȳ::Real, X::AM) = Ȳ * transpose(inv(X))

push!(ops, DiffOp(:(Base.transpose), :(Tuple{DLA.AVM}), [true]))
∇(::typeof(transpose), ::Arg1, p, Y::AVM, Ȳ::AVM, X::AVM) = transpose(Ȳ)

push!(ops, DiffOp(:(Base.ctranspose), :(Tuple{DLA.AVM}), [true]))
∇(::typeof(ctranspose), ::Arg1, p, Y::AVM, Ȳ::AVM, X::AVM) = ctranspose(Ȳ)

push!(ops, DiffOp(:(Base.vecnorm), :(Tuple{DLA.AA}), [true]))
∇(::typeof(vecnorm), ::Arg1, p, Y::Real, Ȳ::Real, X::AA) = Ȳ ./ Y .* abs2.(X) ./ X

push!(ops, DiffOp(:(Base.vecnorm), :(Tuple{Real}), [true]))
∇(::typeof(vecnorm), ::Arg1, p, Y::Real, Ȳ::Real, X::Real) = Ȳ * sign(X)

############################# Binary sensitivities #############################
push!(ops, DiffOp(:(Base.:*), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(*), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ * B'
∇(::typeof(*), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = A' * Ȳ

push!(ops, DiffOp(:(Base.At_mul_B), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(At_mul_B), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = B * Ȳ'
∇(::typeof(At_mul_B), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = A * Ȳ

push!(ops, DiffOp(:(Base.A_mul_Bt), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(A_mul_Bt), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ * B
∇(::typeof(A_mul_Bt), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ' * A

push!(ops, DiffOp(:(Base.At_mul_Bt), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(At_mul_Bt), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = B' * Ȳ'
∇(::typeof(At_mul_Bt), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ' * A'

push!(ops, DiffOp(:(Base.Ac_mul_B), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(Ac_mul_B), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = B * Ȳ'
∇(::typeof(Ac_mul_B), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = A * Ȳ

push!(ops, DiffOp(:(Base.A_mul_Bc), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(A_mul_Bc), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ * B
∇(::typeof(A_mul_Bc), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ' * A

push!(ops, DiffOp(:(Base.Ac_mul_Bc), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(Ac_mul_Bc), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = B' * Ȳ'
∇(::typeof(Ac_mul_Bc), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ' * A'

push!(ops, DiffOp(:(Base.:/), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(/), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ / B'
∇(::typeof(/), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -Y' * (Ȳ / B')

push!(ops, DiffOp(:(Base.At_rdiv_B), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(At_rdiv_B), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = B \ Ȳ'
∇(::typeof(At_rdiv_B), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -Y' * (B \ Ȳ')'

push!(ops, DiffOp(:(Base.A_rdiv_Bt), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(A_rdiv_Bt), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ / B
∇(::typeof(A_rdiv_Bt), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -(B' \ Ȳ') * Y

push!(ops, DiffOp(:(Base.At_rdiv_Bt), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(At_rdiv_Bt), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = B' \ Ȳ'
∇(::typeof(At_rdiv_Bt), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -(B' \ Ȳ') * Y

push!(ops, DiffOp(:(Base.Ac_rdiv_B), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(Ac_rdiv_B), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = B \ Ȳ'
∇(::typeof(Ac_rdiv_B), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -Y' * (B \ Ȳ')'

push!(ops, DiffOp(:(Base.A_rdiv_Bc), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(A_rdiv_Bc), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ / B
∇(::typeof(A_rdiv_Bc), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -(B' \ Ȳ') * Y

push!(ops, DiffOp(:(Base.Ac_rdiv_Bc), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(Ac_rdiv_Bc), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = B' \ Ȳ'
∇(::typeof(Ac_rdiv_Bc), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -(B' \ Ȳ') * Y

push!(ops, DiffOp(:(Base.:\), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(\), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -(A' \ Ȳ) * Y'
∇(::typeof(\), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = A' \ Ȳ

push!(ops, DiffOp(:(Base.At_ldiv_B), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(At_ldiv_B), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -Y *(A \ Ȳ)'
∇(::typeof(At_ldiv_B), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = A \ Ȳ

push!(ops, DiffOp(:(Base.A_ldiv_Bt), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(A_ldiv_Bt), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -(Ȳ' / A)' * Y'
∇(::typeof(A_ldiv_Bt), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ' / A

push!(ops, DiffOp(:(Base.At_ldiv_Bt), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(At_ldiv_Bt), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -Y * (Ȳ' / A')
∇(::typeof(At_ldiv_Bt), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ' / A'

push!(ops, DiffOp(:(Base.Ac_ldiv_B), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(Ac_ldiv_B), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -Y * (A \ Ȳ)'
∇(::typeof(Ac_ldiv_B), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = A \ Ȳ

push!(ops, DiffOp(:(Base.A_ldiv_Bc), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(A_ldiv_Bc), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -(Ȳ' / A)' * Y'
∇(::typeof(A_ldiv_Bc), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ' / A

push!(ops, DiffOp(:(Base.Ac_ldiv_Bc), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(Ac_ldiv_Bc), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -Y * (Ȳ' / A')
∇(::typeof(Ac_ldiv_Bc), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ' / A'

push!(ops, DiffOp(:(Base.vecnorm), :(Tuple{DLA.AA, Real}), [true, true]))
∇(::typeof(vecnorm), ::Arg1, p, Y::Real, Ȳ::Real, A::AA, B::Real) =
    Ȳ .* Y^(1 - B) .* abs.(A).^B ./ A
∇(::typeof(vecnorm), ::Arg2, p, Y::Real, Ȳ::Real, A::AA, B::Real) =
    Ȳ * (Y^(1 - B) * sum(abs.(A).^B .* log.(abs.(A))) - Y * log(Y)) / B

push!(ops, DiffOp(:(Base.vecnorm), :(Tuple{Real, Real}), [true, true]))
∇(::typeof(vecnorm), ::Arg1, p, Y::Real, Ȳ::Real, A::Real, B::Real) = Ȳ * sign(A)
∇(::typeof(vecnorm), ::Arg2, p, Y::Real, Ȳ::Real, A::Real, B::Real) = 0
