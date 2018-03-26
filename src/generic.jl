import Base: -
import LinearAlgebra: tr, inv, det, logdet, transpose, adjoint, vecnorm, kron

############################# Unary sensitivities #############################
push!(ops, DiffOp(:(Base.:-), :(Tuple{DLA.AA}), [true]))
∇(::typeof(-), ::Arg1, p, Y::AA, Ȳ::AA, X::AA) = map(-, Ȳ)

push!(ops, DiffOp(:(LinearAlgebra.tr), :(Tuple{DLA.AM}), [true]))
∇(::typeof(tr), ::Arg1, p, Y::Real, Ȳ::Real, X::AM) = Diagonal(fill!(similar(X), Ȳ))

push!(ops, DiffOp(:(LinearAlgebra.inv), :(Tuple{DLA.AM}), [true]))
∇(::typeof(inv), ::Arg1, p, Y::AM, Ȳ::AM, X::AM) = -transpose(Y) * Ȳ * transpose(Y)

push!(ops, DiffOp(:(LinearAlgebra.det), :(Tuple{DLA.AM}), [true]))
∇(::typeof(det), ::Arg1, p, Y::Real, Ȳ::Real, X::AM) = Y * Ȳ * transpose(inv(X))

push!(ops, DiffOp(:(LinearAlgebra.logdet), :(Tuple{DLA.AM}), [true]))
∇(::typeof(logdet), ::Arg1, p, Y::Real, Ȳ::Real, X::AM) = Ȳ * transpose(inv(X))

push!(ops, DiffOp(:(LinearAlgebra.transpose), :(Tuple{DLA.AVM}), [true]))
∇(::typeof(transpose), ::Arg1, p, Y::AVM, Ȳ::AVM, X::AVM) = transpose(Ȳ)

push!(ops, DiffOp(:(LinearAlgebra.adjoint), :(Tuple{DLA.AVM}), [true]))
∇(::typeof(adjoint), ::Arg1, p, Y::AVM, Ȳ::AVM, X::AVM) = adjoint(Ȳ)

push!(ops, DiffOp(:(LinearAlgebra.vecnorm), :(Tuple{DLA.AA}), [true]))
∇(::typeof(vecnorm), ::Arg1, p, Y::Real, Ȳ::Real, X::AA) = Ȳ ./ Y .* abs2.(X) ./ X

push!(ops, DiffOp(:(LinearAlgebra.vecnorm), :(Tuple{Real}), [true]))
∇(::typeof(vecnorm), ::Arg1, p, Y::Real, Ȳ::Real, X::Real) = Ȳ * sign(X)

############################# Binary sensitivities #############################
push!(ops, DiffOp(:(LinearAlgebra.:*), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(*), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ * B'
∇(::typeof(*), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = A' * Ȳ

push!(ops, DiffOp(:(LinearAlgebra.:/), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(/), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = Ȳ / B'
∇(::typeof(/), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -Y' * (Ȳ / B')

push!(ops, DiffOp(:(LinearAlgebra.:\), :(Tuple{DLA.AVM, DLA.AVM}), [true, true]))
∇(::typeof(\), ::Arg1, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = -(A' \ Ȳ) * Y'
∇(::typeof(\), ::Arg2, p, Y::ASVM, Ȳ::ASVM, A::AVM, B::AVM) = A' \ Ȳ

push!(ops, DiffOp(:(LinearAlgebra.vecnorm), :(Tuple{DLA.AA, Real}), [true, true]))
∇(::typeof(vecnorm), ::Arg1, p, Y::Real, Ȳ::Real, A::AA, B::Real) =
    Ȳ .* Y^(1 - B) .* abs.(A).^B ./ A
∇(::typeof(vecnorm), ::Arg2, p, Y::Real, Ȳ::Real, A::AA, B::Real) =
    Ȳ * (Y^(1 - B) * sum(abs.(A).^B .* log.(abs.(A))) - Y * log(Y)) / B

push!(ops, DiffOp(:(LinearAlgebra.vecnorm), :(Tuple{Real, Real}), [true, true]))
∇(::typeof(vecnorm), ::Arg1, p, Y::Real, Ȳ::Real, A::Real, B::Real) = Ȳ * sign(A)
∇(::typeof(vecnorm), ::Arg2, p, Y::Real, Ȳ::Real, A::Real, B::Real) = 0

push!(ops, DiffOp(:(LinearAlgebra.kron), :(Tuple{AM, AM}), [true, true]))
∇(::typeof(kron), ::Type{Val{1}}, p, Y::AM, Ȳ::AM, A::AM, B::AM) =
    ∇(zero(A), kron, Val{1}, p, Y, Ȳ, A, B)
∇(::typeof(kron), ::Type{Val{2}}, p, Y::AM, Ȳ::AM, A::AM, B::AM) =
    ∇(zero(B), kron, Val{2}, p, Y, Ȳ, A, B)
function ∇(Ā::AM, ::typeof(kron), ::Type{Val{1}}, p, Y::AM, Ȳ::AM, A::AM, B::AM)
    @assert size(Ā) == size(A)
    (I, J), (K, L), m = size(A), size(B), length(Y)
    @inbounds for j = reverse(1:J), l = reverse(1:L), i = reverse(1:I)
        āij = Ā[i, j]
        for k = reverse(1:K)
            āij += Ȳ[m] * B[k, l]
            m -= 1
        end
        Ā[i, j] = āij
    end
    return Ā
end
function ∇(B̄::AM, ::typeof(kron), ::Type{Val{2}}, p, Y::AM, Ȳ::AM, A::AM, B::AM)
    @assert size(B̄) == size(B)
    (I, J), (K, L), m = size(A), size(B), length(Y)
    @inbounds for j = reverse(1:J), l = reverse(1:L), i = reverse(1:I)
        aij = A[i, j]
        for k = reverse(1:K)
            B̄[k, l] += aij * Ȳ[m]
            m -= 1
        end
    end
    return B̄
end
