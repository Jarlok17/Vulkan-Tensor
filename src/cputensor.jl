# nebula/cputensor.jl 

export CPUTensor, zeros, rand, ones

import Base.reshape, Base.log
using CUDA

struct CPUTensor{T <: Number, N, A <: AbstractArray{T,N}}
   data::A
   shape::NTuple{N, Int}
   size::Int
   strides::NTuple{N, Int}
end

_alloc(::Val{:cpu}, f, T, dims::Int...) = f(T, dims...)
_alloc(::Val{:cuda}, f, T, dims::Int...) = f(T, dims...) |> CUDA.CuArray

CPUTensor(data::AbstractArray{T, N}) where {T, N} = CPUTensor{T, N, typeof(data)}(data, size(data), length(data), Base.strides(data))

CPUzeros(T::Type, dims::Int...; device =:cpu) = CPUTensor(_alloc(Val(device), zeros, T, dims...))
CPUzones(T::Type, dims::Int...; device=:cpu) = CPUTensor(_alloc(Val(device), ones, T, dims...))
CPUrand(T::Type, dims::Int...; device=:cpu) = CPUTensor(_alloc(Val(device), rand, T, dims...))

Base.broadcasted(f, a::CPUTensor, b::CPUTensor) = CPUTensor(broadcast(f, a.data, b.data))
Base.broadcasted(f, a::CPUTensor, b::Number) = CPUTensor(broadcast(f, a.data, b))
Base.broadcasted(f, a::Number, b::CPUTensor) = CPUTensor(broadcast(f, a, b.data))

# Tensor × Tensor
Base.:+(a::CPUTensor, b::CPUTensor) = CPUTensor(a.data .+ b.data)
Base.:-(a::CPUTensor, b::CPUTensor) = CPUTensor(a.data .- b.data)
Base.:/(a::CPUTensor, b::CPUTensor) = CPUTensor(a.data ./ b.data)

# Tensor × Number
Base.:+(a::CPUTensor, b::Number) = CPUTensor(a.data .+ b)
Base.:-(a::CPUTensor, b::Number) = CPUTensor(a.data .- b)
Base.:*(a::CPUTensor, b::Number) = CPUTensor(a.data .* b)
Base.:/(a::CPUTensor, b::Number) = CPUTensor(a.data ./ b)

# Number × Tensor
Base.:+(a::Number, b::CPUTensor) = CPUTensor(a .+ b.data)
Base.:-(a::Number, b::CPUTensor) = CPUTensor(a .- b.data)
Base.:*(a::Number, b::CPUTensor) = CPUTensor(a .* b.data)
Base.:/(a::Number, b::CPUTensor) = CPUTensor(a ./ b.data)

Base.:*(a::CPUTensor{T, 2}, b::CPUTensor{T, 2}) where {T} = CPUTensor(a.data * b.data)
Base.:*(a::CPUTensor{T, 2}, b::CPUTensor{T, 1}) where {T} = CPUTensor(a.data * b.data)
Base.:*(a::CPUTensor{T, 1}, b::CPUTensor{T, 2}) where {T} = CPUTensor(a.data' * b.data)
Base.:*(a::CPUTensor{T, 1}, b::CPUTensor{T, 1}) where {T} = sum(a.data .* b.data)

# mathematical functions
Base.exp(t::CPUTensor) = CPUTensor(exp.(t.data))
Base.log(b::T, t::CPUTensor) where {T <: Number} = CPUTensor(log.(b, t.data))
Base.log10(t::CPUTensor) = CPUTensor(log10.(t.data))
Base.log2(t::CPUTensor) = CPUTensor(log2.(t.data))
Base.argmax(t::CPUTensor) = argmax(t.data)
Base.argmin(t::CPUTensor) = argmin(t.data)

Base.transpose(t::CPUTensor) = CPUTensor(permutedims(t.data))
flatten(t::CPUTensor) = CPUTensor(reshape(t.data, :))

Base.sum(t::CPUTensor) = sum(t.data)
Base.maximum(t::CPUTensor) = maximum(t.data)
Base.minimum(t::CPUTensor) = minimum(t.data)
dot(a::CPUTensor{T,1}, b::CPUTensor{T,1}) where {T} = sum(a.data .* b.data)
outer(a::CPUTensor{T,1}, b::CPUTensor{T,1}) where {T} = CPUTensor(a.data .* b.data')

@inline function Base.reshape(t::CPUTensor, dims::Int...)
    if prod(dims) != length(t.data)
        throw(ArgumentError("cannot reshape array of size $(length(t.data)) into shape $(dims)"))
    end
    return CPUTensor(Base.reshape(t.data, dims...))
end

function prettyprint(x::Number) 
    # if number is integer, print with trailing dot for clarity
    if x == floor(x)
        return string(Int(x), ".")
    else
        return string(round(x, digits=4))
    end
end

@inline function Base.show(io::IO, t::CPUTensor)
    print(io, "tensor(")
    _show_array(io, t.data, 1)
    print(io, ", dtype=$(eltype(t.data)))")
end

function _show_array(io::IO, A::AbstractArray, indent::Int)
    nd = ndims(A)
    if nd == 1
        row = [prettyprint(x) for x in A[:]]
        print(io, "[", join(row, ", "), "]")
    else
        print(io, "[")
        for i in 1:size(A, 1)
            if i > 1
                print(io, "\n", " "^(indent+1))
            end
            _show_array(io, Base.view(A, i, :), indent+1)
            if i < size(A, 1)
                print(io, ",")
            end
        end
        print(io, "]")
    end
end
