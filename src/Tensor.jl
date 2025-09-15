module Tensor

include("cputensor.jl")
include("vulkantensor.jl")

export GPUTensor, CPUTensor, add, minus, multiply, divide

end
