module Tensor

include("cputensor.jl")
include("vulkantensor.jl")

export RawTensor, CPUTensor, add, minus, multiply, divide

end
