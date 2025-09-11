# nebula/rawtensor.jl
include("init_vulkan.jl")

using .InitVulkan
using VulkanCore.LibVulkan

export RawTensor, Array

InitVulkan.init_vulkan()

abstract type AbstractGPUTensor{T, N} <: AbstractArray{T, N} end

mutable struct RawTensor{T,N} <: AbstractGPUTensor{T, N}
	buffer::BufferBlock
	dims::Dims{N}
end

function RawTensor(arr::Array{T, N}) where {T, N}
	nbytes = UInt64(sizeof(arr))
	
	staging = create_gpu_buffer(InitVulkan.devices[].logical_device, InitVulkan.m_props[], nbytes;
			     usage = UInt32(VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
			     properties = UInt32(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))

	write_to_staging(staging, arr)

	# gpu buffer
	buf = create_gpu_buffer(InitVulkan.devices[].logical_device, InitVulkan.m_props[], nbytes;
			 usage = UInt32(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		   VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
			 properties = UInt32(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))

	# copy staging -> gpu
	cmd_pool = CommandPoolInfo(InitVulkan.devices[].logical_device, VkCommandBuffer(C_NULL), VkCommandPool(C_NULL))
	InitVulkan.create_command_pool(cmd_pool, InitVulkan.devices[].queue_family)
	copy_buffer(cmd_pool, InitVulkan.devices[].queue, staging, buf, nbytes)

	return RawTensor{T,N}(buf, size(arr))
end

function Array(t::RawTensor{T,N}) where {T,N}
	nbytes = UInt64(prod(t.dims) * sizeof(T))

	readback = create_gpu_buffer(InitVulkan.devices[].logical_device, InitVulkan.m_props[], nbytes;
			      usage = UInt32(VK_BUFFER_USAGE_TRANSFER_DST_BIT),
			      properties = UInt32(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))

	cmd_pool = CommandPoolInfo(InitVulkan.devices[].logical_device, VkCommandBuffer(C_NULL), VkCommandPool(C_NULL))
	InitVulkan.create_command_pool(cmd_pool, InitVulkan.devices[].queue_family)
	copy_buffer(cmd_pool, InitVulkan.devices[].queue, t.buffer, readback, nbytes)

	return reshape(readback_aligned(T, readback, prod(t.dims), sizeof(T)), t.dims)
end

# Arithmetic operations
function add!(out::RawTensor, a::RawTensor, b::RawTensor)
	update_descriptor_sets(
		InitVulkan.devices[].logical_device,
		InitVulkan.descriptor_sets[:add],
		a.buffer, b.buffer, out.buffer
	)
	N = prod(out.dims)
	launch_arithmetic!(
		InitVulkan.cmd_pool[],
		InitVulkan.devices[].queue,
		InitVulkan.pipelines[:add].pipeline,
		InitVulkan.pipelines[:add].pipelineLayout,
		InitVulkan.descriptor_sets[:add],
		N,
	)
	return out
end

@inline function add(a::RawTensor, b::RawTensor)
	@assert a.dims == b.dims

	out = RawTensor(zeros(Float32, a.dims))
	add!(out, a, b)
	return out
end

function minus!(out::RawTensor, a::RawTensor, b::RawTensor)
	update_descriptor_sets(
		InitVulkan.devices[].logical_device,
		InitVulkan.descriptor_sets[:sub],
		a.buffer, b.buffer, out.buffer
	)
	N = prod(out.dims)
	launch_arithmetic!(
		InitVulkan.cmd_pool[],
		InitVulkan.devices[].queue,
		InitVulkan.pipelines[:sub].pipeline,
		InitVulkan.pipelines[:sub].pipelineLayout,
		InitVulkan.descriptor_sets[:sub],
		N,
	)
	return out
end

@inline function minus(a::RawTensor, b::RawTensor)
	out = RawTensor(zeros(Float32, a.dims))
	minus!(out, a, b)
	return out
end

function mul!(out::RawTensor, a::RawTensor, b::RawTensor)
	update_descriptor_sets(
		InitVulkan.devices[].logical_device,
		InitVulkan.descriptor_sets[:mul],
		a.buffer, b.buffer, out.buffer
	)
	N = prod(out.dims)
	launch_arithmetic!(
		InitVulkan.cmd_pool[],
		InitVulkan.devices[].queue,
		InitVulkan.pipelines[:mul].pipeline,
		InitVulkan.pipelines[:mul].pipelineLayout,
		InitVulkan.descriptor_sets[:mul],
		N,
	)
	return out
end

@inline function multiply(a::RawTensor, b::RawTensor)
	@assert a.dims == b.dims
	out = RawTensor(zeros(Float32, a.dims))
	mul!(out, a, b)
	return out
end

function div!(out::RawTensor, a::RawTensor, b::RawTensor)
	update_descriptor_sets(
		InitVulkan.devices[].logical_device,
		InitVulkan.descriptor_sets[:div],
		a.buffer, b.buffer, out.buffer
	)
	N = prod(out.dims)
	launch_arithmetic!(
		InitVulkan.cmd_pool[],
		InitVulkan.devices[].queue,
		InitVulkan.pipelines[:div].pipeline,
		InitVulkan.pipelines[:div].pipelineLayout,
		InitVulkan.descriptor_sets[:div],
		N,
	)
	return out
end

@inline function divide(a::RawTensor, b::RawTensor)
	@assert a.dims == b.dims
	out = RawTensor(zeros(Float32, a.dims))
	div!(out, a, b)
	return out
end

@inline function Base.show(io::IO, t::RawTensor)
    arr = Array(t)
    print(io, "RawTensor(")
    _show_array(io, arr, 1)
    print(io, ", dims=$(t.dims), dtype=$(eltype(arr)))")
end

function _show_array(io::IO, A::AbstractArray, indent::Int)
    nd = ndims(A)
    if nd == 1
        row = [string(x) for x in A[:]]
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
