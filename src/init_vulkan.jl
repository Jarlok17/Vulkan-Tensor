# nebula/init_vulkan.jl 

module InitVulkan
using VulkanCore.LibVulkan

export init_vulkan, devices, m_props, create_gpu_buffer, write_to_staging,
copy_buffer, readback_aligned, update_descriptor_sets, CommandPoolInfo, BufferBlock, launch_arithmetic!

const VK_NULL_HANDLE = C_NULL

mutable struct VulkanInfo
	app_name::String
	engine_name::String
end

VulkanInfo(; app_n::String = "NebulaApp", engine_n::String = "NebulaEngine") =
	VulkanInfo(app_n, engine_n)

const VK_VERSION_1_0 = VK_MAKE_VERSION(1, 0, 0)

function create_instance(init_info::VulkanInfo)
	println("VK_API_VERSION_1_0: ", VK_API_VERSION_1_0)
	app_name_c   = Base.unsafe_convert(Ptr{Cchar}, Base.cconvert(Cstring, init_info.app_name))
	engine_name_c = Base.unsafe_convert(Ptr{Cchar}, Base.cconvert(Cstring, init_info.engine_name))

	app_info = Ref(VkApplicationInfo(
		VK_STRUCTURE_TYPE_APPLICATION_INFO,
		C_NULL,
		app_name_c,
		1,
		engine_name_c,
		1,
		VK_VERSION_1_0
	))

	instance_info = Ref(VkInstanceCreateInfo(
		VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		C_NULL,
		0,
		Base.unsafe_convert(Ptr{VkApplicationInfo}, app_info),
		0,
		C_NULL,
		0,
		C_NULL
	))

	instance = Ref{VkInstance}()

	GC.@preserve app_info instance_info begin
		err = vkCreateInstance(instance_info, C_NULL, instance)
		@assert err == VK_SUCCESS
	end
	println("INSTANCE IS CREATED: ", err)
	return instance[]
end

mutable struct DevicesInfo
	count::Ref{Cuint}
	physical_devices::Vector{VkPhysicalDevice}
	physical_device::VkPhysicalDevice
	props::VkPhysicalDeviceProperties
	name::String

	# logical device
	logical_device::VkDevice
	queue_family::Cuint
	queue::VkQueue

end

function ntuple_into_string(t::NTuple{N, Int8}) where {N}
	data = Vector{UInt8}(undef, N)
	for i in 1:N
		data[i] = reinterpret(UInt8, t[i])
	end
	nul = findfirst(==(0x00), data)
	if isnothing(nul)
		return String(data)
	else
		return String(data[1:nul-1])
	end
end

struct PhysDeviceCriteria
	min_api_version::UInt32
	required_device_type::VkPhysicalDeviceType
	name_contains::Union{Nothing, String}
end

function choose_physical_device(devices::Vector{VkPhysicalDevice}, criteria::PhysDeviceCriteria)
	for device in devices
		props_ref = Ref{VkPhysicalDeviceProperties}()
		vkGetPhysicalDeviceProperties(device, props_ref)
		props = props_ref[]
		gpu_name = ntuple_into_string(props.deviceName)

		if props.apiVersion >= criteria.min_api_version &&
			(criteria.required_device_type == VK_PHYSICAL_DEVICE_TYPE_OTHER ||
				props.deviceType == criteria.required_device_type) &&
			(criteria.name_contains === nothing || occursin(criteria.name_contains, gpu_name))
			return device, props, gpu_name
		end
	end
	error("No GPU matches the given criteria")
end 

function create_logical_device(phys_dev::VkPhysicalDevice, queue_family::Cuint)
	priority = Ref{Cfloat}(1.0f0)

	queue_info = Ref(VkDeviceQueueCreateInfo(
		VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		C_NULL,
		0,
		queue_family,
		1,
		Base.unsafe_convert(Ptr{Cfloat}, priority)
	))

	device_info = Ref(VkDeviceCreateInfo(
		VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		C_NULL,
		0,
		1,
		Base.unsafe_convert(Ptr{VkDeviceQueueCreateInfo}, queue_info),
		0, C_NULL,
		0, C_NULL,
		C_NULL
	))

	device = Ref{VkDevice}()
	GC.@preserve priority queue_info device_info begin
		err = vkCreateDevice(phys_dev, device_info, C_NULL, device)
		@assert err == VK_SUCCESS
	end
	return device[]
end

function find_queue_family(device::VkPhysicalDevice)
	count = Ref{Cuint}(0)
	vkGetPhysicalDeviceQueueFamilyProperties(device, count, C_NULL)
	families = Vector{VkQueueFamilyProperties}(undef, count[])
	vkGetPhysicalDeviceQueueFamilyProperties(device, count, families)

	for (i, fam) in enumerate(families)
		if (fam.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0
			return Cuint(i - 1)
		end
	end
	error("No compute queue found")
end

function get_device_queue(device::VkDevice, queue_family::Cuint)
	queue = Ref{VkQueue}()
	vkGetDeviceQueue(device, queue_family, 0, queue)
	return queue[]
end

function DevicesInfo(instance::VkInstance; criteria::PhysDeviceCriteria = 
		     PhysDeviceCriteria(VK_VERSION_1_0, VK_PHYSICAL_DEVICE_TYPE_OTHER, nothing),
		     with_logical::Bool = true)

	count = Ref{Cuint}(0)
	err = vkEnumeratePhysicalDevices(instance, count, C_NULL)
	@assert err == VK_SUCCESS
	devices = Vector{VkPhysicalDevice}(undef, count[])
	err = vkEnumeratePhysicalDevices(instance, count, devices)
	@assert err == VK_SUCCESS

	physical, props, gpu_name = choose_physical_device(devices, criteria)

	qfam = find_queue_family(physical)

	if with_logical
		logical = create_logical_device(physical, qfam)
		queue   = get_device_queue(logical, qfam)
	else
		logical = VkDevice(C_NULL)
		queue   = VkQueue(C_NULL)
	end

	return DevicesInfo(count, devices, physical, props, gpu_name, logical, qfam, queue)
end

function find_memory_type(mem_props::VkPhysicalDeviceMemoryProperties, type_bits::UInt32, properties::VkMemoryPropertyFlags)
	for i in 0:mem_props.memoryTypeCount-1
		if (type_bits & (1 << i)) != 0 && (mem_props.memoryTypes[i+1].propertyFlags & properties) == properties
			return i
		end
	end
	error("No suitable memory type found")
end

mutable struct BufferBlock
	device::VkDevice
	buffer::Union{VkBuffer, Nothing}
	deviceMemory::Union{VkDeviceMemory, Nothing}
	size::UInt64
	usage::VkBufferUsageFlags
	sharingMode::VkSharingMode
end

function create_buffer_block(device::VkDevice, size::UInt64;
			     usage::VkBufferUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			     sharing::VkSharingMode = VK_SHARING_MODE_EXCLUSIVE)
	buffer_info = Ref(VkBufferCreateInfo(
		VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		C_NULL,
		0,
		size,
		usage,
		sharing,
		0,
		C_NULL
	))
	buffer = Ref{VkBuffer}()
	err = vkCreateBuffer(device, buffer_info, C_NULL, buffer)
	@assert err == VK_SUCCESS
	block_buffer = buffer[]

	return BufferBlock(device, block_buffer, nothing, size, usage, sharing)
end

function alloc_buffer(block::BufferBlock, mem_props::VkPhysicalDeviceMemoryProperties, flags::UInt32)
	mem_req = Ref{VkMemoryRequirements}()
	vkGetBufferMemoryRequirements(block.device, block.buffer, mem_req)
	bmr = mem_req[]

	mem_index = find_memory_type(mem_props, bmr.memoryTypeBits, flags)

	alloc_info = Ref(VkMemoryAllocateInfo(
		VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		C_NULL,
		bmr.size,
		mem_index
	))

	memory = Ref{VkDeviceMemory}()
	err = vkAllocateMemory(block.device, alloc_info, C_NULL, memory)
	@assert err == VK_SUCCESS
	block.deviceMemory = memory[]
end

function bind_buffer(block::BufferBlock)
	err = vkBindBufferMemory(block.device, block.buffer, block.deviceMemory, 0)
	@assert err == VK_SUCCESS
end

function create_gpu_buffer(device::VkDevice, mem_props::VkPhysicalDeviceMemoryProperties, 
			   size::UInt64; usage::UInt32, properties::UInt32)
	buf = create_buffer_block(device, size; usage=VkBufferUsageFlags(usage))
	alloc_buffer(buf, mem_props, properties)
	bind_buffer(buf)
	return buf
end

function write_to_staging(block::BufferBlock, x)
	nbytes = length(x) * Base.elsize(x)

	@assert nbytes <= block.size  "host data bigger than staging buffer"

	ptr = Ref{Ptr{Cvoid}}()
	err = vkMapMemory(block.device, block.deviceMemory, 0, nbytes, 0, ptr)
	@assert err == VK_SUCCESS

	GC.@preserve x begin
		src = Base.unsafe_convert(Ptr{UInt8}, pointer(x))
		dst = Base.unsafe_convert(Ptr{UInt8}, ptr[])
		Base.unsafe_copyto!(dst, src, nbytes)
	end

	vkUnmapMemory(block.device, block.deviceMemory)
end

mutable struct CommandPoolInfo
	device::VkDevice
	commandBuffer::VkCommandBuffer
	commandPool::VkCommandPool
end

function create_command_pool(commandPool::CommandPoolInfo, queue_family::UInt32)
	info = VkCommandPoolCreateInfo(
		VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		C_NULL,
		0,
		queue_family
	)
	pool = Ref{VkCommandPool}()
	err = vkCreateCommandPool(commandPool.device, Ref(info), C_NULL, pool)
	@assert err == VK_SUCCESS
	commandPool.commandPool = pool[]
end

function begin_single_time_commands(commandPool::CommandPoolInfo)
	alloc_info = VkCommandBufferAllocateInfo(
		VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		C_NULL,
		commandPool.commandPool,
		VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		1
	)

	cmd_buf = Ref{VkCommandBuffer}()
	vkAllocateCommandBuffers(commandPool.device, Ref(alloc_info), cmd_buf)

	begin_info = VkCommandBufferBeginInfo(
		VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		C_NULL,
		VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		C_NULL
	)

	vkBeginCommandBuffer(cmd_buf[], Ref(begin_info))
	commandPool.commandBuffer = cmd_buf[]
	return cmd_buf[]
end

function end_single_time_commands(commandPool::CommandPoolInfo, queue::VkQueue)
	vkEndCommandBuffer(commandPool.commandBuffer)

	p_cmd = Ref(commandPool.commandBuffer)

	submit_info = VkSubmitInfo(
		VK_STRUCTURE_TYPE_SUBMIT_INFO,
		C_NULL,
		0, C_NULL, C_NULL,
		1, Base.unsafe_convert(Ptr{VkCommandBuffer}, p_cmd),
		0, C_NULL
	)

	GC.@preserve p_cmd begin
		err = vkQueueSubmit(queue, 1, Ref(submit_info), VK_NULL_HANDLE)
		@assert err == VK_SUCCESS
		vkQueueWaitIdle(queue)
	end

	vkFreeCommandBuffers(commandPool.device, commandPool.commandPool, 1, Ref(commandPool.commandBuffer))
end

function copy_buffer(commandPool::CommandPoolInfo, queue::VkQueue, 
		     src::BufferBlock, dst::BufferBlock, size::UInt64)
	cmd = begin_single_time_commands(commandPool)

	region = VkBufferCopy(0, 0, size)
	vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, Ref(region))

	end_single_time_commands(commandPool, queue)
end

function readback_aligned(T, block::BufferBlock, count::Int, alignment::Int)
	stride = max(sizeof(T), alignment)

	pptr = Ref{Ptr{Cvoid}}()
	err = vkMapMemory(block.device, block.deviceMemory, 0, stride*count, 0, pptr)
	@assert err == VK_SUCCESS

	raw = Ptr{UInt8}(pptr[])

	vals = [unsafe_load(Ptr{T}(raw + stride*(i-1))) for i in 1:count]

	vkUnmapMemory(block.device, block.deviceMemory)
	return vals
end

function create_bindings(num_buffers::Int64)
    return [
        VkDescriptorSetLayoutBinding(
            i,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            1,
            VK_SHADER_STAGE_COMPUTE_BIT,
            C_NULL
        ) for i in 0:num_buffers-1
    ]
end

function create_descriptior_set_layout(device::VkDevice, bindings::Vector{VkDescriptorSetLayoutBinding})
	descr_set_info = VkDescriptorSetLayoutCreateInfo(
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		C_NULL,
		0,
		length(bindings),
		pointer(bindings)
	)
	layout = Ref{VkDescriptorSetLayout}()
	GC.@preserve bindings begin
		err = vkCreateDescriptorSetLayout(device, Ref(descr_set_info), C_NULL, layout)
		@assert err == VK_SUCCESS
	end
	return layout[] 
end

function create_descriptior_pool(device::VkDevice)
	pool_sizes = [
		VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3)
	]

	pool_info = VkDescriptorPoolCreateInfo(
		VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		C_NULL,
		0,
		1,  # maxSets
		length(pool_sizes),  # poolSizeCount
		pointer(pool_sizes)
	)

	pool = Ref{VkDescriptorPool}()
	GC.@preserve pool begin
		err = vkCreateDescriptorPool(device, Ref(pool_info), C_NULL, pool)
		@assert err == VK_SUCCESS
	end
	return pool[]
end

function allocate_descriptor_set(device::VkDevice, pool::VkDescriptorPool, setLayout::VkDescriptorSetLayout)
	layouts = [setLayout]
	alloc_info = VkDescriptorSetAllocateInfo(
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		C_NULL,
		pool,
		length(layouts),
		pointer(layouts)
	)

	desc_set = Ref{VkDescriptorSet}()
	GC.@preserve desc_set begin
		err = vkAllocateDescriptorSets(device, Ref(alloc_info), desc_set)
		@assert err == VK_SUCCESS
	end	
	return desc_set[]
end


function update_descriptor_sets(device::VkDevice,
				desc_set::VkDescriptorSet,
				buffers::BufferBlock...)
	infos = [VkDescriptorBufferInfo(buf.buffer, 0, buf.size) for buf in buffers]

	writes = [
		VkWriteDescriptorSet(
			VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			C_NULL,
			desc_set,
			i,
			0,
			1,
			VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			C_NULL,
			pointer(infos) + (i * sizeof(VkDescriptorBufferInfo)),
			C_NULL
		) for i in 0:length(buffers)-1
	]

	GC.@preserve infos desc_set begin
		vkUpdateDescriptorSets(device,
			 UInt32(length(writes)),
			 pointer(writes),
			 0, C_NULL)
	end
end


function dispatch_compute(commandPool::CommandPoolInfo, 
			  queue::VkQueue, 
			  pipeline::VkPipeline,
			  pipelineLayout::VkPipelineLayout,
			  desc_set::VkDescriptorSet,
			  x::UInt32, y::UInt32, z::UInt32,
			  push_constant::Vector{UInt8} = UInt8[])
	cmd_buf = begin_single_time_commands(commandPool)

	vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)

	desc_sets = [desc_set]
	GC.@preserve desc_sets begin
		vkCmdBindDescriptorSets(
			cmd_buf,
			VK_PIPELINE_BIND_POINT_COMPUTE,
			pipelineLayout,
			0,
			1, pointer(desc_sets),
			0, C_NULL
		)	
	end
	
	if !isempty(push_constant)
		vkCmdPushConstants(
			commandPool.commandBuffer,
			pipelineLayout,
			VK_SHADER_STAGE_COMPUTE_BIT,
			0,
			length(push_constant),
			pointer(push_constant)
		)
	end

	vkCmdDispatch(cmd_buf, x, y, z)

	end_single_time_commands(commandPool, queue)
end

function launch_arithmetic!(cmd_pool::CommandPoolInfo,
                            queue::VkQueue,
                            pipeline::VkPipeline,
                            pipelineLayout::VkPipelineLayout,
			    desc_set::VkDescriptorSet,
			    N::Int; local_size::Int = 64, push_constant::Vector{UInt8} = UInt8[])
	wg = UInt32(cld(N, local_size))
	dispatch_compute(cmd_pool, queue, pipeline, pipelineLayout, desc_set,
		  wg, UInt32(1), UInt32(1), push_constant)
end

mutable struct PipelineInfo
	device::VkDevice
	pipelineLayout::VkPipelineLayout
	pipeline::VkPipeline
end

function create_pipeline_layout(device::VkDevice, setLayout::VkDescriptorSetLayout)
	layouts = [setLayout]
	pip_lay_info = VkPipelineLayoutCreateInfo(
		VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		C_NULL,
		0,
		length(layouts),
		pointer(layouts),
		0,
		C_NULL
	)

	layout = Ref{VkPipelineLayout}()
	err = vkCreatePipelineLayout(device, Ref(pip_lay_info), C_NULL, layout)
	@assert err == VK_SUCCESS
	return layout[]
end

function create_shader_module(device::VkDevice, filename::String)
	code_bytes = read(filename)
	@assert length(code_bytes) % 4 == 0 "SPIR-V size must be multiple of 4"

	code = reinterpret(UInt32, code_bytes)

	create_info = VkShaderModuleCreateInfo(
		VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		C_NULL,
		0,
		length(code) * sizeof(UInt32),
		pointer(code)
	)

	shader_module = Ref{VkShaderModule}()
	err = vkCreateShaderModule(device, Ref(create_info), C_NULL, shader_module)
	@assert err == VK_SUCCESS "Failed to create shader module"
	return shader_module[]
end

function create_pipeline(device::VkDevice, pipelineLayout::VkPipelineLayout, shaderModule::VkShaderModule, entryPoint::String)
	entry_cstr = Base.cconvert(Cstring, entryPoint)
	entry_ptr = Base.unsafe_convert(Ptr{Cchar}, entry_cstr)

	shader_stage = VkPipelineShaderStageCreateInfo(
		VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		C_NULL,
		0,
		VK_SHADER_STAGE_COMPUTE_BIT,
		shaderModule,
		entry_ptr,
		C_NULL
	)

	pipeline_info = VkComputePipelineCreateInfo(
		VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		C_NULL,
		0,
		shader_stage,
		pipelineLayout,
		C_NULL,
		-1
	)

	pipeline = Ref{VkPipeline}()
	err = vkCreateComputePipelines(
		device,
		VK_NULL_HANDLE,
		1,
		Ref(pipeline_info),
		C_NULL,
		pipeline
	)
	@assert err == VK_SUCCESS "Failed to create compute pipeline"
	return pipeline[]
end

function generate_spv(comp_path::String, spv_path::String)::String
	gsls_command = "glslc " * comp_path * " -o " * spv_path
	run(glslc_command)
	return spv_path
end

const instance = Ref{VkInstance}()
const devices = Ref{DevicesInfo}()
const m_props = Ref{VkPhysicalDeviceMemoryProperties}()
const cmd_pool = Ref{CommandPoolInfo}()
const physical_device_props = Ref{VkPhysicalDeviceProperties}()
const pipelines = Dict{Symbol, PipelineInfo}()
const descriptor_sets = Dict{Symbol, VkDescriptorSet}()
const descriptor_pools = Dict{Symbol, VkDescriptorPool}()


function init_vulkan()
	inst = create_instance(VulkanInfo())
	instance[] = inst
	devices[] = DevicesInfo(inst; with_logical=true)
	mem = Ref{VkPhysicalDeviceMemoryProperties}()
	vkGetPhysicalDeviceMemoryProperties(devices[].physical_device, mem)
	m_props[] = mem[]
	
	vkGetPhysicalDeviceProperties(devices[].physical_device, physical_device_props)

	println("Max Invocations: ", physical_device_props[].limits.maxComputeWorkGroupInvocations)
	println("Max WorkGroup Size: X=", physical_device_props[].limits.maxComputeWorkGroupSize[1],
	 " Y=", physical_device_props[].limits.maxComputeWorkGroupSize[2],
	 " Z=", physical_device_props[].limits.maxComputeWorkGroupSize[3])

	cmd_pool[] = CommandPoolInfo(devices[].logical_device, VkCommandBuffer(C_NULL), VkCommandPool(C_NULL))
	create_command_pool(cmd_pool[], devices[].queue_family)

	shaders_dir = joinpath(@__DIR__, "shaders")
	spv_files = [
		("add", "add.spv"),
		("sub", "sub.spv"), 
		("mul", "mul.spv"),
		("div", "div.spv")
	]

	for (name, spv_file) in spv_files
		spv_path = joinpath(shaders_dir, spv_file)
		if !isfile(spv_path)
			error("SPV file not found: $spv_path. Please run compile_shaders.jl first.")
		end

		pip, pool, desc = init_pipeline(devices[].logical_device, spv_path)
		pipelines[Symbol(name)] = pip
		descriptor_pools[Symbol(name)] = pool
		descriptor_sets[Symbol(name)] = desc
	end

	println("Vulkan initialized with GPU: ", devices[].name)
end

function init_pipeline(device::VkDevice, spv_path::String; num_buffers::Int64 = 3, with_push_const::Bool=true)
    shaderModule = create_shader_module(device, spv_path)
	setLayout    = create_descriptior_set_layout(device, create_bindings(num_buffers))

    push_const = VkPushConstantRange(
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(Int32)
    )

    pip_lay_info = VkPipelineLayoutCreateInfo(
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        C_NULL,
        0,
        1, pointer([setLayout]),
        with_push_const ? 1 : 0,
	with_push_const ? pointer([push_const]) : C_NULL
    )

    pipelineLayout = Ref{VkPipelineLayout}()
    err = vkCreatePipelineLayout(device, Ref(pip_lay_info), C_NULL, pipelineLayout)
    @assert err == VK_SUCCESS

    pipeline = create_pipeline(device, pipelineLayout[], shaderModule, "main")

    pool     = create_descriptior_pool(device)
    desc_set = allocate_descriptor_set(device, pool, setLayout)

    return PipelineInfo(device, pipelineLayout[], pipeline), pool, desc_set
end

end
