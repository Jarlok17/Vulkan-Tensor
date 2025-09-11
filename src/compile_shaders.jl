function compile_shaders()
    current_dir = @__DIR__
    shaders_dir = joinpath(current_dir, "shaders")
    
    if !isdir(shaders_dir)
        error("Shaders directory not found: $shaders_dir")
    end
    
    comp_files = filter(f -> endswith(f, ".comp"), readdir(shaders_dir))
    
    if isempty(comp_files)
        println("No .comp files found in $shaders_dir")
        return
    end
    
    for comp_file in comp_files
        comp_path = joinpath(shaders_dir, comp_file)
        spv_path = replace(comp_path, ".comp" => ".spv")
        
        try
            run(`glslc $comp_path -o $spv_path`)
            println("✓ Compiled: $comp_file -> $(basename(spv_path))")
        catch e
            println("✗ Failed to compile $comp_file: $e")
        end
    end
end

function check_glslc()
    try
        success(`glslc --version`)
        return true
    catch
        return false
    end
end

if !check_glslc()
    error("glslc not found. Please install Vulkan SDK.")
end

compile_shaders()
