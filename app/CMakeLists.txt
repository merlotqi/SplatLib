if(UNIX)
    find_package(PkgConfig REQUIRED)
    find_package(Vulkan REQUIRED)
    pkg_check_modules(ABSL_FLAGS REQUIRED absl_flags_parse absl_flags_usage absl_flags_marshalling absl_flags_program_name absl_flags_reflection absl_flags_commandlineflag)
endif()

add_executable(SplatTransform main.cpp process.h process.cpp gpudevice.h)
target_link_libraries(SplatTransform PRIVATE SPLAT::splat)

if(UNIX)

    if(Vulkan_FOUND)
        if(TARGET Vulkan::Vulkan)
            target_link_libraries(SplatTransform PRIVATE Vulkan::Vulkan)
        else()
            target_include_directories(SplatTransform PRIVATE ${Vulkan_INCLUDE_DIRS})
            target_link_libraries(SplatTransform PRIVATE ${Vulkan_LIBRARIES})
        endif()
    else()
        message(FATAL_ERROR "Vulkan not found! On Ubuntu/Debian, run: sudo apt install libvulkan-dev")
    endif()

    target_link_libraries(SplatTransform PRIVATE 
        absl_flags_parse
        absl_flags_usage
        absl_flags_reflection
        absl_flags_commandlineflag
        absl_flags_marshalling
        absl_flags_internal
        absl_flags_program_name
        absl_flags_config
        absl_raw_logging_internal
    )

elseif(WIN32)

    target_link_libraries(SplatTransform PRIVATE 
        absl::flags_parse 
        absl::flags
    )
endif()


