include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-decision.cpp
    server-decision.h
    server-context.cpp
    server-context.h
    server-realtime.cpp
    server-realtime.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-mcp.cpp
    server-mcp.h
    server-schema.cpp
    server-schema.h
    server-slot-sched.cpp
    server-slot-sched.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${TARGET} PRIVATE vendor::hash)

# Optional NVTX v3 markers for NEPENTHES request correlation in CUDA builds.
# NVTX v3 is header-only; CPU-only builds do not acquire a CUDA/NVTX dependency.
if (GGML_CUDA)
    find_path(NEPENTHES_NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        /usr/local/cuda/include)
    if (NEPENTHES_NVTX_INCLUDE_DIR)
        target_include_directories(server-context PRIVATE ${NEPENTHES_NVTX_INCLUDE_DIR})
        target_compile_definitions(server-context PRIVATE NEPENTHES_USE_NVTX=1)
        if (CMAKE_DL_LIBS)
            target_link_libraries(server-context PRIVATE ${CMAKE_DL_LIBS})
        endif()
        message(STATUS "NEPENTHES: NVTX request markers enabled")
    else()
        message(STATUS "NEPENTHES: NVTX v3 headers not found; request markers disabled")
    endif()
endif()

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-connect.cpp
    server-connect.h
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
    server-models-sched.cpp
    server-models-sched.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

add_dependencies(${TARGET} llama-ui-assets)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
