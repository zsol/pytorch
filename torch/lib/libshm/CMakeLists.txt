project(libshm LANGUAGES CXX)
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(TORCH_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../)

add_executable(torch_shm_manager manager.cpp core.cpp)
set_target_properties(torch_shm_manager PROPERTIES
  CXX_STANDARD 17)

target_include_directories(torch_shm_manager PUBLIC
  ${TORCH_ROOT}/torch/lib # provides "libshm/libshm.h"
)

### Torch packages supposes libraries prefix is "lib"
set_target_properties(shm PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib"
  CXX_STANDARD 17)
target_link_libraries(shm PUBLIC torch)

set_target_properties(torch_shm_manager PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/../lib")

install(TARGETS torch_shm_manager DESTINATION "bin")
