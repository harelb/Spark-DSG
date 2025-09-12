add_executable(dsg_repeater dsg_repeater.cpp)
target_link_libraries(dsg_repeater ${PROJECT_NAME})

add_executable(dsg_endpoint dsg_endpoint.cpp)
target_link_libraries(dsg_endpoint ${PROJECT_NAME})

add_executable(dsg_migrator dsg_migrator.cpp)
target_link_libraries(dsg_migrator ${PROJECT_NAME})

install(TARGETS dsg_repeater dsg_endpoint dsg_migrator RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
