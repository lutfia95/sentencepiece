add_executable(sentencepiece_cpp_all_use_cases
  cpp/all_use_cases.cc)

target_include_directories(sentencepiece_cpp_all_use_cases PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR})

target_link_libraries(sentencepiece_cpp_all_use_cases
  sentencepiece
  sentencepiece_train)

target_compile_definitions(sentencepiece_cpp_all_use_cases PRIVATE
  SENTENCEPIECE_REPO_ROOT="${PROJECT_SOURCE_DIR}")
