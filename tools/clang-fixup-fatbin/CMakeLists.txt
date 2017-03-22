set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_executable(clang-fixup-fatbin
  ClangFixupFatbin.cpp
  )

install(TARGETS clang-fixup-fatbin
  RUNTIME DESTINATION bin)
