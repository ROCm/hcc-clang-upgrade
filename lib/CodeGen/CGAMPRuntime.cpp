//===----- CGAMPRuntime.cpp - Interface to C++ AMP Runtime ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for C++ AMP code generation.  Concrete
// subclasses of this implement code generation for specific C++ AMP
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGAMPRuntime.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "CGCall.h"
#include "TargetInfo.h"

namespace clang {
namespace CodeGen {

CGAMPRuntime::~CGAMPRuntime() {}

/// Creates an instance of a C++ AMP runtime class.
CGAMPRuntime *CreateAMPRuntime(CodeGenModule &CGM) {
  return new CGAMPRuntime(CGM);
}
} // namespace CodeGen
} // namespace clang
