//===----- CGAMPRuntime.cpp - Interface to C++ AMP Runtime ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for HC code generation. Concrete subclasses
// of this implement code generation for specific HC runtime libraries.
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

/// Creates an instance of a HC runtime class.
CGHCRuntime *CreateHCRuntime(CodeGenModule &CGM) {
  return new CGHCRuntime(CGM);
}
} // namespace CodeGen
} // namespace clang
