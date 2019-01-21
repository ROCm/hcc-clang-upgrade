//===----- CGAMPRuntime.h - Interface to C++ AMP Runtime --------*- C++ -*-===//
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

#ifndef CLANG_CODEGEN_AMPRUNTIME_H
#define CLANG_CODEGEN_AMPRUNTIME_H

namespace clang {

namespace CodeGen {

class CodeGenFunction;
class CodeGenModule;
class FunctionArgList;
class ReturnValueSlot;
class RValue;

class CGHCRuntime {
protected: // TODO: Who would ever inherit from this?
  CodeGenModule &CGM;

public:
  CGHCRuntime(CodeGenModule &CGM) : CGM(CGM) {}
  virtual ~CGHCRuntime() = default;
};

/// Creates an instance of a C++ AMP runtime class.
CGHCRuntime *CreateHCRuntime(CodeGenModule &CGM);

}
}

#endif
