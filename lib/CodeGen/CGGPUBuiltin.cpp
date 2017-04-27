//===------ CGGPUBuiltin.cpp - Codegen for GPU builtins -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Generates code for built-in GPU calls which are not runtime-specific.
// (Runtime-specific codegen lives in programming model specific files.)
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/Basic/Builtins.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/MathExtras.h"

using namespace clang;
using namespace CodeGen;

static llvm::Function *GetVprintfDeclaration(CodeGenModule &CGM) {
  auto &M = CGM.getModule();
  llvm::Type *ArgTypes[] = {CGM.Int8PtrTy, CGM.Int8PtrTy};
  llvm::FunctionType *VprintfFuncType = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(M.getContext()), ArgTypes, false);

  if (auto* F = M.getFunction("vprintf")) {
    // Our CUDA system header declares vprintf with the right signature, so
    // nobody else should have been able to declare vprintf with a bogus
    // signature.
    assert(F->getFunctionType() == VprintfFuncType);
    return F;
  }

  // vprintf doesn't already exist; create a declaration and insert it into the
  // module.
  return llvm::Function::Create(
      VprintfFuncType, llvm::GlobalVariable::ExternalLinkage, "vprintf", &M);
}

static llvm::Function *GetVprintflDeclaration(CodeGenModule &CGM) {
  auto &M = CGM.getModule();
  llvm::Type *ArgTypes[] = 
    {CGM.Int8PtrTy, CGM.Int8PtrTy , CGM.Int32Ty, CGM.Int32Ty};
  llvm::FunctionType *VprintflFuncType = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(M.getContext()), ArgTypes, false);

  if (auto* F = M.getFunction("vprintfl")) {
    assert(F->getFunctionType() == VprintflFuncType);
    return F;
  }
  return llvm::Function::Create(
      VprintflFuncType, llvm::GlobalVariable::ExternalLinkage, "vprintfl", &M);
}


// Transforms a call to printf into a call to the NVPTX vprintf syscall or
// AMDGCN device routine vprintfl. Neither are particularly special.
// They are invoked just like a regular function).
// vprintf takes two args: A format string, and a pointer to a buffer containing
// the varargs.
//
// For example, the call
//
//   printf("format string", arg1, arg2, arg3);
//
// is converted into something resembling
//
//   struct Tmp {
//     Arg1 a1;
//     Arg2 a2;
//     Arg3 a3;
//   };
//   char* buf = alloca(sizeof(Tmp));
//   *(Tmp*)buf = {a1, a2, a3};
//   vprintf("format string", buf);
//
// buf is aligned to the max of {alignof(Arg1), ...}.  Furthermore, each of the
// args is itself aligned to its preferred alignment.
//
// Note that by the time this function runs, E's args have already undergone the
// standard C vararg promotion (short -> int, float -> double, etc.).
//
// vprintfl is same as vprintf but with length of both structures provided
// as compile time constants.  This makes it easier to construct a simple GPU
// device library functon vprintfl to move data to host runtime for proxy printf.
RValue
CodeGenFunction::EmitNVPTXDevicePrintfCallExpr(const CallExpr *E,
                                               ReturnValueSlot ReturnValue) {
  assert(getLangOpts().CUDAIsDevice || getLangOpts().OpenMPIsDevice);
  assert(E->getBuiltinCallee() == Builtin::BIprintf);
  assert(E->getNumArgs() >= 1); // printf always has at least one arg.

  bool isGCN = (CGM.getTriple().getArch() == llvm::Triple::amdgcn)?true:false;

  const llvm::DataLayout &DL = CGM.getDataLayout();

  CallArgList Args;
  EmitCallArgs(Args,
               E->getDirectCallee()->getType()->getAs<FunctionProtoType>(),
               E->arguments(), E->getDirectCallee(),
               /* ParamsToSkip = */ 0);

  // We don't know how to emit non-scalar varargs.
  if (std::any_of(Args.begin() + 1, Args.end(),
                  [](const CallArg &A) { return !A.RV.isScalar(); })) {
    CGM.ErrorUnsupported(E, "non-scalar arg to printf");
    return RValue::get(llvm::ConstantInt::get(IntTy, 0));
  }

  // Construct and fill the args buffer that we'll pass to vprintf.
  llvm::Value *BufferPtr;
  llvm::Value *BufLen;
  if (Args.size() <= 1) {
    // If there are no args, pass a null pointer to vprintf.
    BufferPtr = llvm::ConstantPointerNull::get(CGM.Int8PtrTy);
    BufLen    = llvm::ConstantInt::get(Int32Ty, 0);
  } else {
    llvm::SmallVector<llvm::Type *, 8> ArgTypes;
    for (unsigned I = 1, NumArgs = Args.size(); I < NumArgs; ++I) {
      llvm::Type* ArgType = Args[I].RV.getScalarVal()->getType();
      ArgTypes.push_back(ArgType);

/*
    FIXME:  Strings are not packing yet, need more work here
      // Look for possible string in the args
      if(llvm::PointerType* Pty = dyn_cast<llvm::PointerType>(ArgType)) {
        if(llvm::IntegerType* Ity = dyn_cast<llvm::IntegerType>(Pty->getElementType())) {
          if(Ity->getBitWidth()==8)  {
            llvm::Value* V = Args[I].RV.getScalarVal();
            const Expr * FormatStringExpr = E->getArg(I)->IgnoreParenCasts();
            StringRef FormatString = cast<StringLiteral>(FormatStringExpr)->getString();
            printf("WARNING Type %d is pointer to int , width:%d size:%d VID:%d V:%p\n",I,
              Ity->getBitWidth(),
              (int)FormatString.size()+1,
              V->getValueID(), (void*) V
            );
            if(llvm::User * USR = dyn_cast<llvm::User>(V)){
              llvm::Value* OV = USR->getOperand(0);
              printf("Is User %p NumO:%d OV:%p \n" ,
               (void*) USR , USR->getNumOperands(),(void*) OV);
              if(llvm::Constant* CDS = dyn_cast<llvm::Constant>(OV)){
                 printf("Is Constant %p \n",(void*) CDS);
              }
            }
          }
        }
      }
*/
    }

    // Using llvm::StructType is correct only because printf doesn't accept
    // aggregates.  If we had to handle aggregates here, we'd have to manually
    // compute the offsets within the alloca -- we wouldn't be able to assume
    // that the alignment of the llvm type was the same as the alignment of the
    // clang type.
    llvm::StructType *AllocaTy = llvm::StructType::create(ArgTypes, "printf_args");
    llvm::Value* Alloca ;
    if (isGCN)  {
      // Until we have __private alloca, we use a global struct
      unsigned int AS = getContext().getTargetAddressSpace(LangAS::cuda_device);
      llvm::PointerType* Pty = llvm::PointerType::get(AllocaTy,AS);
      llvm::GlobalVariable* GV = new llvm::GlobalVariable( CGM.getModule(),
        Pty, false, llvm::GlobalValue::PrivateLinkage,
        llvm::ConstantPointerNull::get(Pty), AllocaTy->getName(), nullptr,
        llvm::GlobalVariable::NotThreadLocal, AS);
      GV->setAlignment(8);
      Alloca = GV;
    } else
      Alloca = CreateTempAlloca(AllocaTy);

    for (unsigned I = 1, NumArgs = Args.size(); I < NumArgs; ++I) {
      llvm::Value *P = Builder.CreateStructGEP(AllocaTy, Alloca, I - 1);
      llvm::Value *Arg = Args[I].RV.getScalarVal();
      Builder.CreateAlignedStore(Arg, P, DL.getPrefTypeAlignment(Arg->getType()));
    }
    BufferPtr = Builder.CreatePointerCast(Alloca, CGM.Int8PtrTy);
    BufLen    = llvm::ConstantInt::get(Int32Ty,
      (int) DL.getTypeAllocSize(AllocaTy));
  }

  // Invoke vprintfl for amdgcn or vprintf for CUDA and return.
  if (isGCN) {
    const Expr * FormatStringExpr = E->getArg(0)->IgnoreParenCasts();
    StringRef FormatString = cast<StringLiteral>(FormatStringExpr)->getString();
    llvm::Value *FmtStrLen = llvm::ConstantInt::get(Int32Ty,
      (int)FormatString.size()+1);
    llvm::Function* VprintflFunc = GetVprintflDeclaration(CGM);
    return RValue::get(
      Builder.CreateCall(VprintflFunc, {Args[0].RV.getScalarVal(), BufferPtr,
        FmtStrLen, BufLen}));
  } else {
    llvm::Function* VprintfFunc = GetVprintfDeclaration(CGM);
    return RValue::get(
      Builder.CreateCall(VprintfFunc, {Args[0].RV.getScalarVal(), BufferPtr}));
  }
}
