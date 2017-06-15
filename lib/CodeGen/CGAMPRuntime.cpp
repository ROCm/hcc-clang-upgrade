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

#include <algorithm> // For std::find_if.
#include <utility>   // For std::make_pair and std::pair.

namespace clang {
namespace CodeGen {

CGAMPRuntime::~CGAMPRuntime() {}

/// Creates an instance of a C++ AMP runtime class.
CGAMPRuntime *CreateAMPRuntime(CodeGenModule &CGM) {
  return new CGAMPRuntime(CGM);
}
static CXXMethodDecl *findValidIndexType(QualType IndexTy) {
  CXXRecordDecl *IndexClass = (*IndexTy).getAsCXXRecordDecl();
  CXXMethodDecl *IndexConstructor = NULL;
  if (IndexClass) {
    for (CXXRecordDecl::method_iterator CtorIt = IndexClass->method_begin(),
        CtorE = IndexClass->method_end();
        CtorIt != CtorE; ++CtorIt) {
      if (CtorIt->hasAttr<AnnotateAttr>() &&
          CtorIt->getAttr<AnnotateAttr>()->getAnnotation() ==
            "__cxxamp_opencl_index") {
        IndexConstructor = *CtorIt;
      }
    }
  }
  return IndexConstructor;
}

static
std::pair<CXXMethodDecl*,
          CXXMethodDecl*> find_lambda_kernel_decl_index_ctor(CXXMethodDecl* c)
{
    std::pair<CXXMethodDecl*, CXXMethodDecl*> r{nullptr, nullptr};

    if (c->hasAttr<CXXAMPRestrictAMPAttr>() && c->getNumParams() == 1) {
        if (c->getTemplatedKind() ==
            FunctionDecl::TemplatedKind::TK_FunctionTemplate) {
            const auto fl = c->getDescribedFunctionTemplate()->specializations();
            const auto it = std::find_if(fl.begin(),
                                         fl.end(),
                                         [&](decltype(*fl.begin()) x) {
                return r.second = findValidIndexType(x->getParamDecl(0)
                                                      ->getType()
                                                      .getNonReferenceType());
            });

            r.first = (it != fl.end()) ? cast<CXXMethodDecl>(*it) : nullptr;
        }
        else {
            r = std::make_pair(c, findValidIndexType(c->getParamDecl(0)
                                                      ->getType()
                                                      .getNonReferenceType()));
        }
    }

    return r;
}

static
std::pair<CXXMethodDecl*,
          CXXMethodDecl*> find_functor_kernel_decl_index_ctor(const CXXRecordDecl* c)
{
    std::pair<CXXMethodDecl*, CXXMethodDecl*> r{nullptr, nullptr};

    const auto it = std::find_if(c->method_begin(),
                                 c->method_end(),
                                 [&](decltype(*c->method_begin()) x) {
        return x->isOverloadedOperator() &&
               x->getOverloadedOperator() == OO_Call &&
               x->hasAttr<CXXAMPRestrictAMPAttr>() &&
               x->getNumParams() == 1 &&
               (r.second = findValidIndexType(x->getParamDecl(0)
                                               ->getType()
                                               .getNonReferenceType()));
    });

    r.first = (it != c->method_end()) ? *it : nullptr;

    return r;
}

static
std::pair<CXXMethodDecl*,
          CXXMethodDecl*> find_kernel_decl_index_ctor(const CXXRecordDecl* ClassDecl)
{
    if (ClassDecl->isLambda()) {
        return find_lambda_kernel_decl_index_ctor(ClassDecl->getLambdaCallOperator());
    }
    else {
        return find_functor_kernel_decl_index_ctor(ClassDecl);
    }
}

/// Operations:
/// For each reference-typed members, construct temporary object
/// Invoke constructor of index
/// Invoke constructor of the class
/// Invoke operator(index)
void CGAMPRuntime::EmitTrampolineBody(CodeGenFunction &CGF,
                                      const FunctionDecl *Trampoline,
                                      FunctionArgList& Args)
{
  const CXXRecordDecl *ClassDecl = dyn_cast<CXXMethodDecl>(Trampoline)->getParent();
  assert(ClassDecl);
  // Allocate "this"
  Address ai = CGF.CreateMemTemp(QualType(ClassDecl->getTypeForDecl(),0));
  // Locate the constructor to call
  if(ClassDecl->getCXXAMPDeserializationConstructor()==NULL) {
    return;
  }
  CXXConstructorDecl *DeserializeConstructor =
    dyn_cast<CXXConstructorDecl>(
      ClassDecl->getCXXAMPDeserializationConstructor());
  assert(DeserializeConstructor);
  CallArgList DeserializerArgs;
  // this
  DeserializerArgs.add(RValue::get(ai.getPointer()),
    DeserializeConstructor ->getThisType(CGF.getContext()));
  // the rest of constructor args. Create temporary objects for references
  // on stack
  CXXConstructorDecl::param_iterator CPI = DeserializeConstructor->param_begin(),
    CPE = DeserializeConstructor->param_end();
  for (FunctionArgList::iterator I = Args.begin();
       I != Args.end() && CPI != CPE; ++CPI) {
    // Reference types are only allowed to have one level; i.e. no
    // class base {&int}; class foo { bar &base; };
    QualType MemberType = (*CPI)->getType().getNonReferenceType();
    if (MemberType != (*CPI)->getType()) {
      if (!CGM.getLangOpts().HSAExtension) {
        assert(MemberType.getTypePtr()->isClassType() == true &&
               "Only supporting taking reference of classes");
        CXXRecordDecl *MemberClass = MemberType.getTypePtr()->getAsCXXRecordDecl();
        CXXConstructorDecl *MemberDeserializer =
  	dyn_cast<CXXConstructorDecl>(
            MemberClass->getCXXAMPDeserializationConstructor());
        assert(MemberDeserializer);
        std::vector<Expr*>MemberArgDeclRefs;
        for (CXXMethodDecl::param_iterator MCPI = MemberDeserializer->param_begin(),
          MCPE = MemberDeserializer->param_end(); MCPI!=MCPE; ++MCPI, ++I) {
          Expr *ArgDeclRef = DeclRefExpr::Create(CGM.getContext(),
                                                 NestedNameSpecifierLoc(),
                                                 SourceLocation(),
                                                 const_cast<VarDecl *>(*I),
                                                 false,
                                                 SourceLocation(),
                                                 (*MCPI)->getType(), VK_RValue);
  	  MemberArgDeclRefs.push_back(ArgDeclRef);
        }
        // Allocate "this" for member referenced objects
        Address mai = CGF.CreateMemTemp(MemberType);
        // Emit code to call the deserializing constructor of temp objects
        CXXConstructExpr *CXXCE = CXXConstructExpr::Create(
          CGM.getContext(), MemberType,
          SourceLocation(),
          MemberDeserializer,
          false,
          MemberArgDeclRefs,
          false, false, false, false,
          CXXConstructExpr::CK_Complete,
          SourceLocation());
        CGF.EmitCXXConstructorCall(MemberDeserializer,
          Ctor_Complete, false, false, mai, CXXCE);
        DeserializerArgs.add(RValue::get(mai.getPointer()), (*CPI)->getType());
      } else { // HSA extension check
        if (MemberType.getTypePtr()->isClassType()) {
          // hc::array should still be serialized as traditional C++AMP objects
          if (MemberType.getTypePtr()->isGPUArrayType()) {
            CXXRecordDecl *MemberClass = MemberType.getTypePtr()->getAsCXXRecordDecl();
            CXXConstructorDecl *MemberDeserializer =
            dyn_cast<CXXConstructorDecl>(
                MemberClass->getCXXAMPDeserializationConstructor());
            assert(MemberDeserializer);
            std::vector<Expr*>MemberArgDeclRefs;
            for (CXXMethodDecl::param_iterator MCPI = MemberDeserializer->param_begin(),
              MCPE = MemberDeserializer->param_end(); MCPI!=MCPE; ++MCPI, ++I) {
              Expr *ArgDeclRef = DeclRefExpr::Create(CGM.getContext(),
                                                     NestedNameSpecifierLoc(),
                                                     SourceLocation(),
                                                     const_cast<VarDecl *>(*I),
                                                     false,
                                                     SourceLocation(),
                                                     (*MCPI)->getType(), VK_RValue);
               MemberArgDeclRefs.push_back(ArgDeclRef);
            }
            // Allocate "this" for member referenced objects
            Address mai = CGF.CreateMemTemp(MemberType);
            // Emit code to call the deserializing constructor of temp objects
            CXXConstructExpr *CXXCE = CXXConstructExpr::Create(
              CGM.getContext(), MemberType,
              SourceLocation(),
              MemberDeserializer,
              false,
              MemberArgDeclRefs,
              false, false, false, false,
              CXXConstructExpr::CK_Complete,
              SourceLocation());
            CGF.EmitCXXConstructorCall(MemberDeserializer,
              Ctor_Complete, false, false, mai, CXXCE);
            DeserializerArgs.add(RValue::get(mai.getPointer()), (*CPI)->getType());
          } else {
            // capture by refernce for HSA
            Expr *ArgDeclRef = DeclRefExpr::Create(CGM.getContext(),
                                                   NestedNameSpecifierLoc(),
                                                   SourceLocation(),
                                                   const_cast<VarDecl *>(*I), false,
                                                   SourceLocation(),
                                                   (*I)->getType(), VK_RValue);
            RValue ArgRV = CGF.EmitAnyExpr(ArgDeclRef);
            DeserializerArgs.add(ArgRV, CGM.getContext().getPointerType(MemberType));
            ++I;
          }
        } else {
          // capture by refernce for HSA
          Expr *ArgDeclRef = DeclRefExpr::Create(CGM.getContext(),
              NestedNameSpecifierLoc(),
              SourceLocation(),
              const_cast<VarDecl *>(*I), false,
              SourceLocation(),
              (*I)->getType(), VK_RValue);
          RValue ArgRV = CGF.EmitAnyExpr(ArgDeclRef);
          DeserializerArgs.add(ArgRV, CGM.getContext().getPointerType(MemberType));
          ++I;
        }
      } // HSA extension check
    } else {
      Expr *ArgDeclRef = DeclRefExpr::Create(CGM.getContext(),
	  NestedNameSpecifierLoc(),
	  SourceLocation(),
	  const_cast<VarDecl *>(*I), false,
	  SourceLocation(),
	  (*I)->getType(), VK_RValue);
      RValue ArgRV = CGF.EmitAnyExpr(ArgDeclRef);
      DeserializerArgs.add(ArgRV, (*CPI)->getType());
      ++I;
    }
  }
  // Emit code to call the deserializing constructor
  {
    llvm::Constant *Callee = CGM.getAddrOfCXXStructor(
      DeserializeConstructor, StructorType::Complete);
    const FunctionProtoType *FPT =
      DeserializeConstructor->getType()->castAs<FunctionProtoType>();
    const CGFunctionInfo &DesFnInfo =
      CGM.getTypes().arrangeCXXStructorDeclaration(
          DeserializeConstructor, StructorType::Complete);
    for (unsigned I = 1, E = DeserializerArgs.size(); I != E; ++I) {
      auto T = FPT->getParamType(I-1);
      // EmitFromMemory is necessary in case function has bool parameter.
      if (T->isBooleanType()) {
        DeserializerArgs[I] = CallArg(RValue::get(
            CGF.EmitFromMemory(DeserializerArgs[I].RV.getScalarVal(), T)),
            T, false);
      }
    }
    CGF.EmitCall(DesFnInfo, CGCallee::forDirect(Callee), ReturnValueSlot(), DeserializerArgs);
  }

  // Locate the type of Concurrency::index<1>
  // Locate the operator to call
  const auto kernel_decl_index_ctor = find_kernel_decl_index_ctor(ClassDecl);

  // in case we couldn't find any kernel declarator
  // raise error
  if (!kernel_decl_index_ctor.first || !kernel_decl_index_ctor.second) {
    CGF.CGM.getDiags().Report(ClassDecl->getLocation(),
                              diag::err_amp_ill_formed_functor);
    return;
  }

  // Allocate Index
  Address index = CGF.CreateMemTemp(kernel_decl_index_ctor.first
                                                          ->getParamDecl(0)
                                                          ->getType()
                                                          .getNonReferenceType());

  // Emit code to call the Concurrency::index<1>::__cxxamp_opencl_index()
  if (!CGF.getLangOpts().AMPCPU) {
    if (CXXConstructorDecl *Constructor =
        dyn_cast<CXXConstructorDecl>(kernel_decl_index_ctor.second)) {
      CXXConstructExpr *CXXCE = CXXConstructExpr::Create(
        CGM.getContext(),
        kernel_decl_index_ctor.first
                              ->getParamDecl(0)
                              ->getType()
                              .getNonReferenceType(),
        SourceLocation(),
        Constructor,
        false,
        ArrayRef<Expr*>(),
        false, false, false, false,
        CXXConstructExpr::CK_Complete,
        SourceLocation());
      CGF.EmitCXXConstructorCall(Constructor,
        Ctor_Complete, false, false, index, CXXCE);
    } else {
      llvm::FunctionType* indexInitType =
              CGM.getTypes().GetFunctionType(
          CGM.getTypes()
             .arrangeCXXMethodDeclaration(kernel_decl_index_ctor.second));
      llvm::Value *indexInitAddr = CGM.GetAddrOfFunction(
        kernel_decl_index_ctor.second,
        indexInitType);

      CGF.EmitCXXMemberOrOperatorCall(kernel_decl_index_ctor.second,
                                      indexInitAddr,
                                      ReturnValueSlot(),
                                      index.getPointer(),
                                      /*ImplicitParam=*/0,
                                      QualType(),
                                      /*CallExpr=*/nullptr,
                                      /*RtlArgs=*/nullptr);
    }
  }
  // Invoke this->operator(index)
  // Prepate the operator() to call
  llvm::FunctionType* fnType =
    CGM.getTypes()
       .GetFunctionType(CGM.getTypes()
       .arrangeCXXMethodDeclaration(kernel_decl_index_ctor.first));
  llvm::Value *fnAddr = CGM.GetAddrOfFunction(kernel_decl_index_ctor.first,
                                              fnType);
  // Prepare argument
  CallArgList KArgs;
  // this
  KArgs.add(RValue::get(ai.getPointer()),
            kernel_decl_index_ctor.first->getThisType(CGF.getContext()));
  // *index
  KArgs.add(RValue::getAggregate(index),
            kernel_decl_index_ctor.first
                                  ->getParamDecl(0)
                                  ->getType()
                                  .getNonReferenceType());

  const auto MT = dyn_cast<FunctionType>(kernel_decl_index_ctor.first
                                                               ->getType()
                                                               .getTypePtr());
  assert(MT);

  const CGFunctionInfo& FnInfo = CGM.getTypes().arrangeFreeFunctionCall(KArgs,
                                                                        MT,
                                                                        false);
  CGF.EmitCall(FnInfo, fnAddr, ReturnValueSlot(), KArgs);
}

void CGAMPRuntime::EmitTrampolineNameBody(CodeGenFunction &CGF,
  const FunctionDecl *Trampoline, FunctionArgList& Args) {
  const CXXRecordDecl *ClassDecl = dyn_cast<CXXMethodDecl>(Trampoline)->getParent();
  assert(ClassDecl);
  // Locate the trampoline
  // Locate the operator to call
  CXXMethodDecl *TrampolineDecl = NULL;
  for (CXXRecordDecl::method_iterator Method = ClassDecl->method_begin(),
      MethodEnd = ClassDecl->method_end();
      Method != MethodEnd; ++Method) {
    CXXMethodDecl *MethodDecl = *Method;
    if (Method->hasAttr<AnnotateAttr>() &&
        Method->getAttr<AnnotateAttr>()->getAnnotation() == "__cxxamp_trampoline") {
      TrampolineDecl = MethodDecl;
      break;
    }
  }
  assert(TrampolineDecl && "Trampoline not declared!");
  GlobalDecl GD(TrampolineDecl);
  llvm::Constant *S = llvm::ConstantDataArray::getString(CGM.getLLVMContext(),
    CGM.getMangledName(GD));
  llvm::GlobalVariable *GV = new llvm::GlobalVariable(CGM.getModule(), S->getType(),
    true, llvm::GlobalValue::PrivateLinkage, S, "__cxxamp_trampoline.kernelname");
  GV->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

  //Create GetElementPtr(0, 0)
  std::vector<llvm::Constant*> indices;
  llvm::ConstantInt *zero = llvm::ConstantInt::get(CGM.getLLVMContext(), llvm::APInt(32, StringRef("0"), 10));
  indices.push_back(zero);
  indices.push_back(zero);
  llvm::Constant *const_ptr = llvm::ConstantExpr::getGetElementPtr(GV->getValueType(), GV, indices);
  CGF.Builder.CreateStore(const_ptr, CGF.ReturnValue);

}
} // namespace CodeGen
} // namespace clang
