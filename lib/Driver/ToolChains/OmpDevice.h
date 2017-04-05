//===--- OmpDevice.h - OpenMP Device ToolChain Implementations --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_OMPDEVICE_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_OMPDEVICE_H

#include "Cuda.h"
#include "clang/Basic/VersionTuple.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Multilib.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Tool.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Compiler.h"
#include <set>
#include <vector>

namespace clang {
namespace driver {

namespace tools {
namespace OMPDEV {

// for amdgcn the backend is llvm-link + opt
class LLVM_LIBRARY_VISIBILITY Backend : public Tool {
  public:
    Backend(const ToolChain &TC)
      : Tool("OMPDEV::Backend", "GPU-backend", TC, RF_Full, llvm::sys::WEM_UTF8,
              "--options-file") {}
    virtual bool hasIntegratedCPP() const override { return false; }
    //virtual bool isLinkJob() const { return false; }
    virtual void ConstructJob(Compilation &C, const JobAction &JA,
                              const InputInfo &Output,
                              const InputInfoList &Inputs,
                              const llvm::opt::ArgList &TCArgs,
                              const char *LinkingOutput) const override;
};

// Run ptxas, the OMPDEV assembler.
class LLVM_LIBRARY_VISIBILITY Assembler : public Tool {
 public:
   Assembler(const ToolChain &TC)
       : Tool("OMPDEV::Assembler", "GPU-assembler", TC, RF_Full, llvm::sys::WEM_UTF8,
              "--options-file") {}

   bool hasIntegratedCPP() const override { return false; }

   void ConstructJob(Compilation &C, const JobAction &JA,
                     const InputInfo &Output, const InputInfoList &Inputs,
                     const llvm::opt::ArgList &TCArgs,
                     const char *LinkingOutput) const override;
};

// Runs fatbinary, which combines GPU object files ("cubin" files) and/or PTX
// assembly into a single output file.
class LLVM_LIBRARY_VISIBILITY Linker : public Tool {
 public:
   Linker(const ToolChain &TC)
       : Tool("OMPDEV::Linker", "GPU-linker", TC, RF_Full, llvm::sys::WEM_UTF8,
              "--options-file") {}

   bool hasIntegratedCPP() const override { return false; }

   void ConstructJob(Compilation &C, const JobAction &JA,
                     const InputInfo &Output, const InputInfoList &Inputs,
                     const llvm::opt::ArgList &TCArgs,
                     const char *LinkingOutput) const override;
};

} // end namespace OMPDEV
} // end namespace tools

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY OmpDeviceToolChain : public ToolChain {
public:
  OmpDeviceToolChain(const Driver &D, const llvm::Triple &Triple,
                const ToolChain &HostTC, const llvm::opt::ArgList &Args);

  virtual const llvm::Triple *getAuxTriple() const override {
    return &HostTC.getTriple();
  }

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
                Action::OffloadKind DeviceOffloadKind) const override;
  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                             llvm::opt::ArgStringList &CC1Args) const override;

  // Never try to use the integrated assembler with CUDA; always fork out to
  // ptxas.
  bool useIntegratedAs() const override { return false; }
  bool isCrossCompiling() const override { return true; }
  bool isPICDefault() const override { return false; }
  bool isPIEDefault() const override { return false; }
  bool isPICDefaultForced() const override { return false; }
  bool SupportsProfiling() const override { return false; }
  bool SupportsObjCGC() const override { return false; }
  void AddCudaIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                          llvm::opt::ArgStringList &CC1Args) const override;
  const ToolChain &HostTC;
  CudaInstallationDetector CudaInstallation;

protected:
  Tool *buildBackend() const override;    // for amdgcn, link and opt
  Tool *buildAssembler() const override;  // ptxas
  Tool *buildLinker() const override;     // fatbinary (ok, not really a linker)
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_OMPDEVICE_H
