//===-- OmpDevice.cpp - OpenMP Device ToolChain Implementations -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OmpDevice.h"
#include "CommonArgs.h"
#include "InputInfo.h"
#include "clang/Basic/Cuda.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"
#include <system_error>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

void OMPDEV::Backend::ConstructJob(Compilation &C, const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {

  assert(StringRef(JA.getOffloadingArch()).startswith("gfx") &&
    " unless gfx processor, backend should be clang") ;

  // For amdgcn the Backend Job will call llvm-link & opt steps
  ArgStringList CmdArgs;
  // Add the input bc's created by compile step
  for (InputInfoList::const_iterator
       it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
    const InputInfo &II = *it;
    CmdArgs.push_back(II.getFilename());
  }

  std::string GFXNAME = JA.getOffloadingArch();

  // Find in -L<path> and LIBRARY_PATH.
  ArgStringList LibraryPaths;
  for (auto Arg : Args) {
    if (Arg->getSpelling() == "-L") {
      std::string Current = "-L";
      Current += Arg->getValue();
      LibraryPaths.push_back(Args.MakeArgString(Current.c_str()));
    }
  }
  addDirectoryList(Args, LibraryPaths, "-L", "LIBRARY_PATH");

  const char * libamdgcn;
  libamdgcn = getenv("LIBAMDGCN");
  if (!libamdgcn) libamdgcn = "/opt/rocm/libamdgcn";
  LibraryPaths.push_back(Args.MakeArgString(
    "-L" + std::string(libamdgcn) + "/" + std::string(GFXNAME)  + "/lib"));

  //  Remove hcc2 search when libcuda2gcn.bc moves to LIBAMDGCN
  const char * hcc2;
  hcc2 = getenv("HCC2");
  if (!hcc2) hcc2 = "/opt/rocm/hcc2";
  LibraryPaths.push_back(Args.MakeArgString( "-L" + std::string(hcc2) + "/lib"));

  // As of now based on AMD ROCm 1.4.0
  addBCLib(C, Args, CmdArgs, LibraryPaths, "libcuda2gcn.bc");
  addBCLib(C, Args, CmdArgs, LibraryPaths, "opencl.amdgcn.bc");
  addBCLib(C, Args, CmdArgs, LibraryPaths, "ockl.amdgcn.bc");
  addBCLib(C, Args, CmdArgs, LibraryPaths, "irif.amdgcn.bc");
  addBCLib(C, Args, CmdArgs, LibraryPaths, "ocml.amdgcn.bc");
  addBCLib(C, Args, CmdArgs, LibraryPaths, "oclc_finite_only_off.amdgcn.bc");
  addBCLib(C, Args, CmdArgs, LibraryPaths, "oclc_daz_opt_off.amdgcn.bc");
  addBCLib(C, Args, CmdArgs, LibraryPaths,
    "oclc_correctly_rounded_sqrt_on.amdgcn.bc");
  addBCLib(C, Args, CmdArgs, LibraryPaths, "oclc_unsafe_math_off.amdgcn.bc");
  addBCLib(C, Args, CmdArgs, LibraryPaths, "hc.amdgcn.bc");
  addBCLib(C, Args, CmdArgs, LibraryPaths, "oclc_isa_version.amdgcn.bc");

  addEnvListWithSpaces(Args, CmdArgs, "CLANG_TARGET_LINK_OPTS");
  CmdArgs.push_back("-suppress-warnings");

  // Add an intermediate output file which is input to opt
  CmdArgs.push_back("-o");
  std::string TmpName = C.getDriver().GetTemporaryPath("OPT_INPUT", "bc");
  const char *ResultingBitcodeF = C.addTempFile(C.getArgs().MakeArgString(TmpName.c_str()));
  CmdArgs.push_back(ResultingBitcodeF);
  const char *Exec = Args.MakeArgString(C.getDriver().Dir + "/llvm-link");
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));

  ArgStringList OptArgs;
  // The input to opt is the output from llvm-link
  OptArgs.push_back(ResultingBitcodeF);
  // Add CLANG_TARGETOPT_OPTS override options to opt
  if (getenv("CLANG_TARGET_OPT_OPTS"))
    addEnvListWithSpaces(Args, OptArgs, "CLANG_TARGET_OPT_OPTS");
  else
    OptArgs.push_back(Args.MakeArgString("-O2"));

  OptArgs.push_back("-S");
  const char *mcpustr = Args.MakeArgString("-mcpu=" + GFXNAME);
  OptArgs.push_back(mcpustr);
  OptArgs.push_back("-infer-address-spaces");
  //OptArgs.push_back("-load");
  //OptArgs.push_back("LLVMSugarAddrSpaceCast.so");
  //OptArgs.push_back("-sugar-addrspacecast");
  OptArgs.push_back("-dce");
  OptArgs.push_back("-globaldce");
  OptArgs.push_back("-o");
  OptArgs.push_back(Output.getFilename());
  const char *OptExec = Args.MakeArgString(C.getDriver().Dir + "/opt");
  C.addCommand(llvm::make_unique<Command>(JA, *this, OptExec, OptArgs, Inputs ));

  if (Args.hasArg(options::OPT_v)) {
    ArgStringList nmArgs;
    nmArgs.push_back(ResultingBitcodeF);
    nmArgs.push_back("-debug-syms");
    const char *nmExec = Args.MakeArgString(C.getDriver().Dir + "/llvm-nm");
    C.addCommand(llvm::make_unique<Command>(JA, *this, nmExec, nmArgs, Inputs ));
  }
}

void OMPDEV::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::OmpDeviceToolChain &>(getToolChain());
  assert( (TC.getTriple().isOMPDEV() || TC.getTriple().isAMDGCN())
    && "Wrong platform");

  // Obtain architecture from the action.
  CudaArch gpu_arch = StringToCudaArch(JA.getOffloadingArch());
  assert(gpu_arch != CudaArch::UNKNOWN &&
         "Device action expected to have an architecture.");

  // For amdgcn this job will call llc  (Lightning Compiler)
  if (StringRef(JA.getOffloadingArch()).startswith("gfx")) {
    ArgStringList CmdArgs;
    for (InputInfoList::const_iterator
         it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
      const InputInfo &II = *it;
      CmdArgs.push_back(II.getFilename());
    }
    CmdArgs.push_back("-mtriple=amdgcn--cuda");
    CmdArgs.push_back("-filetype=obj");
    addEnvListWithSpaces(Args, CmdArgs, "CLANG_TARGET_LLC_OPTS");
    std::string GFXNAME = JA.getOffloadingArch();
    CmdArgs.push_back(Args.MakeArgString("-mcpu=" + GFXNAME));
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
    const char *Exec = Args.MakeArgString(C.getDriver().Dir + "/llc");
    C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
    return;
  }

  // Check that our installation's ptxas supports gpu_arch.
  if (!Args.hasArg(options::OPT_no_cuda_version_check)) {
    TC.CudaInstallation.CheckCudaVersionSupportsArch(gpu_arch);
  }

  ArgStringList CmdArgs;
  CmdArgs.push_back(TC.getTriple().isArch64Bit() ? "-m64" : "-m32");
  if (Args.hasFlag(options::OPT_cuda_noopt_device_debug,
                   options::OPT_no_cuda_noopt_device_debug, false)) {
    // ptxas does not accept -g option if optimization is enabled, so
    // we ignore the compiler's -O* options if we want debug info.
    CmdArgs.push_back("-g");
    CmdArgs.push_back("--dont-merge-basicblocks");
    CmdArgs.push_back("--return-at-end");
  } else if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    // Map the -O we received to -O{0,1,2,3}.
    //
    // TODO: Perhaps we should map host -O2 to ptxas -O3. -O3 is ptxas's
    // default, so it may correspond more closely to the spirit of clang -O2.

    // -O3 seems like the least-bad option when -Osomething is specified to
    // clang but it isn't handled below.
    StringRef OOpt = "3";
    if (A->getOption().matches(options::OPT_O4) ||
        A->getOption().matches(options::OPT_Ofast))
      OOpt = "3";
    else if (A->getOption().matches(options::OPT_O0))
      OOpt = "0";
    else if (A->getOption().matches(options::OPT_O)) {
      // -Os, -Oz, and -O(anything else) map to -O2, for lack of better options.
      OOpt = llvm::StringSwitch<const char *>(A->getValue())
                 .Case("1", "1")
                 .Case("2", "2")
                 .Case("3", "3")
                 .Case("s", "2")
                 .Case("z", "2")
                 .Default("2");
    }
    CmdArgs.push_back(Args.MakeArgString(llvm::Twine("-O") + OOpt));
  } else {
    // If no -O was passed, pass -O0 to ptxas -- no opt flag should correspond
    // to no optimizations, but ptxas's default is -O3.
    CmdArgs.push_back("-O0");
  }

  CmdArgs.push_back("--gpu-name");
  CmdArgs.push_back(Args.MakeArgString(CudaArchToString(gpu_arch)));
  CmdArgs.push_back("--output-file");
  CmdArgs.push_back(Args.MakeArgString(Output.getFilename()));
  for (const auto& II : Inputs)
    CmdArgs.push_back(Args.MakeArgString(II.getFilename()));

  for (const auto& A : Args.getAllArgValues(options::OPT_Xcuda_ptxas))
    CmdArgs.push_back(Args.MakeArgString(A));

  if(JA.isOffloading(Action::OFK_OpenMP))
    CmdArgs.push_back("-c");

  const char *Exec;
  if (Arg *A = Args.getLastArg(options::OPT_ptxas_path_EQ))
    Exec = A->getValue();
  else
    Exec = Args.MakeArgString(TC.GetProgramPath("ptxas"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void OMPDEV::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                 const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {

  // For amdgcn linker is lld 
  if (getToolChain().getArch() == llvm::Triple::amdgcn) {
    ArgStringList CmdArgs;

    CmdArgs.push_back("-flavor");
    CmdArgs.push_back("gnu");

    CmdArgs.push_back("--no-undefined");
    CmdArgs.push_back("-shared");

    // The output from ld.lld is an HSA code object file
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());

    // Add the input object files created by "Assemble" step
    for (InputInfoList::const_iterator
       it = Inputs.begin(), ie = Inputs.end(); it != ie; ++it) {
      const InputInfo &II = *it;
      if (II.isFilename()) {
        CmdArgs.push_back(II.getFilename());
        //StringRef Name = llvm::sys::path::filename(II.getFilename());
      }
    }

    const char * amdllvm;
    amdllvm = getenv("AMDLLVM");
    if (!amdllvm) amdllvm = "/opt/amd/llvm";
    const char *Exec = Args.MakeArgString(std::string(amdllvm) + "/bin/lld");
    C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));

  } else {

  const auto &TC =
      static_cast<const toolchains::OmpDeviceToolChain &>(getToolChain());
  assert(TC.getTriple().isOMPDEV() && "Wrong platform");

  ArgStringList CmdArgs;

  // OpenMP uses nvlink to link cubin files. The result will be embedded in the
  // host binary by the host linker.
  assert(!JA.isHostOffloading(Action::OFK_OpenMP) &&
         "CUDA toolchain not expected for an OpenMP host device.");
  if (JA.isDeviceOffloading(Action::OFK_OpenMP)) {
    if (Output.isFilename()) {
      CmdArgs.push_back("-o");
      CmdArgs.push_back(Output.getFilename());
    } else {
      assert(Output.isNothing() && "Invalid output.");
    }

    if (Args.hasArg(options::OPT_g_Flag))
      CmdArgs.push_back("-g");

    if (Args.hasArg(options::OPT_v))
      CmdArgs.push_back("-v");

    std::vector<std::string> gpu_archs =
        Args.getAllArgValues(options::OPT_march_EQ);
    assert(gpu_archs.size() == 1 && "Exactly one GPU Arch required for ptxas.");
    const std::string &gpu_arch = gpu_archs[0];

    CmdArgs.push_back("-arch");
    CmdArgs.push_back(Args.MakeArgString(gpu_arch));

    // add linking against library implementing OpenMP calls on OMPDEV target
    CmdArgs.push_back("-lomptarget-nvptx");

    // nvlink relies on the extension used by the input files
    // to decide what to do. Given that ptxas produces cubin files
    // we need to copy the input files to a new file with the right
    // extension.
    // FIXME: this can be efficiently done by specifying a new
    // output type for the assembly action, however this would expose
    // the target details to the driver and maybe we do not want to do
    // that
    for (const auto &II : Inputs) {

      if (II.getType() == types::TY_LLVM_IR ||
          II.getType() == types::TY_LTO_IR ||
          II.getType() == types::TY_LTO_BC ||
          II.getType() == types::TY_LLVM_BC) {
        C.getDriver().Diag(diag::err_drv_no_linker_llvm_support)
            << getToolChain().getTripleString();
        continue;
      }

      // Currently, we only pass the input files to the linker, we do not pass
      // any libraries that may be valid only for the host.
      if (!II.isFilename())
        continue;

      StringRef Name = llvm::sys::path::filename(II.getFilename());
      std::pair<StringRef, StringRef> Split = Name.rsplit('.');
      std::string TmpName =
          C.getDriver().GetTemporaryPath(Split.first, "cubin");

      const char *CubinF =
          C.addTempFile(C.getArgs().MakeArgString(TmpName.c_str()));

      const char *CopyExec = Args.MakeArgString(getToolChain().GetProgramPath(
          C.getDriver().IsCLMode() ? "copy" : "cp"));

      ArgStringList CopyCmdArgs;
      CopyCmdArgs.push_back(II.getFilename());
      CopyCmdArgs.push_back(CubinF);
      C.addCommand(
          llvm::make_unique<Command>(JA, *this, CopyExec, CopyCmdArgs, Inputs));

      CmdArgs.push_back(CubinF);
    }

    AddOpenMPLinkerScript(getToolChain(), C, Output, Inputs, Args, CmdArgs, JA);

    // add paths specified in LIBRARY_PATH environment variable as -L options
    addDirectoryList(Args, CmdArgs, "-L", "LIBRARY_PATH");

    const char *Exec =
        Args.MakeArgString(getToolChain().GetProgramPath("nvlink"));
    C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
    return;
  }

  CmdArgs.push_back("--cuda");
  CmdArgs.push_back(TC.getTriple().isArch64Bit() ? "-64" : "-32");
  CmdArgs.push_back(Args.MakeArgString("--create"));
  CmdArgs.push_back(Args.MakeArgString(Output.getFilename()));

  for (const auto& II : Inputs) {
    auto *A = II.getAction();
    assert(A->getInputs().size() == 1 &&
           "Device offload action is expected to have a single input");
    const char *gpu_arch_str = A->getOffloadingArch();
    assert(gpu_arch_str &&
           "Device action expected to have associated a GPU architecture!");
    CudaArch gpu_arch = StringToCudaArch(gpu_arch_str);

    // We need to pass an Arch of the form "sm_XX" for cubin files and
    // "compute_XX" for ptx.
    const char *Arch =
        (II.getType() == types::TY_PP_Asm)
            ? CudaVirtualArchToString(VirtualArchForCudaArch(gpu_arch))
            : gpu_arch_str;
    CmdArgs.push_back(Args.MakeArgString(llvm::Twine("--image=profile=") +
                                         Arch + ",file=" + II.getFilename()));
  }

  for (const auto& A : Args.getAllArgValues(options::OPT_Xcuda_fatbinary))
    CmdArgs.push_back(Args.MakeArgString(A));

  const char *Exec = Args.MakeArgString(TC.GetProgramPath("fatbinary"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));

  }
}

/// CUDA toolchain.  Our assembler is ptxas, and our "linker" is fatbinary,
/// which isn't properly a linker but nonetheless performs the step of stitching
/// together object files from the assembler into a single blob.

OmpDeviceToolChain::OmpDeviceToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args)
    : ToolChain(D, Triple, Args), HostTC(HostTC),
      CudaInstallation(D, HostTC.getTriple(), Args) {
  if (CudaInstallation.isValid())
    getProgramPaths().push_back(CudaInstallation.getBinPath());

  printf("OmpDeviceToolChain created Triple:%s Host Triple:%s\n",
    Triple.str().c_str(),HostTC.getTripleString().c_str());

}

void OmpDeviceToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args);

  if (DriverArgs.hasFlag(options::OPT_fcuda_flush_denormals_to_zero,
                         options::OPT_fno_cuda_flush_denormals_to_zero, false))
    CC1Args.push_back("-fcuda-flush-denormals-to-zero");

  if (DriverArgs.hasFlag(options::OPT_fcuda_approx_transcendentals,
                         options::OPT_fno_cuda_approx_transcendentals, false))
    CC1Args.push_back("-fcuda-approx-transcendentals");

  if (DriverArgs.hasArg(options::OPT_nocudalib))
    return;

  StringRef GpuArch = DriverArgs.getLastArgValue(options::OPT_march_EQ);
  printf("OmpDeviceToolChain::addClangTargetOptions - GpuArch:%s\n ", GpuArch.str().c_str());
  assert(!GpuArch.empty() && "Must have an explicit GPU arch.");
  std::string LibDeviceFile = CudaInstallation.getLibDeviceFile(GpuArch);

  if (LibDeviceFile.empty()) {
    getDriver().Diag(diag::err_drv_no_cuda_libdevice) << GpuArch;
    return;
  }

  // Do not add -link-cuda-bitcode or ptx42 features if gfx
  for (Arg *A : DriverArgs)
    if( A->getOption().matches(options::OPT_cuda_gpu_arch_EQ) &&
        StringRef(A->getValue()).startswith("gfx") )
      return;

  CC1Args.push_back("-mlink-cuda-bitcode");
  CC1Args.push_back(DriverArgs.MakeArgString(LibDeviceFile));

  // Libdevice in CUDA-7.0 requires PTX version that's more recent
  // than LLVM defaults to. Use PTX4.2 which is the PTX version that
  // came with CUDA-7.0.
  CC1Args.push_back("-target-feature");
  CC1Args.push_back("+ptx42");
}

void OmpDeviceToolChain::AddCudaIncludeArgs(const ArgList &DriverArgs,
                                       ArgStringList &CC1Args) const {
  // Check our CUDA version if we're going to include the CUDA headers.
  if (!DriverArgs.hasArg(options::OPT_nocudainc) &&
      !DriverArgs.hasArg(options::OPT_no_cuda_version_check)) {
    StringRef Arch = DriverArgs.getLastArgValue(options::OPT_march_EQ);
    assert(!Arch.empty() && "Must have an explicit GPU arch.");
    CudaInstallation.CheckCudaVersionSupportsArch(StringToCudaArch(Arch));
  }
  CudaInstallation.AddCudaIncludeArgs(DriverArgs, CC1Args);
}

llvm::opt::DerivedArgList *
OmpDeviceToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             StringRef BoundArch,
                             Action::OffloadKind DeviceOffloadKind) const {
  DerivedArgList *DAL =
      HostTC.TranslateArgs(Args, BoundArch, DeviceOffloadKind);
  if (!DAL)
    DAL = new DerivedArgList(Args.getBaseArgs());

  const OptTable &Opts = getDriver().getOpts();

  for (Arg *A : Args) {
    if (A->getOption().matches(options::OPT_Xarch__)) {
      // Skip this argument unless the architecture matches BoundArch
      if (BoundArch.empty() || A->getValue(0) != BoundArch)
        continue;

      unsigned Index = Args.getBaseArgs().MakeIndex(A->getValue(1));
      unsigned Prev = Index;
      std::unique_ptr<Arg> XarchArg(Opts.ParseOneArg(Args, Index));

      // If the argument parsing failed or more than one argument was
      // consumed, the -Xarch_ argument's parameter tried to consume
      // extra arguments. Emit an error and ignore.
      //
      // We also want to disallow any options which would alter the
      // driver behavior; that isn't going to work in our model. We
      // use isDriverOption() as an approximation, although things
      // like -O4 are going to slip through.
      if (!XarchArg || Index > Prev + 1) {
        getDriver().Diag(diag::err_drv_invalid_Xarch_argument_with_args)
            << A->getAsString(Args);
        continue;
      } else if (XarchArg->getOption().hasFlag(options::DriverOption)) {
        getDriver().Diag(diag::err_drv_invalid_Xarch_argument_isdriver)
            << A->getAsString(Args);
        continue;
      }
      XarchArg->setBaseArg(A);
      A = XarchArg.release();
      DAL->AddSynthesizedArg(A);
    }
    DAL->append(A);
  }

  if (!BoundArch.empty()) {
    printf("OmpDeviceToolChain::TranslateArgs setting BoundArch %s\n",BoundArch.str().c_str());
    DAL->eraseArg(options::OPT_march_EQ);
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ), BoundArch);
  }
  return DAL;
}

Tool *OmpDeviceToolChain::buildBackend() const {
  printf("OMPDEV::Backend used\n");
  return new tools::OMPDEV::Backend(*this);
}

Tool *OmpDeviceToolChain::buildAssembler() const {
  printf("OMPDEV::Assembler used\n");
  return new tools::OMPDEV::Assembler(*this);
}

Tool *OmpDeviceToolChain::buildLinker() const {
  printf("OMPDEV::Linker used\n");
  return new tools::OMPDEV::Linker(*this);
}
