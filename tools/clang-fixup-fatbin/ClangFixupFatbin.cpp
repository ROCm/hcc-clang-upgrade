//===-- clang-offload-bundler/ClangOffloadBundler.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a tool to Fixup Cuda Fatbinary input file by
/// changing the sm arch field by gfx one.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::support::endian;

const char *FatBin_Section_Name = ".nv_fatbin";
const uint16_t FatBin_Section_Internal_Offset = 16;
const uint16_t FatBin_Section_Id_PTX = 1;
const uint16_t FatBin_Section_Id_CUDA = 2;
const uint16_t FatBin_Section_Id_GFX = 3;
struct Arch {
  std::string prefix;
  uint16_t number;
};
const Arch FatBin_Section_Arch_Subst = {"sm_", 37};
std::vector<Arch> Archs;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory
    ClangFixupFatbinCategory("clang-fixup-fatbin options");

static cl::opt<std::string>
    InputFileName(cl::Required, cl::Positional,
                   cl::desc("<input file>"),
                   cl::cat(ClangFixupFatbinCategory));
static cl::opt<std::string>
    OutputFileName(cl::Required, cl::Positional,
                    cl::desc("<output file>"),
                    cl::cat(ClangFixupFatbinCategory));
static cl::list<std::string>
    ArchNames("offload-archs", cl::Required, cl::CommaSeparated, cl::OneOrMore,
                cl::desc("[<offload arch>,...]"),
                cl::cat(ClangFixupFatbinCategory));


int main(int argc, const char **argv) {
  cl::HideUnrelatedOptions(ClangFixupFatbinCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to Fixup Cuda Fatbinary input file by changing the sm arch field by gfx one.\n");

  if (Help)
    cl::PrintHelpMessage();

  std::string status = "";
  std::string str;
  size_t pp, pos = 0;
  unsigned i;
  int ii;

//TBD: add check for exceptions for stoi(), e.g. for not a number, like sm_eee 
//TBD: add check for (uint16_t) conversion
  for(i = 0; i < ArchNames.size(); ++i) {
     if((pos = ArchNames[i].find("sm_")) != std::string::npos) {
        str = ArchNames[i].substr(pos + 3);
        ii = std::stoi(str, &pp);
        if(pp != str.size()) break;
        Archs.push_back({"sm_", (uint16_t)ii});
     } else if ((pos = ArchNames[i].find("gfx")) != std::string::npos) {
        str = ArchNames[i].substr(pos + 3);
        ii = std::stoi(str, &pp);
        if(pp != str.size()) break;
        Archs.push_back({"gfx", (uint16_t)ii});
     } else break;
  }
  if(pos == std::string::npos or pp != str.size()) {
    llvm::errs() << "Error: Unsupported arch: " << ArchNames[i] <<"\n";
    return -1;
  }

  std::ifstream inputFile(InputFileName, std::ios::binary);
  std::ofstream outputFile(OutputFileName, std::ios::binary);

  if(!inputFile.is_open()) {
    llvm::errs() << "Error: Cannot open file: " << InputFileName << "\n";
    outputFile.close();
    return -1;
  }

  if(!outputFile.is_open()) {
    llvm::errs() << "Error: Cannot open file: " << OutputFileName << "\n";
    inputFile.close();
    return -2;
  }

// Won't work if file is greater than 2Gb, though it is very unlikely for files in the target area
  inputFile.seekg (0, std::ios::end);
  uint64_t sizeFile = inputFile.tellg();
  char *buffer = new char [sizeFile];
  inputFile.seekg (0, std::ios::beg);
  inputFile.read(buffer, sizeFile);
  inputFile.close();

// Actually the header size is 52 for 32-bit
  if(sizeFile < 64) {
    status = "Error: Not ELF file: too short";
  }
  else if(buffer[0] != 0x7F && buffer[1] != 0x45 && buffer[2] != 0x4C && buffer[3] != 0x46) {
    status = "Error: Not ELF file: wrong signature";
  }
  else if((unsigned short)buffer[4] != 2){
    status = "Error: Not supported ELF file: not 64-bit";
  }
  else if((unsigned short)buffer[5] != 1){
    status = "Error: Not supported ELF file: not little endian";
  }
  if(status.compare("") != 0) {
    llvm::errs() << status <<"\n";
    outputFile.close();
    return -3;
  }

  uint64_t e_shoff = read64le(&buffer[40]);
  uint16_t e_shentsize = read16le(&buffer[58]);
  uint16_t e_shnum = read16le(&buffer[60]);
  uint16_t e_shstrndx = read16le(&buffer[62]);
  uint64_t shstr = e_shoff + e_shstrndx * e_shentsize;

  if(sizeFile < e_shoff + e_shnum * e_shentsize) {
    status = "Error: Broken ELF file: too short";
    llvm::errs() << status <<"\n";
    outputFile.close();
    return -3;
  }

  uint64_t sh_names_offset = read64le(&buffer[shstr] + 24);

  uint64_t sh;
  uint32_t sh_name, sh_size, sh_offset;
  std::string name;
  int idx;
  int idx_check = 0;

  for (idx = 0; idx < e_shnum; ++idx) {
    sh = e_shoff + idx * e_shentsize;
    sh_name = read32le(&buffer[sh]);
    name = &buffer[sh_names_offset + sh_name];
    if(!name.compare(FatBin_Section_Name)) {
     ++idx_check;
     break;
    }
  }
  if(idx_check != 1) {
    llvm::errs() << "Error: Not supported ELF file: number of " << FatBin_Section_Name << " sections is " << idx_check <<"\n";
    outputFile.close();
    return -4;
  }

  sh_offset = read64le(&buffer[sh] + 24);
  sh_size = read64le(&buffer[sh] + 32);

  uint16_t fatbin_sec_id, fatbin_sec_header_size, fatbin_sec_arch;
  uint64_t fatbin_sec_size;
  uint64_t fatbin_sec_start = sh_offset + FatBin_Section_Internal_Offset;
  int n = -1;

  while(fatbin_sec_start < sh_offset + sh_size - 1) {
    fatbin_sec_id = read16le(&buffer[fatbin_sec_start]);
    fatbin_sec_header_size = read16le(&buffer[fatbin_sec_start + 4]);
    fatbin_sec_size = read64le(&buffer[fatbin_sec_start + 8]);
    fatbin_sec_arch = read16le(&buffer[fatbin_sec_start + 28]);

    llvm::outs() << "FatBin Section found at " << format("%#x", fatbin_sec_start) << "  Arch: " << fatbin_sec_arch << "  Id: " <<
      fatbin_sec_id << "  Header size: " << fatbin_sec_header_size << "  Section size: " << fatbin_sec_size << "\n";

    if(fatbin_sec_id != FatBin_Section_Id_PTX && fatbin_sec_id != FatBin_Section_Id_CUDA && fatbin_sec_id != FatBin_Section_Id_GFX)
    {
      llvm::errs() << "Error: Unsupported FatBin Section id: " << fatbin_sec_id << " at " << format("%#x", fatbin_sec_start) << "\n";
      outputFile.close();
      return -10;
    }

    if(fatbin_sec_id == FatBin_Section_Id_CUDA &&
       buffer[fatbin_sec_start + fatbin_sec_header_size] != 0x7F &&
       buffer[fatbin_sec_start + fatbin_sec_header_size + 1] != 0x45 &&
       buffer[fatbin_sec_start + fatbin_sec_header_size + 2] != 0x4C &&
       buffer[fatbin_sec_start + fatbin_sec_header_size + 3] != 0x46) {
      status = "Error: FatBin Section has incorrect structure: Not ELF sub section: wrong signature";
    }
    if(status.compare("") != 0) {
      llvm::errs() << status << " at " << format("%#x", fatbin_sec_start + fatbin_sec_header_size) <<"\n";
      outputFile.close();
      return -100;
    }

    if(fatbin_sec_id == FatBin_Section_Id_CUDA) {
      ++n;
      if((unsigned)n >= Archs.size()) {
        llvm::errs() << "Error: more FatBin CUDA Sections than expected: --offload_archs doesn't enumerate all the sections\n";
        outputFile.close();
        return -20;
      }
      if(! Archs[n].prefix.compare("gfx")) {
        if(fatbin_sec_arch != FatBin_Section_Arch_Subst.number) {
          llvm::errs() << "Error: FatBin Section Subst Arch differs from the expected: " << fatbin_sec_arch << " vs " << FatBin_Section_Arch_Subst.number << "\n";
          outputFile.close();
          return -30;
        } else {
          write16le(&buffer[fatbin_sec_start], FatBin_Section_Id_GFX);
          write16le(&buffer[fatbin_sec_start + 28], Archs[n].number);
          llvm::outs() << "   Section modified...  Arch: " << Archs[n].number << "  Id: " <<  FatBin_Section_Id_GFX << "\n";
        }
      } else if(fatbin_sec_arch != Archs[n].number) {
        llvm::errs() << "Error: FatBin Section Arch differs from the expected: " << fatbin_sec_arch << " vs " << Archs[n].number << "\n";
        outputFile.close();
        return -40;
      } else {
        llvm::outs() << "   Section skipped...\n";
      }
    }

    fatbin_sec_start += fatbin_sec_header_size + fatbin_sec_size;

    if(fatbin_sec_id == FatBin_Section_Id_GFX) ++n;
    if(fatbin_sec_id != FatBin_Section_Id_CUDA) {
      llvm::outs() << "   Section skipped...\n";
    }
  }

  if(status.compare("") != 0) {
    llvm::errs() << status <<"\n";
    outputFile.close();
    return -10;
  }

  outputFile.write(buffer, sizeFile);
  outputFile.close();
  delete[] buffer;

  return 0;
}
