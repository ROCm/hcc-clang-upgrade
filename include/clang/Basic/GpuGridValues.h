//===--- GpuGridValues.h - Language-specific address spaces -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provides definitions for Target specific Grid Values
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OPENMPGRIDVALUES_H
#define LLVM_CLANG_BASIC_OPENMPGRIDVALUES_H

namespace clang {

namespace GPU {

/// \brief Defines various target specific Gpu grid values 
///        that must be consistent with host and device RTL. 
///
/// Example usage in clang:  
/// const unsigned slot_size = ctx.GetTargetInfo().getGridValue(GV_Slot_Size);
///
/// Example usage in deviceRTL:  
///    #ifdef GPUCC_AMDGPU 
///    #define GRIDVAL AMDGPUGpuGridValues
///    #else
///    #define GRIDVAL NVPTXGpuPGridValues
///    #endif
///    GRIDVAL[GV_Slot_Size] 
///
enum GVIDX{
  /// The maximum number of workers in a kernel.
  /// (THREAD_ABSOLUTE_LIMIT) - (GV_Warp_Size), might be issue for blockDim.z
  GV_Threads,
  /// The size reserved for data in a shared memory slot.
  GV_Slot_Size,
  /// The maximum number of threads in a worker warp.
  GV_Warp_Size,
  /// The number of bits required to represent the max number of threads in warp
  GV_Warp_Size_Log2,
  /// GV_Warp_Size * GV_Slot_Size,
  GV_Warp_Slot_Size,
  /// the maximum number of teams.
  GV_Max_Teams,
  /// Global Memory Alignment
  GV_Mem_Align,
  /// (~0u >> (GV_Warp_Size - GV_Warp_Size_Log2))
  GV_Warp_Size_Log2_Mask
};

enum GVLIDX{
  /// The slot size that should be reserved for a working warp.
  /// (~0u >> (GV_Warp_Size - GV_Warp_Size_Log2))
  GV_Warp_Size_Log2_MaskL
}; 

}

/// For AMDGPU GPUs
static const int AMDGPUGpuGridValues[] = {
  960,               // GV_Threads
  256,               // GV_Slot_Size 
  64,                // GV_Warp_Size 
  6,                 // GV_Warp_Size_Log2 
  64 * 256,          // GV_Warp_Slot_Size 
  1024,              // GV_Max_Teams 
  256,
  (~0u >> (32 - 6))  // GV_Warp_Size_Log2_Mask
};
static const long long AMDGPUGpuLongGridValues[] = {
  (~0ull >> (64 - 6)) // GV_Warp_Size_Log2_MaskL
};

/// For Nvidia GPUs
static const int NVPTXGpuGridValues[] = {
  992,               // GV_Threads
  256,               // GV_Slot_Size 
  32,                // GV_Warp_Size 
  5,                 // GV_Warp_Size_Log2 
  32 * 256,          // GV_Warp_Slot_Size 
  1024,              // GV_Max_Teams 
  256,               // GV_Mem_Align
  (~0u >> (32 - 5))  // GV_Warp_Size_Log2_Mask
};

static const long long NVPTXGpuLongGridValues[] = {
  (~0ull >> (64 - 5))  // GV_Warp_Size_Log2_MaskL
};

}
#endif
