// RUN: %clang_cc1 -fhc-is-device -fhsa-ext -std=c++11 -x hc-kernel -triple amdgcn -target-cpu fiji -emit-llvm -disable-llvm-passes -o - %s| FileCheck %s
//
// This test emulates parallel-for-each without relying on HCC header files.
// By using pseudo definitions of some HCC types this test can generate the trampoline functions which are
// needed for testing the register control attributes.
// The objective is to focus on language aspects without introducing unnecessary declarations in the header files.

class accelerator_view { int dummy; };
class extent { int dummy; };
struct index {
  index() {}
  int x;
};

struct array {
  int x;
  void foo() [[hc]] {}
};

template <typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    const accelerator_view& av, const extent& compute_domain, const Kernel& f) [[hc]] {
  auto foo = &Kernel::__cxxamp_trampoline;
  auto bar = &Kernel::operator();
}

int* foo(int x)[[hc]];

int main() {
  int x[10];

  accelerator_view acc;
  extent ext;
  array arr;

  // Test parallel-for-each with functor.
  class A {
  public:
    void foo() [[hc]] {}
    // CHECK-LABEL: define internal amdgpu_kernel void @_ZZ4mainEN1A19__cxxamp_trampolineEi(i32)
    // CHECK-SAME: #[[ATTR2:[0-9]+]]
    void operator()(index& i)
    [[hc]]
    [[hc_waves_per_eu(3)]]
    [[hc_flat_workgroup_size(1,1)]]
    [[hc_flat_workgroup_size(2,2,"gfx700")]]
    [[hc_flat_workgroup_size(3,3,"gfx701")]]
    [[hc_flat_workgroup_size(7,7,"gfx803")]]
    [[hc_flat_workgroup_size(4,4,"gfx800")]]
    [[hc_flat_workgroup_size(5,5,"gfx801")]]
    [[hc_flat_workgroup_size(6,6,"gfx802")]]
    [[hc_max_workgroup_dim(4,5,6)]]
    { x = i.x; }
    int x;
  } a;

  parallel_for_each(acc, ext, a);

  // Test parallel-for-each with lambda function.
  // CHECK-LABEL: define internal amdgpu_kernel void @"_ZZ4mainEN3$_019__cxxamp_trampolineEP5array"(%struct.array*)
  // CHECK-SAME: #[[ATTR3:[0-9]+]]
  parallel_for_each(acc, ext, [&](index& i)
      [[hc]]
      [[hc_waves_per_eu(4)]]
      [[hc_flat_workgroup_size(5)]]
      [[hc_max_workgroup_dim(6,7,8)]]
      {
        arr.x = 123;
      });

  // Test parallel-for-each with lambda function.
  // CHECK-LABEL: define internal amdgpu_kernel void @"_ZZ4mainEN3$_119__cxxamp_trampolineEP5array"(%struct.array*)
  // CHECK-SAME: #[[ATTR4:[0-9]+]]
  parallel_for_each(acc, ext, [&](index& i)
      [[hc]]
      [[hc_max_workgroup_dim(3,4,5)]]
      {
        arr.x = 123;
      });

  return 0;
}

// CHECK: attributes #[[ATTR2]] ={{.*}}"amdgpu-flat-work-group-size"="7,7" "amdgpu-max-work-group-dim"="4,5,6" "amdgpu-waves-per-eu"="3"
// CHECK: attributes #[[ATTR3]] ={{.*}}"amdgpu-flat-work-group-size"="5" "amdgpu-max-work-group-dim"="6,7,8" "amdgpu-waves-per-eu"="4"
// CHECK: attributes #[[ATTR4]] ={{.*}}"amdgpu-flat-work-group-size"="1,60" "amdgpu-max-work-group-dim"="3,4,5"
