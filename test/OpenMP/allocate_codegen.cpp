// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-apple-darwin10.6.0 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-apple-darwin10.6.0 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-apple-darwin10.6.0 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-apple-darwin10.6.0 -emit-llvm -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-apple-darwin10.6.0 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-apple-darwin10.6.0 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

typedef void **omp_allocator_handle_t;
extern const omp_allocator_handle_t omp_default_mem_alloc;
extern const omp_allocator_handle_t omp_large_cap_mem_alloc;
extern const omp_allocator_handle_t omp_const_mem_alloc;
extern const omp_allocator_handle_t omp_high_bw_mem_alloc;
extern const omp_allocator_handle_t omp_low_lat_mem_alloc;
extern const omp_allocator_handle_t omp_cgroup_mem_alloc;
extern const omp_allocator_handle_t omp_pteam_mem_alloc;
extern const omp_allocator_handle_t omp_thread_mem_alloc;

struct St{
 int a;
};

struct St1{
 int a;
 static int b;
#pragma omp allocate(b) allocator(omp_default_mem_alloc)
} d;

int a, b, c;
#pragma omp allocate(a) allocator(omp_large_cap_mem_alloc)
#pragma omp allocate(b) allocator(omp_const_mem_alloc)
#pragma omp allocate(d, c) allocator(omp_high_bw_mem_alloc)

template <class T>
struct ST {
  static T m;
  #pragma omp allocate(m) allocator(omp_low_lat_mem_alloc)
};

template <class T> T foo() {
  T v;
  #pragma omp allocate(v) allocator(omp_cgroup_mem_alloc)
  v = ST<T>::m;
  return v;
}

namespace ns{
  int a;
}
#pragma omp allocate(ns::a) allocator(omp_pteam_mem_alloc)

// CHECK-NOT:  call {{.+}} {{__kmpc_alloc|__kmpc_free}}

// CHECK-LABEL: @main
int main () {
  static int a;
#pragma omp allocate(a) allocator(omp_thread_mem_alloc)
  a=2;
  // CHECK-NOT:  {{__kmpc_alloc|__kmpc_free}}
  // CHECK:      alloca double,
  // CHECK-NOT:  {{__kmpc_alloc|__kmpc_free}}
  double b = 3;
#pragma omp allocate(b)
  return (foo<int>());
}

// CHECK: define {{.*}}i32 @{{.+}}foo{{.+}}()
// CHECK:      [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{.+}})
// CHECK-NEXT: [[OMP_CGROUP_MEM_ALLOC:%.+]] = load i8**, i8*** @omp_cgroup_mem_alloc,
// CHECK-NEXT: [[V_VOID_ADDR:%.+]] = call i8* @__kmpc_alloc(i32 [[GTID]], i64 4, i8** [[OMP_CGROUP_MEM_ALLOC]])
// CHECK-NEXT: [[V_ADDR:%.+]] = bitcast i8* [[V_VOID_ADDR]] to i32*
// CHECK-NOT:  {{__kmpc_alloc|__kmpc_free}}
// CHECK:      store i32 %{{.+}}, i32* [[V_ADDR]],
// CHECK-NEXT: [[V_VAL:%.+]] = load i32, i32* [[V_ADDR]],
// CHECK-NEXT: call void @__kmpc_free(i32 [[GTID]], i8* [[V_VOID_ADDR]], i8** [[OMP_CGROUP_MEM_ALLOC]])
// CHECK-NOT:  {{__kmpc_alloc|__kmpc_free}}
// CHECK:      ret i32 [[V_VAL]]

// CHECK-NOT:  call {{.+}} {{__kmpc_alloc|__kmpc_free}}
extern template int ST<int>::m;
#endif
