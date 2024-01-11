/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_COMPAT_RT_VERSION 12010
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#if (DPCT_COMPAT_RT_VERSION >= 11000)
#else
// Hack to get cub device reduce on older toolkits
#include <thrust/system/cuda/detail/cub/device/device_reduce.cuh>
using namespace thrust::cuda_cub;
#endif
#include <algorithm>
#include <functional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>

namespace gpu_treeshap {

struct XgboostSplitCondition {
  XgboostSplitCondition() = default;
  XgboostSplitCondition(float feature_lower_bound, float feature_upper_bound,
                        bool is_missing_branch)
      : feature_lower_bound(feature_lower_bound),
        feature_upper_bound(feature_upper_bound),
        is_missing_branch(is_missing_branch) {
    assert(feature_lower_bound <= feature_upper_bound);
  }

  /*! Feature values >= lower and < upper flow down this path. */
  float feature_lower_bound;
  float feature_upper_bound;
  /*! Do missing values flow down this path? */
  bool is_missing_branch;

  // Does this instance flow down this path?
  bool EvaluateSplit(float x) const {
    // is nan
    if (sycl::isnan(x)) {
      return is_missing_branch;
    }
    return x >= feature_lower_bound && x < feature_upper_bound;
  }

  // Combine two split conditions on the same feature
  void Merge(
      const XgboostSplitCondition& other) {  // Combine duplicate features
    feature_lower_bound =
        std::max(feature_lower_bound, other.feature_lower_bound);
    feature_upper_bound =
        std::min(feature_upper_bound, other.feature_upper_bound);
    is_missing_branch = is_missing_branch && other.is_missing_branch;
  }
};

/*!
 * An element of a unique path through a decision tree. Can implement various
 * types of splits via the templated SplitConditionT. Some decision tree
 * implementations may wish to use double precision or single precision, some
 * may use < or <= as the threshold, missing values can be handled differently,
 * categoricals may be supported.
 *
 * \tparam  SplitConditionT A split condition implementing the methods
 * EvaluateSplit and Merge.
 */
template <typename SplitConditionT>
struct PathElement {
  using split_type = SplitConditionT;
  PathElement(size_t path_idx, int64_t feature_idx,
                                  int group, SplitConditionT split_condition,
                                  double zero_fraction, float v)
      : path_idx(path_idx),
        feature_idx(feature_idx),
        group(group),
        split_condition(split_condition),
        zero_fraction(zero_fraction),
        v(v) {}

  PathElement() = default;
  bool IsRoot() const { return feature_idx == -1; }

  template <typename DatasetT>
  bool EvaluateSplit(DatasetT X, size_t row_idx) const {
    if (this->IsRoot()) {
      return 1.0;
    }
    return split_condition.EvaluateSplit(X.GetElement(row_idx, feature_idx));
  }

  /*! Unique path index. */
  size_t path_idx;
  /*! Feature of this split, -1 indicates bias term. */
  int64_t feature_idx;
  /*! Indicates class for multiclass problems. */
  int group;
  SplitConditionT split_condition;
  /*! Probability of following this path when feature_idx is not in the active
   * set. */
  double zero_fraction;
  float v;  // Leaf weight at the end of the path
};

// Helper function that accepts an index into a flat contiguous array and the
// dimensions of a tensor and returns the indices with respect to the tensor
template <typename T, size_t N>
void FlatIdxToTensorIdx(T flat_idx, const T (&shape)[N],
                                   T (&out_idx)[N]) {
  T current_size = shape[0];
  for (auto i = 1ull; i < N; i++) {
    current_size *= shape[i];
  }
  for (auto i = 0ull; i < N; i++) {
    current_size /= shape[i];
    out_idx[i] = flat_idx / current_size;
    flat_idx -= current_size * out_idx[i];
  }
}

// Given a shape and coordinates into a tensor, return the index into the
// backing storage one-dimensional array
template <typename T, size_t N>
T TensorIdxToFlatIdx(const T (&shape)[N], const T (&tensor_idx)[N]) {
  T current_size = shape[0];
  for (auto i = 1ull; i < N; i++) {
    current_size *= shape[i];
  }
  T idx = 0;
  for (auto i = 0ull; i < N; i++) {
    current_size /= shape[i];
    idx += tensor_idx[i] * current_size;
  }
  return idx;
}

// Maps values to the phi array according to row, group and column
inline size_t IndexPhi(size_t row_idx, size_t num_groups,
                                           size_t group, size_t num_columns,
                                           size_t column_idx) {
  return (row_idx * num_groups + group) * (num_columns + 1) + column_idx;
}

inline size_t IndexPhiInteractions(size_t row_idx,
                                                       size_t num_groups,
                                                       size_t group,
                                                       size_t num_columns,
                                                       size_t i, size_t j) {
  size_t matrix_size = (num_columns + 1) * (num_columns + 1);
  size_t matrix_offset = (row_idx * num_groups + group) * matrix_size;
  return matrix_offset + i * (num_columns + 1) + j;
}

namespace detail {

// Shorthand for creating a device vector with an appropriate allocator type
template <class T, class DeviceAllocatorT>
using RebindVector =
    dpct::device_vector<T,
                        typename DeviceAllocatorT::template rebind<T>::other>;

#if !defined(DPCT_COMPATIBILITY_TEMP) || DPCT_COMPATIBILITY_TEMP >= 600 ||     \
    defined(__clang__)
__dpct_inline__ double atomicAddDouble(double *address, double val) {
  return dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
      address, val);
}
#else  // In device code and CUDA < 600
__device__ __forceinline__ double atomicAddDouble(double* address,
                                                  double val) {  // NOLINT
  unsigned long long int* address_as_ull =                       // NOLINT
      (unsigned long long int*)address;                          // NOLINT
  unsigned long long int old = *address_as_ull, assumed;         // NOLINT

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

__dpct_inline__ unsigned int lanemask32_lt() {
  unsigned int lanemask32_lt;
  /*
  DPCT1053:0: Migration of device assembly code is not supported.
  */
  asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask32_lt));
  return (lanemask32_lt);
}

// Like a coalesced group, except we can make the assumption that all threads in
// a group are next to each other. This makes shuffle operations much cheaper.
class ContiguousGroup {
 public:
  ContiguousGroup(uint32_t mask) : mask_(mask) {}

  uint32_t size() const { return sycl::popcount(mask_); }
  uint32_t thread_rank() const {
    return sycl::popcount(mask_ & lanemask32_lt());
  }
  template <typename T>
  T shfl(T val, uint32_t src, const sycl::nd_item<3> &item_ct1) const {
    /*
    DPCT1023:1: The SYCL sub-group does not support mask options for
    dpct::select_from_sub_group. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_sync.
    */
    return dpct::select_from_sub_group(item_ct1.get_sub_group(), val,
                                       src + dpct::ffs<int>(mask_) - 1);
  }
  template <typename T>
  T shfl_up(T val, uint32_t delta, const sycl::nd_item<3> &item_ct1) const {
    /*
    DPCT1023:2: The SYCL sub-group does not support mask options for
    dpct::shift_sub_group_right. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_up_sync.
    */
    return dpct::shift_sub_group_right(item_ct1.get_sub_group(), val, delta);
  }
  uint32_t ballot(int predicate, const sycl::nd_item<3> &item_ct1) const {
    return sycl::reduce_over_group(
               item_ct1.get_sub_group(),
               (mask_ &
                (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
                       predicate
                   ? (0x1 << item_ct1.get_sub_group().get_local_linear_id())
                   : 0,
               sycl::ext::oneapi::plus<>()) >>
           (dpct::ffs<int>(mask_) - 1);
  }

  template <typename T, typename OpT>
  T reduce(T val, OpT op) {
    for (int i = 1; i < this->size(); i *= 2) {
      T shfl = shfl_up(val, i);
      if (static_cast<int>(thread_rank()) - i >= 0) {
        val = op(val, shfl);
      }
    }
    return shfl(val, size() - 1);
  }
  uint32_t mask_;
};

// Separate the active threads by labels
// This functionality is available in cuda 11.0 on cc >=7.0
// We reimplement for backwards compatibility
// Assumes partitions are contiguous
inline ContiguousGroup active_labeled_partition(uint32_t mask,
                                                           int label,
                                                           const sycl::nd_item<3> &item_ct1) {
#if DPCT_COMPATIBILITY_TEMP >= 700
  uint32_t subgroup_mask =
      dpct::match_any_over_sub_group(item_ct1.get_sub_group(), mask, label);
#else
  uint32_t subgroup_mask = 0;
  for (int i = 0; i < 32;) {
    int current_label = __shfl_sync(mask, label, i);
    uint32_t ballot = __ballot_sync(mask, label == current_label);
    if (label == current_label) {
      subgroup_mask = ballot;
    }
    uint32_t completed_mask =
        (1 << (32 - __clz(ballot))) - 1;  // Threads that have finished
    // Find the start of the next group, mask off completed threads from active
    // threads Then use ffs - 1 to find the position of the next group
    int next_i = __ffs(mask & ~completed_mask) - 1;
    if (next_i == -1) break;  // -1 indicates all finished
    assert(next_i > i);  // Prevent infinite loops when the constraints not met
    i = next_i;
  }
#endif
  return ContiguousGroup(subgroup_mask);
}

// Group of threads where each thread holds a path element
class GroupPath {
 protected:
  const ContiguousGroup& g_;
  // These are combined so we can communicate them in a single 64 bit shuffle
  // instruction
  float zero_one_fraction_[2];
  float pweight_;
  int unique_depth_;

 public:
  GroupPath(const ContiguousGroup& g, float zero_fraction,
                       float one_fraction)
      : g_(g),
        zero_one_fraction_{zero_fraction, one_fraction},
        pweight_(g.thread_rank() == 0 ? 1.0f : 0.0f),
        unique_depth_(0) {}

  // Cooperatively extend the path with a group of threads
  // Each thread maintains pweight for its path element in register
  void Extend(const sycl::nd_item<3> &item_ct1) {
    unique_depth_++;

    // Broadcast the zero and one fraction from the newly added path element
    // Combine 2 shuffle operations into 64 bit word
    const size_t rank = g_.thread_rank();
    const float inv_unique_depth = 1.0f / static_cast<float>(unique_depth_ + 1);
    uint64_t res = g_.shfl(*reinterpret_cast<uint64_t *>(&zero_one_fraction_),
                           unique_depth_, item_ct1);
    const float new_zero_fraction = reinterpret_cast<float*>(&res)[0];
    const float new_one_fraction = reinterpret_cast<float*>(&res)[1];
    float left_pweight = g_.shfl_up(pweight_, 1, item_ct1);

    // pweight of threads with rank < unique_depth_ is 0
    // We use max(x,0) to avoid using a branch
    // pweight_ *=
    // new_zero_fraction * max(unique_depth_ - rank, 0llu) * inv_unique_depth;
    /*
    DPCT1013:5: The rounding mode could not be specified and the generated code
    may have different accuracy than the original code. Verify the correctness.
    SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
    standard.
    */
    pweight_ = pweight_ * new_zero_fraction *
               sycl::max(unique_depth_ - rank, size_t(0)) * inv_unique_depth;

    // pweight_  += new_one_fraction * left_pweight * rank * inv_unique_depth;
    /*
    DPCT1013:8: The rounding mode could not be specified and the generated code
    may have different accuracy than the original code. Verify the correctness.
    SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
    standard.
    */
    pweight_ = sycl::fma(new_one_fraction * left_pweight,
                         rank * inv_unique_depth, pweight_);
  }

  // Each thread unwinds the path for its feature and returns the sum
  float UnwoundPathSum(const sycl::nd_item<3> &item_ct1) {
    float next_one_portion = g_.shfl(pweight_, unique_depth_, item_ct1);
    float total = 0.0f;
    const float zero_frac_div_unique_depth =
        zero_one_fraction_[0] / static_cast<float>(unique_depth_ + 1);
    for (int i = unique_depth_ - 1; i >= 0; i--) {
      float ith_pweight = g_.shfl(pweight_, i, item_ct1);
      float precomputed =
          /*
          DPCT1013:10: The rounding mode could not be specified and the
          generated code may have different accuracy than the original code.
          Verify the correctness. SYCL math built-in function rounding mode is
          aligned with OpenCL C 1.2 standard.
          */
          (unique_depth_ - i) * zero_frac_div_unique_depth;
      const float tmp =
          /*
          DPCT1013:11: The rounding mode could not be specified and the
          generated code may have different accuracy than the original code.
          Verify the correctness. SYCL math built-in function rounding mode is
          aligned with OpenCL C 1.2 standard.
          */
          next_one_portion * unique_depth_ + 1 / (i + 1);
      /*
      DPCT1013:12: The rounding mode could not be specified and the generated
      code may have different accuracy than the original code. Verify the
      correctness. SYCL math built-in function rounding mode is aligned with
      OpenCL C 1.2 standard.
      */
      total = sycl::fma((float)tmp, zero_one_fraction_[1], total);
      /*
      DPCT1013:13: The rounding mode could not be specified and the generated
      code may have different accuracy than the original code. Verify the
      correctness. SYCL math built-in function rounding mode is aligned with
      OpenCL C 1.2 standard.
      */
      next_one_portion = sycl::fma(-tmp, precomputed, ith_pweight);
      float numerator =
          /*
          DPCT1013:14: The rounding mode could not be specified and the
          generated code may have different accuracy than the original code.
          Verify the correctness. SYCL math built-in function rounding mode is
          aligned with OpenCL C 1.2 standard.
          */
          1.0f - zero_one_fraction_[1] * ith_pweight;
      if (precomputed > 0.0f) {
        total += numerator / precomputed;
      }
    }

    return total;
  }
};

// Has different permutation weightings to the above
// Used in Taylor Shapley interaction index
class TaylorGroupPath : GroupPath {
 public:
  TaylorGroupPath(const ContiguousGroup& g, float zero_fraction,
                             float one_fraction)
      : GroupPath(g, zero_fraction, one_fraction) {}

  // Extend the path is normal, all reweighting can happen in UnwoundPathSum
  void Extend(const sycl::nd_item<3> &item_ct1) {
      GroupPath::Extend(item_ct1);
  }

  // Each thread unwinds the path for its feature and returns the sum
  // We use a different permutation weighting for Taylor interactions
  // As if the total number of features was one larger
  float UnwoundPathSum(const sycl::nd_item<3> &item_ct1) {
    float one_fraction = zero_one_fraction_[1];
    float zero_fraction = zero_one_fraction_[0];
    float next_one_portion = g_.shfl(pweight_, unique_depth_, item_ct1) /
                             static_cast<float>(unique_depth_ + 2);

    float total = 0.0f;
    for (int i = unique_depth_ - 1; i >= 0; i--) {
      float ith_pweight = g_.shfl(pweight_, i, item_ct1) *
                          (static_cast<float>(unique_depth_ - i + 1) /
                           static_cast<float>(unique_depth_ + 2));
      if (one_fraction > 0.0f) {
        const float tmp =
            next_one_portion * (unique_depth_ + 2) / ((i + 1) * one_fraction);

        total += tmp;
        next_one_portion =
            ith_pweight - tmp * zero_fraction *
                              ((unique_depth_ - i + 1) /
                               static_cast<float>(unique_depth_ + 2));
      } else if (zero_fraction > 0.0f) {
        total +=
            (ith_pweight / zero_fraction) /
            ((unique_depth_ - i + 1) / static_cast<float>(unique_depth_ + 2));
      }
    }

    return 2 * total;
  }
};

template <typename DatasetT, typename SplitConditionT>
float ComputePhi(const PathElement<SplitConditionT>& e,
                            size_t row_idx, const DatasetT& X,
                            const ContiguousGroup& group, float zero_fraction,
                            const sycl::nd_item<3> &item_ct1) {
  float one_fraction =
      e.EvaluateSplit(X, row_idx);
  GroupPath path(group, zero_fraction, one_fraction);
  size_t unique_path_length = group.size();

  // Extend the path
  for (auto unique_depth = 1ull; unique_depth < unique_path_length;
       unique_depth++) {
    path.Extend(item_ct1);
  }

  float sum = path.UnwoundPathSum(item_ct1);
  return sum * (one_fraction - zero_fraction) * e.v;
}

inline size_t DivRoundUp(size_t a, size_t b) {
  return (a + b - 1) / b;
}

template <typename DatasetT, size_t kBlockSize, size_t kRowsPerWarp,
          typename SplitConditionT>
void 
ConfigureThread(const DatasetT& X, const size_t bins_per_row,
                const PathElement<SplitConditionT>* path_elements,
                const size_t* bin_segments, size_t* start_row, size_t* end_row,
                PathElement<SplitConditionT>* e, bool* thread_active,
                const sycl::nd_item<3> &item_ct1) {
  // Partition work
  // Each warp processes a set of training instances applied to a path
  size_t tid = kBlockSize * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  const size_t warp_size = 32;
  size_t warp_rank = tid / warp_size;
  if (warp_rank >= bins_per_row * DivRoundUp(X.NumRows(), kRowsPerWarp)) {
    *thread_active = false;
    return;
  }
  size_t bin_idx = warp_rank % bins_per_row;
  size_t bank = warp_rank / bins_per_row;
  size_t path_start = bin_segments[bin_idx];
  size_t path_end = bin_segments[bin_idx + 1];
  uint32_t thread_rank = item_ct1.get_local_id(2) % warp_size;
  if (thread_rank >= path_end - path_start) {
    *thread_active = false;
  } else {
    *e = path_elements[path_start + thread_rank];
    *start_row = bank * kRowsPerWarp;
    *end_row = dpct::min((bank + 1) * kRowsPerWarp, X.NumRows());
    *thread_active = true;
  }
}

#define GPUTREESHAP_MAX_THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff

template <typename DatasetT, size_t kBlockSize, size_t kRowsPerWarp,
          typename SplitConditionT>
void 
    ShapKernel(DatasetT X, size_t bins_per_row,
               const PathElement<SplitConditionT>* path_elements,
               const size_t* bin_segments, size_t num_groups, double* phis,
               const sycl::nd_item<3> &item_ct1, DatasetT &s_X,
               PathElement<SplitConditionT> *s_elements) {
  // Use shared memory for structs, otherwise nvcc puts in local memory

  s_X = X;

  PathElement<SplitConditionT> &e = s_elements[item_ct1.get_local_id(2)];

  size_t start_row, end_row;
  bool thread_active;
  ConfigureThread<DatasetT, kBlockSize, kRowsPerWarp>(
      s_X, bins_per_row, path_elements, bin_segments, &start_row, &end_row, &e,
      &thread_active, item_ct1);
  uint32_t mask = sycl::reduce_over_group(
      item_ct1.get_sub_group(),
      (FULL_MASK & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
              thread_active
          ? (0x1 << item_ct1.get_sub_group().get_local_linear_id())
          : 0,
      sycl::ext::oneapi::plus<>());
  if (!thread_active) return;

  float zero_fraction = e.zero_fraction;
  auto labelled_group = active_labeled_partition(mask, e.path_idx, item_ct1);

  for (int64_t row_idx = start_row; row_idx < end_row; row_idx++) {
    float phi =
        ComputePhi(e, row_idx, X, labelled_group, zero_fraction, item_ct1);

    if (!e.IsRoot()) {
      atomicAddDouble(&phis[IndexPhi(row_idx, num_groups, e.group, X.NumCols(),
                                     e.feature_idx)],
                      phi);
    }
  }
}

template <typename DatasetT, typename SizeTAllocatorT, typename PathAllocatorT,
          typename SplitConditionT>
void ComputeShap(
    DatasetT X,
    const dpct::device_vector<size_t, SizeTAllocatorT> &bin_segments,
    const dpct::device_vector<PathElement<SplitConditionT>, PathAllocatorT>
        &path_elements,
    size_t num_groups, double *phis) {
  size_t bins_per_row = bin_segments.size() - 1;
  const int kBlockThreads = GPUTREESHAP_MAX_THREADS_PER_BLOCK;
  const int warps_per_block = kBlockThreads / 32;
  const int kRowsPerWarp = 1024;
  size_t warps_needed = bins_per_row * DivRoundUp(X.NumRows(), kRowsPerWarp);

  const uint32_t grid_size = DivRoundUp(warps_needed, warps_per_block);

  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      sycl::local_accessor<DatasetT, 0> s_X_acc_ct1(cgh);
      sycl::local_accessor<
          PathElement<SplitConditionT>, 1>
          s_elements_acc_ct1(sycl::range<1>(kBlockThreads), cgh);

      auto path_elements_data_get_ct2 = path_elements.data().get();
      auto bin_segments_data_get_ct3 = bin_segments.data().get();

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                sycl::range<3>(1, 1, kBlockThreads),
                            sycl::range<3>(1, 1, kBlockThreads)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            ShapKernel<DatasetT, kBlockThreads, kRowsPerWarp>(
                X, bins_per_row, path_elements_data_get_ct2,
                bin_segments_data_get_ct3, num_groups, phis, item_ct1,
                s_X_acc_ct1,
                (PathElement<SplitConditionT> *)
                    s_elements_acc_ct1.get_pointer());
          });
    });
  }
}

template <typename PathT, typename DatasetT, typename SplitConditionT>
float ComputePhiCondition(const PathElement<SplitConditionT>& e,
                                     size_t row_idx, const DatasetT& X,
                                     const ContiguousGroup& group,
                                     int64_t condition_feature,
                                     const sycl::nd_item<3> &item_ct1) {
  float one_fraction = e.EvaluateSplit(X, row_idx);
  PathT path(group, e.zero_fraction, one_fraction);
  size_t unique_path_length = group.size();
  float condition_on_fraction = 1.0f;
  float condition_off_fraction = 1.0f;

  // Extend the path
  for (auto i = 1ull; i < unique_path_length; i++) {
    bool is_condition_feature =
        group.shfl(e.feature_idx, i, item_ct1) == condition_feature;
    float o_i = group.shfl(one_fraction, i, item_ct1);
    float z_i = group.shfl(e.zero_fraction, i, item_ct1);

    if (is_condition_feature) {
      condition_on_fraction = o_i;
      condition_off_fraction = z_i;
    } else {
      path.Extend(item_ct1);
    }
  }
  float sum = path.UnwoundPathSum(item_ct1);
  if (e.feature_idx == condition_feature) {
    return 0.0f;
  }
  float phi = sum * (one_fraction - e.zero_fraction) * e.v;
  return phi * (condition_on_fraction - condition_off_fraction) * 0.5f;
}

// If there is a feature in the path we are conditioning on, swap it to the end
// of the path
template <typename SplitConditionT>
inline void SwapConditionedElement(
    PathElement<SplitConditionT>** e, PathElement<SplitConditionT>* s_elements,
    uint32_t condition_rank, const ContiguousGroup& group,
    const sycl::nd_item<3> &item_ct1) {
  auto last_rank = group.size() - 1;
  auto this_rank = group.thread_rank();
  if (this_rank == last_rank) {
    *e = &s_elements[(item_ct1.get_local_id(2) - this_rank) + condition_rank];
  } else if (this_rank == condition_rank) {
    *e = &s_elements[(item_ct1.get_local_id(2) - this_rank) + last_rank];
  }
}

template <typename DatasetT, size_t kBlockSize, size_t kRowsPerWarp,
          typename SplitConditionT>
void 
    ShapInteractionsKernel(DatasetT X, size_t bins_per_row,
                           const PathElement<SplitConditionT>* path_elements,
                           const size_t* bin_segments, size_t num_groups,
                           double* phis_interactions,
                           const sycl::nd_item<3> &item_ct1, DatasetT &s_X,
                           PathElement<SplitConditionT> *s_elements) {
  // Use shared memory for structs, otherwise nvcc puts in local memory

  s_X = X;

  PathElement<SplitConditionT> *e = &s_elements[item_ct1.get_local_id(2)];

  size_t start_row, end_row;
  bool thread_active;
  ConfigureThread<DatasetT, kBlockSize, kRowsPerWarp>(
      s_X, bins_per_row, path_elements, bin_segments, &start_row, &end_row, e,
      &thread_active, item_ct1);
  uint32_t mask = sycl::reduce_over_group(
      item_ct1.get_sub_group(),
      (FULL_MASK & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
              thread_active
          ? (0x1 << item_ct1.get_sub_group().get_local_linear_id())
          : 0,
      sycl::ext::oneapi::plus<>());
  if (!thread_active) return;

  auto labelled_group = active_labeled_partition(mask, e->path_idx, item_ct1);

  for (int64_t row_idx = start_row; row_idx < end_row; row_idx++) {
    float phi =
        ComputePhi(*e, row_idx, X, labelled_group, e->zero_fraction, item_ct1);
    if (!e->IsRoot()) {
      auto phi_offset =
          IndexPhiInteractions(row_idx, num_groups, e->group, X.NumCols(),
                               e->feature_idx, e->feature_idx);
      atomicAddDouble(phis_interactions + phi_offset, phi);
    }

    for (auto condition_rank = 1ull; condition_rank < labelled_group.size();
         condition_rank++) {
      e = &s_elements[item_ct1.get_local_id(2)];
      int64_t condition_feature =
          labelled_group.shfl(e->feature_idx, condition_rank, item_ct1);
      SwapConditionedElement(&e, s_elements, condition_rank, labelled_group,
                             item_ct1);
      float x = ComputePhiCondition<GroupPath>(*e, row_idx, X, labelled_group,
                                               condition_feature, item_ct1);
      if (!e->IsRoot()) {
        auto phi_offset =
            IndexPhiInteractions(row_idx, num_groups, e->group, X.NumCols(),
                                 e->feature_idx, condition_feature);
        atomicAddDouble(phis_interactions + phi_offset, x);
        // Subtract effect from diagonal
        auto phi_diag =
            IndexPhiInteractions(row_idx, num_groups, e->group, X.NumCols(),
                                 e->feature_idx, e->feature_idx);
        atomicAddDouble(phis_interactions + phi_diag, -x);
      }
    }
  }
}

template <typename DatasetT, typename SizeTAllocatorT, typename PathAllocatorT,
          typename SplitConditionT>
void ComputeShapInteractions(
    DatasetT X,
    const dpct::device_vector<size_t, SizeTAllocatorT> &bin_segments,
    const dpct::device_vector<PathElement<SplitConditionT>, PathAllocatorT>
        &path_elements,
    size_t num_groups, double *phis) {
  size_t bins_per_row = bin_segments.size() - 1;
  const int kBlockThreads = GPUTREESHAP_MAX_THREADS_PER_BLOCK;
  const int warps_per_block = kBlockThreads / 32;
  const int kRowsPerWarp = 100;
  size_t warps_needed = bins_per_row * DivRoundUp(X.NumRows(), kRowsPerWarp);

  const uint32_t grid_size = DivRoundUp(warps_needed, warps_per_block);

  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      sycl::local_accessor<DatasetT, 0> s_X_acc_ct1(cgh);
      sycl::local_accessor<
          PathElement<SplitConditionT>, 1>
          s_elements_acc_ct1(sycl::range<1>(kBlockThreads), cgh);

      auto path_elements_data_get_ct2 = path_elements.data().get();
      auto bin_segments_data_get_ct3 = bin_segments.data().get();

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                sycl::range<3>(1, 1, kBlockThreads),
                            sycl::range<3>(1, 1, kBlockThreads)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            ShapInteractionsKernel<DatasetT, kBlockThreads, kRowsPerWarp>(
                X, bins_per_row, path_elements_data_get_ct2,
                bin_segments_data_get_ct3, num_groups, phis, item_ct1,
                s_X_acc_ct1,
                (PathElement<SplitConditionT> *)
                    s_elements_acc_ct1.get_pointer());
          });
    });
  }
}

template <typename DatasetT, size_t kBlockSize, size_t kRowsPerWarp,
          typename SplitConditionT>
void 
    ShapTaylorInteractionsKernel(
        DatasetT X, size_t bins_per_row,
        const PathElement<SplitConditionT>* path_elements,
        const size_t* bin_segments, size_t num_groups,
        double* phis_interactions, const sycl::nd_item<3> &item_ct1,
        DatasetT &s_X, PathElement<SplitConditionT> *s_elements) {
  // Use shared memory for structs, otherwise nvcc puts in local memory

  if (item_ct1.get_local_id(2) == 0) {
    s_X = X;
  }
  /*
  DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  PathElement<SplitConditionT> *e = &s_elements[item_ct1.get_local_id(2)];

  size_t start_row, end_row;
  bool thread_active;
  ConfigureThread<DatasetT, kBlockSize, kRowsPerWarp>(
      s_X, bins_per_row, path_elements, bin_segments, &start_row, &end_row, e,
      &thread_active, item_ct1);
  uint32_t mask = sycl::reduce_over_group(
      item_ct1.get_sub_group(),
      (FULL_MASK & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
              thread_active
          ? (0x1 << item_ct1.get_sub_group().get_local_linear_id())
          : 0,
      sycl::ext::oneapi::plus<>());
  if (!thread_active) return;

  auto labelled_group = active_labeled_partition(mask, e->path_idx, item_ct1);

  for (int64_t row_idx = start_row; row_idx < end_row; row_idx++) {
    for (auto condition_rank = 1ull; condition_rank < labelled_group.size();
         condition_rank++) {
      e = &s_elements[item_ct1.get_local_id(2)];
      // Compute the diagonal terms
      // TODO(Rory): this can be more efficient
      float reduce_input =
          e->IsRoot() || labelled_group.thread_rank() == condition_rank
              ? 1.0f
              : e->zero_fraction;
      float reduce =
          labelled_group.reduce(reduce_input, std::multiplies<float>());
      if (labelled_group.thread_rank() == condition_rank) {
        float one_fraction = e->split_condition.EvaluateSplit(
            X.GetElement(row_idx, e->feature_idx));
        auto phi_offset =
            IndexPhiInteractions(row_idx, num_groups, e->group, X.NumCols(),
                                 e->feature_idx, e->feature_idx);
        atomicAddDouble(phis_interactions + phi_offset,
                        reduce * (one_fraction - e->zero_fraction) * e->v);
      }

      int64_t condition_feature =
          labelled_group.shfl(e->feature_idx, condition_rank);

      SwapConditionedElement(&e, s_elements, condition_rank, labelled_group,
                             item_ct1);

      float x = ComputePhiCondition<TaylorGroupPath>(
          *e, row_idx, X, labelled_group, condition_feature, item_ct1);
      if (!e->IsRoot()) {
        auto phi_offset =
            IndexPhiInteractions(row_idx, num_groups, e->group, X.NumCols(),
                                 e->feature_idx, condition_feature);
        atomicAddDouble(phis_interactions + phi_offset, x);
      }
    }
  }
}

template <typename DatasetT, typename SizeTAllocatorT, typename PathAllocatorT,
          typename SplitConditionT>
void ComputeShapTaylorInteractions(
    DatasetT X,
    const dpct::device_vector<size_t, SizeTAllocatorT> &bin_segments,
    const dpct::device_vector<PathElement<SplitConditionT>, PathAllocatorT>
        &path_elements,
    size_t num_groups, double *phis) {
  size_t bins_per_row = bin_segments.size() - 1;
  const int kBlockThreads = GPUTREESHAP_MAX_THREADS_PER_BLOCK;
  const int warps_per_block = kBlockThreads / 32;
  const int kRowsPerWarp = 100;
  size_t warps_needed = bins_per_row * DivRoundUp(X.NumRows(), kRowsPerWarp);

  const uint32_t grid_size = DivRoundUp(warps_needed, warps_per_block);

  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      sycl::local_accessor<DatasetT, 0> s_X_acc_ct1(cgh);
      sycl::local_accessor<
          PathElement<SplitConditionT>, 1>
          s_elements_acc_ct1(sycl::range<1>(kBlockThreads), cgh);

      auto path_elements_data_get_ct2 = path_elements.data().get();
      auto bin_segments_data_get_ct3 = bin_segments.data().get();

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                sycl::range<3>(1, 1, kBlockThreads),
                            sycl::range<3>(1, 1, kBlockThreads)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            ShapTaylorInteractionsKernel<DatasetT, kBlockThreads, kRowsPerWarp>(
                X, bins_per_row, path_elements_data_get_ct2,
                bin_segments_data_get_ct3, num_groups, phis, item_ct1,
                s_X_acc_ct1,
                (PathElement<SplitConditionT> *)
                    s_elements_acc_ct1.get_pointer());
          });
    });
  }
}


inline int64_t Factorial(int64_t x) {
  int64_t y = 1;
  for (auto i = 2; i <= x; i++) {
    y *= i;
  }
  return y;
}

// Compute factorials in log space using lgamma to avoid overflow
inline double W(double s, double n) {
  assert(n - s - 1 >= 0);
  return sycl::exp(sycl::lgamma(s + 1) - sycl::lgamma(n + 1) +
                   sycl::lgamma(n - s));
}

template <typename DatasetT, size_t kBlockSize, size_t kRowsPerWarp,
          typename SplitConditionT>
void 
    ShapInterventionalKernel(DatasetT X, DatasetT R, size_t bins_per_row,
                             const PathElement<SplitConditionT>* path_elements,
                             const size_t* bin_segments, size_t num_groups,
                             double* phis, const sycl::nd_item<3> &item_ct1,
                             sycl::local_accessor<float, 2> s_W,
                             PathElement<SplitConditionT> *s_elements) {
  // Cache W coefficients

  for (int i = item_ct1.get_local_id(2); i < 33 * 33; i += kBlockSize) {
    auto s = i % 33;
    auto n = i / 33;
    if (n - s - 1 >= 0) {
      s_W[s][n] = W(s, n);
    } else {
      s_W[s][n] = 0.0;
    }
  }

  /*
  DPCT1065:16: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  PathElement<SplitConditionT> &e = s_elements[item_ct1.get_local_id(2)];

  size_t start_row, end_row;
  bool thread_active;
  ConfigureThread<DatasetT, kBlockSize, kRowsPerWarp>(
      X, bins_per_row, path_elements, bin_segments, &start_row, &end_row, &e,
      &thread_active, item_ct1);

  uint32_t mask = sycl::reduce_over_group(
      item_ct1.get_sub_group(),
      (FULL_MASK & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
              thread_active
          ? (0x1 << item_ct1.get_sub_group().get_local_linear_id())
          : 0,
      sycl::ext::oneapi::plus<>());
  if (!thread_active) return;

  auto labelled_group = active_labeled_partition(mask, e.path_idx, item_ct1);

  for (int64_t x_idx = start_row; x_idx < end_row; x_idx++) {
    float result = 0.0f;
    bool x_cond = e.EvaluateSplit(X, x_idx);
    uint32_t x_ballot = labelled_group.ballot(x_cond, item_ct1);
    for (int64_t r_idx = 0; r_idx < R.NumRows(); r_idx++) {
      bool r_cond = e.EvaluateSplit(R, r_idx);
      uint32_t r_ballot = labelled_group.ballot(r_cond, item_ct1);
      /*
      DPCT1007:17: Migration of __assert_fail is not supported.
      */
      assert(!e.IsRoot() ||
             (x_cond == r_cond)); // These should be the same for the root
      uint32_t s = sycl::popcount(x_ballot & ~r_ballot);
      uint32_t n = sycl::popcount(x_ballot ^ r_ballot);
      float tmp = 0.0f;
      // Theorem 1
      if (x_cond && !r_cond) {
        tmp += s_W[s - 1][n];
      }
      tmp -= s_W[s][n] * (r_cond && !x_cond);

      // No foreground samples make it to this leaf, increment bias
      if (e.IsRoot() && s == 0) {
        tmp += 1.0f;
      }
      // If neither foreground or background go down this path, ignore this path
      bool reached_leaf = !labelled_group.ballot(!x_cond && !r_cond, item_ct1);
      tmp *= reached_leaf;
      result += tmp;
    }

    if (result != 0.0) {
      result /= R.NumRows();

      // Root writes bias
      auto feature = e.IsRoot() ? X.NumCols() : e.feature_idx;
      atomicAddDouble(
          &phis[IndexPhi(x_idx, num_groups, e.group, X.NumCols(), feature)],
          result * e.v);
    }
  }
}

template <typename DatasetT, typename SizeTAllocatorT, typename PathAllocatorT,
          typename SplitConditionT>
void ComputeShapInterventional(
    DatasetT X, DatasetT R,
    const dpct::device_vector<size_t, SizeTAllocatorT> &bin_segments,
    const dpct::device_vector<PathElement<SplitConditionT>, PathAllocatorT>
        &path_elements,
    size_t num_groups, double *phis) {
  size_t bins_per_row = bin_segments.size() - 1;
  const int kBlockThreads = GPUTREESHAP_MAX_THREADS_PER_BLOCK;
  const int warps_per_block = kBlockThreads / 32;
  const int kRowsPerWarp = 100;
  size_t warps_needed = bins_per_row * DivRoundUp(X.NumRows(), kRowsPerWarp);

  const uint32_t grid_size = DivRoundUp(warps_needed, warps_per_block);

  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 2> s_W_acc_ct1(sycl::range<2>(33, 33), cgh);
      sycl::local_accessor<
          PathElement<SplitConditionT>, 1>
          s_elements_acc_ct1(sycl::range<1>(kBlockThreads), cgh);

      auto path_elements_data_get_ct3 = path_elements.data().get();
      auto bin_segments_data_get_ct4 = bin_segments.data().get();

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                sycl::range<3>(1, 1, kBlockThreads),
                            sycl::range<3>(1, 1, kBlockThreads)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            ShapInterventionalKernel<DatasetT, kBlockThreads, kRowsPerWarp>(
                X, R, bins_per_row, path_elements_data_get_ct3,
                bin_segments_data_get_ct4, num_groups, phis, item_ct1,
                s_W_acc_ct1,
                (PathElement<SplitConditionT> *)
                    s_elements_acc_ct1.get_pointer());
          });
    });
  }
}

template <typename PathVectorT, typename SizeVectorT, typename DeviceAllocatorT>
void GetBinSegments(const PathVectorT &paths, const SizeVectorT &bin_map,
                    SizeVectorT *bin_segments) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  DeviceAllocatorT alloc;
  size_t num_bins =
      /*
      DPCT1107:18: Migration for this overload of thrust::reduce is not
      supported.
      */
      std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), bin_map.begin(), bin_map.end(), size_t(0), 
          oneapi::dpl::maximum<int>()) + 1;

  bin_segments->resize(num_bins + 1, 0);
  auto counting = dpct::make_counting_iterator(0llu);
  auto d_paths = paths.data().get();
  auto d_bin_segments = bin_segments->data().get();
  auto d_bin_map = bin_map.data();
  oneapi::dpl::for_each_n(
      oneapi::dpl::execution::make_device_policy(q_ct1), counting, paths.size(),
      [=](size_t idx) {
    auto path_idx = d_paths[idx].path_idx;
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        reinterpret_cast<unsigned long long *>(d_bin_segments) + // NOLINT
            d_bin_map[path_idx],
        1);
  });
  std::exclusive_scan(
      oneapi::dpl::execution::make_device_policy(q_ct1),
      bin_segments->begin(), bin_segments->end());//#,
      //(decltype(bin_segments->end())::value_type)bin_segments->begin());
}

struct DeduplicateKeyTransformOp {
  template <typename SplitConditionT>
  std::pair<size_t, int64_t> operator()(const PathElement<SplitConditionT> &e) {
    return {e.path_idx, e.feature_idx};
  }
};
//inline void CheckCuda(dpct::err0 err) {
  /*
  DPCT1000:20: Error handling if-stmt was detected but could not be rewritten.
  */
//  if (err != 0) {
    /*
    DPCT1001:19: The statement could not be removed.
    */
//    throw std::system_error(err, std::generic_category());
//  }
//}


inline void CheckCuda(dpct::err0 err) {
  try {
    if (err != 0) {
      throw std::system_error(err, std::generic_category());
    }
  }
    catch (const std::exception& e) {
        // print the exception
        std::cerr << "Exception "  << e.what() << std::endl;
    }
}

template <typename Return>
class DiscardOverload : public oneapi::dpl::discard_iterator {
 public:
  using value_type = Return;  // NOLINT
};

template <typename PathVectorT, typename DeviceAllocatorT,
          typename SplitConditionT>
void DeduplicatePaths(PathVectorT *device_paths,
                      PathVectorT *deduplicated_paths) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  DeviceAllocatorT alloc;
  // Sort by feature
  std::sort(oneapi::dpl::execution::make_device_policy(q_ct1), device_paths->begin(),
               device_paths->end(),
               [=] (const PathElement<SplitConditionT>& a,
                              const PathElement<SplitConditionT>& b) {
                 if (a.path_idx < b.path_idx) return true;
                 if (b.path_idx < a.path_idx) return false;

                 if (a.feature_idx < b.feature_idx) return true;
                 if (b.feature_idx < a.feature_idx) return false;
                 return false;
               });

  deduplicated_paths->resize(device_paths->size());

  using Pair = std::pair<size_t, int64_t>;
  auto key_transform = oneapi::dpl::make_transform_iterator(
      device_paths->begin(), DeduplicateKeyTransformOp());

  dpct::device_vector<size_t> d_num_runs_out(1);
  size_t* h_num_runs_out;
  CheckCuda(
      DPCT_CHECK_ERROR(h_num_runs_out = sycl::malloc_host<size_t>(1, q_ct1)));

  auto combine = [] (PathElement<SplitConditionT> a,
                               PathElement<SplitConditionT> b) {
    // Combine duplicate features
    a.split_condition.Merge(b.split_condition);
    a.zero_fraction *= b.zero_fraction;
    return a;
  };  // NOLINT
  size_t temp_size = 0;
  /*
  DPCT1027:3: The call to cub::DeviceReduce::ReduceByKey was replaced with 0
  because this call is redundant in SYCL.
  */
  CheckCuda(0);
  using TempAlloc = RebindVector<char, DeviceAllocatorT>;
  TempAlloc tmp(temp_size);
  CheckCuda(
      q_ct1
          .fill(d_num_runs_out.begin(),
                std::distance(
                    DiscardOverload<Pair>(),
                    oneapi::dpl::reduce_by_key(
                        oneapi::dpl::execution::device_policy(q_ct1),
                        key_transform, key_transform + device_paths->size(),
                        device_paths->begin(), DiscardOverload<Pair>(),
                        deduplicated_paths->begin(),
                        std::equal_to<typename std::iterator_traits<
                            decltype(key_transform)>::value_type>(),
                        combine)
                        .first),
                1)
          .wait());

  CheckCuda(DPCT_CHECK_ERROR(
      q_ct1.memcpy(h_num_runs_out, d_num_runs_out.data(), sizeof(size_t))
          .wait()));
  deduplicated_paths->resize(*h_num_runs_out);
  CheckCuda(DPCT_CHECK_ERROR(sycl::free(h_num_runs_out, q_ct1)));
}

template <typename PathVectorT, typename SplitConditionT, typename SizeVectorT,
          typename DeviceAllocatorT>
void SortPaths(PathVectorT* paths, const SizeVectorT& bin_map) {
  auto d_bin_map = bin_map.data();
  DeviceAllocatorT alloc;
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  //DeviceAllocatorT q_ct1;
  std::sort(oneapi::dpl::execution::make_device_policy(q_ct1), paths->begin(), paths->end(),
               [=] (const PathElement<SplitConditionT>& a,
                              const PathElement<SplitConditionT>& b) {
                 size_t a_bin = d_bin_map[a.path_idx];
                 size_t b_bin = d_bin_map[b.path_idx];
                 if (a_bin < b_bin) return true;
                 if (b_bin < a_bin) return false;

                 if (a.path_idx < b.path_idx) return true;
                 if (b.path_idx < a.path_idx) return false;

                 if (a.feature_idx < b.feature_idx) return true;
                 if (b.feature_idx < a.feature_idx) return false;
                 return false;
               });
}

using kv = std::pair<size_t, int>;

struct BFDCompare {
  bool operator()(const kv& lhs, const kv& rhs) const {
    if (lhs.second == rhs.second) {
      return lhs.first < rhs.first;
    }
    return lhs.second < rhs.second;
  }
};

// Best Fit Decreasing bin packing
// Efficient O(nlogn) implementation with balanced tree using std::set
template <typename IntVectorT>
std::vector<size_t> BFDBinPacking(const IntVectorT& counts,
                                  int bin_limit = 32) {
  std::vector<int> counts_host(counts);
  std::vector<kv> path_lengths(counts_host.size());
  for (auto i = 0ull; i < counts_host.size(); i++) {
    path_lengths[i] = {i, counts_host[i]};
  }

  std::sort(path_lengths.begin(), path_lengths.end(),
            [&](const kv& a, const kv& b) {
              std::greater<> op;
              return op(a.second, b.second);
            });

  // map unique_id -> bin
  std::vector<size_t> bin_map(counts_host.size());
  std::set<kv, BFDCompare> bin_capacities;
  bin_capacities.insert({bin_capacities.size(), bin_limit});
  for (auto pair : path_lengths) {
    int new_size = pair.second;
    auto itr = bin_capacities.lower_bound({0, new_size});
    // Does not fit in any bin
    if (itr == bin_capacities.end()) {
      size_t new_bin_idx = bin_capacities.size();
      bin_capacities.insert({new_bin_idx, bin_limit - new_size});
      bin_map[pair.first] = new_bin_idx;
    } else {
      kv entry = *itr;
      entry.second -= new_size;
      bin_map[pair.first] = entry.first;
      bin_capacities.erase(itr);
      bin_capacities.insert(entry);
    }
  }

  return bin_map;
}

// First Fit Decreasing bin packing
// Inefficient O(n^2) implementation
template <typename IntVectorT>
std::vector<size_t> FFDBinPacking(const IntVectorT& counts,
                                  int bin_limit = 32) {
  std::vector<int> counts_host(counts);
  std::vector<kv> path_lengths(counts_host.size());
  for (auto i = 0ull; i < counts_host.size(); i++) {
    path_lengths[i] = {i, counts_host[i]};
  }
  std::sort(path_lengths.begin(), path_lengths.end(),
            [&](const kv& a, const kv& b) {
              std::greater<> op;
              return op(a.second, b.second);
            });

  // map unique_id -> bin
  std::vector<size_t> bin_map(counts_host.size());
  std::vector<int> bin_capacities(path_lengths.size(), bin_limit);
  for (auto pair : path_lengths) {
    int new_size = pair.second;
    for (auto j = 0ull; j < bin_capacities.size(); j++) {
      int& capacity = bin_capacities[j];

      if (capacity >= new_size) {
        capacity -= new_size;
        bin_map[pair.first] = j;
        break;
      }
    }
  }

  return bin_map;
}

// Next Fit bin packing
// O(n) implementation
template <typename IntVectorT>
std::vector<size_t> NFBinPacking(const IntVectorT& counts, int bin_limit = 32) {
  std::vector<int> counts_host(counts);
  std::vector<size_t> bin_map(counts_host.size());
  size_t current_bin = 0;
  int current_capacity = bin_limit;
  for (auto i = 0ull; i < counts_host.size(); i++) {
    int new_size = counts_host[i];
    size_t path_idx = i;
    if (new_size <= current_capacity) {
      current_capacity -= new_size;
      bin_map[path_idx] = current_bin;
    } else {
      current_capacity = bin_limit - new_size;
      bin_map[path_idx] = ++current_bin;
    }
  }
  return bin_map;
}

template <typename DeviceAllocatorT, typename SplitConditionT,
          typename PathVectorT, typename LengthVectorT>
void GetPathLengths(const PathVectorT& device_paths,
                    LengthVectorT* path_lengths) {
  path_lengths->resize(
      static_cast<PathElement<SplitConditionT>>(device_paths.back()).path_idx +
          1,
      0);
  auto counting = dpct::make_counting_iterator(0llu);
  auto d_paths = device_paths.data().get();
  auto d_lengths = path_lengths->data().get();
  oneapi::dpl::for_each_n(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), counting, device_paths.size(), [=] (size_t idx) {
    auto path_idx = d_paths[idx].path_idx;
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        d_lengths + path_idx, 1ull);
  });
}

struct PathTooLongOp {
  size_t operator()(size_t length) { return length > 32; }
};

template <typename SplitConditionT>
struct IncorrectVOp {
  const PathElement<SplitConditionT>* paths;
  size_t operator()(size_t idx) {
    auto a = paths[idx - 1];
    auto b = paths[idx];
    return a.path_idx == b.path_idx && a.v != b.v;
  }
};

template <typename DeviceAllocatorT, typename SplitConditionT,
          typename PathVectorT, typename LengthVectorT>
void ValidatePaths(const PathVectorT& device_paths,
                   const LengthVectorT& path_lengths) {
  DeviceAllocatorT alloc;
  PathTooLongOp too_long_op;

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();

  auto invalid_length =
      /*
      DPCT1107:21: Migration for this overload of thrust::any_of is not
      supported.
      */
      std::any_of(oneapi::dpl::execution::make_device_policy(q_ct1), path_lengths.begin(),
                     path_lengths.end(), too_long_op);

  if (invalid_length) {
    throw std::invalid_argument("Tree depth must be < 32");
  }

  IncorrectVOp<SplitConditionT> incorrect_v_op{device_paths.data().get()};
  auto counting = oneapi::dpl::counting_iterator<size_t>(0);
  auto incorrect_v = oneapi::dpl::any_of(
      oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()),
      counting + 1, counting + device_paths.size(), incorrect_v_op);

  if (incorrect_v) {
    throw std::invalid_argument(
        "Leaf value v should be the same across a single path");
  }
}

template <typename DeviceAllocatorT, typename SplitConditionT,
          typename PathVectorT, typename SizeVectorT>
void PreprocessPaths(PathVectorT* device_paths, PathVectorT* deduplicated_paths,
                     SizeVectorT* bin_segments) {
  // Sort paths by length and feature
  detail::DeduplicatePaths<PathVectorT, DeviceAllocatorT, SplitConditionT>(
      device_paths, deduplicated_paths);
  using int_vector = RebindVector<int, DeviceAllocatorT>;
  int_vector path_lengths;
  detail::GetPathLengths<DeviceAllocatorT, SplitConditionT>(*deduplicated_paths,
                                                            &path_lengths);
  SizeVectorT device_bin_map = detail::BFDBinPacking(path_lengths);
  ValidatePaths<DeviceAllocatorT, SplitConditionT>(*deduplicated_paths,
                                                   path_lengths);
  detail::SortPaths<PathVectorT, SplitConditionT, SizeVectorT,
                    DeviceAllocatorT>(deduplicated_paths, device_bin_map);
  detail::GetBinSegments<PathVectorT, SizeVectorT, DeviceAllocatorT>(
      *deduplicated_paths, device_bin_map, bin_segments);
}

struct PathIdxTransformOp {
  template <typename SplitConditionT>
  size_t operator()(const PathElement<SplitConditionT>& e) {
    return e.path_idx;
  }
};

struct GroupIdxTransformOp {
  template <typename SplitConditionT>
  size_t operator()(const PathElement<SplitConditionT>& e) {
    return e.group;
  }
};

struct BiasTransformOp {
  template <typename SplitConditionT>
  double operator()(const PathElement<SplitConditionT>& e) {
    return e.zero_fraction * e.v;
  }
};

// While it is possible to compute bias in the primary kernel, we do it here
// using double precision to avoid numerical stability issues
template <typename PathVectorT, typename DoubleVectorT,
          typename DeviceAllocatorT, typename SplitConditionT>
void ComputeBias(const PathVectorT &device_paths, DoubleVectorT *bias) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  using double_vector = dpct::device_vector<
      double, typename DeviceAllocatorT::template rebind<double>::other>;
  PathVectorT sorted_paths(device_paths);
  DeviceAllocatorT alloc;
  // Make sure groups are contiguous
  std::sort(oneapi::dpl::execution::make_device_policy(q_ct1), sorted_paths.begin(),
               sorted_paths.end(),
               [=] (const PathElement<SplitConditionT>& a,
                              const PathElement<SplitConditionT>& b) {
                 if (a.group < b.group) return true;
                 if (b.group < a.group) return false;

                 if (a.path_idx < b.path_idx) return true;
                 if (b.path_idx < a.path_idx) return false;

                 return false;
               });
  // Combine zero fraction for all paths
  auto path_key = oneapi::dpl::make_transform_iterator(sorted_paths.begin(),
                                                       PathIdxTransformOp());
  PathVectorT combined(sorted_paths.size());
  auto combined_out = oneapi::dpl::reduce_by_segment(
      oneapi::dpl::execution::make_device_policy(q_ct1), path_key,
      path_key + sorted_paths.size(), sorted_paths.begin(),
      oneapi::dpl::discard_iterator(), combined.begin(),
      oneapi::dpl::equal_to<size_t>(),
      [=](PathElement<SplitConditionT> a,
          const PathElement<SplitConditionT> &b) {
        a.zero_fraction *= b.zero_fraction;
        return a;
      });
  size_t num_paths = combined_out.second - combined.begin();
  // Combine bias for each path, over each group
  using size_vector = dpct::device_vector<
      size_t, typename DeviceAllocatorT::template rebind<size_t>::other>;
  size_vector keys_out(num_paths);
  double_vector values_out(num_paths);
  auto group_key = oneapi::dpl::make_transform_iterator(combined.begin(),
                                                        GroupIdxTransformOp());
  auto values =
      oneapi::dpl::make_transform_iterator(combined.begin(), BiasTransformOp());

  auto out_itr = oneapi::dpl::reduce_by_segment(
      oneapi::dpl::execution::make_device_policy(q_ct1), group_key,
      group_key + num_paths, values, keys_out.begin(), values_out.begin());

  // Write result
  size_t n = out_itr.first - keys_out.begin();
  auto counting = dpct::make_counting_iterator(0llu);
  auto d_keys_out = keys_out.data().get();
  auto d_values_out = values_out.data().get();
  auto d_bias = bias->data().get();
  oneapi::dpl::for_each_n(oneapi::dpl::execution::seq, counting, n,
                          [=](size_t idx) {
    d_bias[d_keys_out[idx]] = d_values_out[idx];
                          });
}

};  // namespace detail

/*!
 * Compute feature contributions on the GPU given a set of unique paths through
 * a tree ensemble and a dataset. Uses device memory proportional to the tree
 * ensemble size.
 *
 * \exception std::invalid_argument Thrown when an invalid argument error
 * condition occurs. \tparam  PathIteratorT     Thrust type iterator, may be
 * thrust::device_ptr for device memory, or stl iterator/raw pointer for host
 * memory. \tparam  PhiIteratorT      Thrust type iterator, may be
 * thrust::device_ptr for device memory, or stl iterator/raw pointer for host
 * memory. Value type must be floating point. \tparam  DatasetT User-specified
 * dataset container. \tparam  DeviceAllocatorT  Optional thrust style
 * allocator.
 *
 * \param X           Thin wrapper over a dataset allocated in device memory. X
 * should be trivially copyable as a kernel parameter (i.e. contain only
 * pointers to actual data) and must implement the methods
 * NumRows()/NumCols()/GetElement(size_t row_idx, size_t col_idx) as __device__
 * functions. GetElement may return NaN where the feature value is missing.
 * \param begin       Iterator to paths, where separate paths are delineated by
 *                    PathElement.path_idx. Each unique path should contain 1
 * root with feature_idx = -1 and zero_fraction = 1.0. The ordering of path
 * elements inside a unique path does not matter - the result will be the same.
 * Paths may contain duplicate features. See the PathElement class for more
 * information. \param end         Path end iterator. \param num_groups  Number
 * of output groups. In multiclass classification the algorithm outputs feature
 * contributions per output class. \param phis_begin  Begin iterator for output
 * phis. \param phis_end    End iterator for output phis.
 */
template <typename DeviceAllocatorT = dpct::deprecated::usm_device_allocator<int>,
//template <typename DeviceAllocatorT = thrust::device_allocator<int>,
          typename DatasetT, typename PathIteratorT, typename PhiIteratorT>
void GPUTreeShap(DatasetT X, PathIteratorT begin, PathIteratorT end,
                 size_t num_groups, PhiIteratorT phis_begin,
                 PhiIteratorT phis_end) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  if (X.NumRows() == 0 || X.NumCols() == 0 || end - begin <= 0) return;

  if (size_t(phis_end - phis_begin) <
      X.NumRows() * (X.NumCols() + 1) * num_groups) {
    throw std::invalid_argument(
        "phis_out must be at least of size X.NumRows() * (X.NumCols() + 1) * "
        "num_groups");
  }

  using size_vector = detail::RebindVector<size_t, DeviceAllocatorT>;
  using double_vector = detail::RebindVector<double, DeviceAllocatorT>;
  using path_vector = detail::RebindVector<
      typename std::iterator_traits<PathIteratorT>::value_type,
      DeviceAllocatorT>;
  using split_condition =
      typename std::iterator_traits<PathIteratorT>::value_type::split_type;

  // Compute the global bias
  double_vector temp_phi(phis_end - phis_begin, 0.0);
  path_vector device_paths(begin, end);
  double_vector bias(num_groups, 0.0);
  detail::ComputeBias<path_vector, double_vector, DeviceAllocatorT,
                      split_condition>(device_paths, &bias);
  auto d_bias = bias.data().get();
  auto d_temp_phi = temp_phi.data().get();
  oneapi::dpl::for_each_n(oneapi::dpl::execution::make_device_policy(q_ct1),
                          dpct::make_counting_iterator(0llu),
                          X.NumRows() * num_groups, [=](size_t idx) {
                       size_t group = idx % num_groups;
                       size_t row_idx = idx / num_groups;
                       d_temp_phi[IndexPhi(row_idx, num_groups, group,
                                           X.NumCols(), X.NumCols())] +=
                           d_bias[group];
                          });

  path_vector deduplicated_paths;
  size_vector device_bin_segments;
  detail::PreprocessPaths<DeviceAllocatorT, split_condition>(
      &device_paths, &deduplicated_paths, &device_bin_segments);

  detail::ComputeShap(X, device_bin_segments, deduplicated_paths, num_groups,
                      temp_phi.data().get());
  std::copy(oneapi::dpl::execution::make_device_policy(q_ct1), temp_phi.begin(),
            temp_phi.end(), phis_begin);
}

/*!
 * Compute feature interaction contributions on the GPU given a set of unique
 * paths through a tree ensemble and a dataset. Uses device memory
 * proportional to the tree ensemble size.
 *
 * \exception std::invalid_argument Thrown when an invalid argument error
 *                                  condition occurs.
 * \tparam  DeviceAllocatorT  Optional thrust style allocator.
 * \tparam  DatasetT          User-specified dataset container.
 * \tparam  PathIteratorT     Thrust type iterator, may be thrust::device_ptr
 *                            for device memory, or stl iterator/raw pointer for
 *                            host memory.
 * \tparam  PhiIteratorT      Thrust type iterator, may be thrust::device_ptr
 *                            for device memory, or stl iterator/raw pointer for
 *                            host memory. Value type must be floating point.
 *
 * \param X           Thin wrapper over a dataset allocated in device memory. X
 *                    should be trivially copyable as a kernel parameter (i.e.
 *                    contain only pointers to actual data) and must implement
 *                    the methods NumRows()/NumCols()/GetElement(size_t row_idx,
 *                    size_t col_idx) as __device__ functions. GetElement may
 *                    return NaN where the feature value is missing.
 * \param begin       Iterator to paths, where separate paths are delineated by
 *                    PathElement.path_idx. Each unique path should contain 1
 *                    root with feature_idx = -1 and zero_fraction = 1.0. The
 *                    ordering of path elements inside a unique path does not
 *                    matter - the result will be the same. Paths may contain
 *                    duplicate features. See the PathElement class for more
 *                    information.
 * \param end         Path end iterator.
 * \param num_groups  Number of output groups. In multiclass classification the
 *                    algorithm outputs feature contributions per output class.
 * \param phis_begin  Begin iterator for output phis.
 * \param phis_end    End iterator for output phis.
 */
// switching usm here cause 29 more errors
template <typename DeviceAllocatorT = dpct::deprecated::usm_device_allocator<int>,
//template <typename DeviceAllocatorT = thrust::device_allocator<int>,
          typename DatasetT, typename PathIteratorT, typename PhiIteratorT>
void GPUTreeShapInteractions(DatasetT X, PathIteratorT begin, PathIteratorT end,
                             size_t num_groups, PhiIteratorT phis_begin,
                             PhiIteratorT phis_end) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  if (X.NumRows() == 0 || X.NumCols() == 0 || end - begin <= 0) return;
  if (size_t(phis_end - phis_begin) <
      X.NumRows() * (X.NumCols() + 1) * (X.NumCols() + 1) * num_groups) {
    throw std::invalid_argument(
        "phis_out must be at least of size X.NumRows() * (X.NumCols() + 1)  * "
        "(X.NumCols() + 1) * "
        "num_groups");
  }

  using size_vector = detail::RebindVector<size_t, DeviceAllocatorT>;
  using double_vector = detail::RebindVector<double, DeviceAllocatorT>;
  using path_vector = detail::RebindVector<
      typename std::iterator_traits<PathIteratorT>::value_type,
      DeviceAllocatorT>;
  using split_condition =
      typename std::iterator_traits<PathIteratorT>::value_type::split_type;

  // Compute the global bias
  double_vector temp_phi(phis_end - phis_begin, 0.0);
  path_vector device_paths(begin, end);
  double_vector bias(num_groups, 0.0);
  detail::ComputeBias<path_vector, double_vector, DeviceAllocatorT,
                      split_condition>(device_paths, &bias);
  auto d_bias = bias.data().get();
  auto d_temp_phi = temp_phi.data().get();
  oneapi::dpl::for_each_n(oneapi::dpl::execution::make_device_policy(q_ct1),
                          dpct::make_counting_iterator(0llu),
                          X.NumRows() * num_groups, [=](size_t idx) {
        size_t group = idx % num_groups;
        size_t row_idx = idx / num_groups;
        d_temp_phi[IndexPhiInteractions(row_idx, num_groups, group, X.NumCols(),
                                        X.NumCols(), X.NumCols())] +=
            d_bias[group];
                          });

  path_vector deduplicated_paths;
  size_vector device_bin_segments;
  detail::PreprocessPaths<DeviceAllocatorT, split_condition>(
      &device_paths, &deduplicated_paths, &device_bin_segments);

  detail::ComputeShapInteractions(X, device_bin_segments, deduplicated_paths,
                                  num_groups, temp_phi.data().get());
  std::copy(oneapi::dpl::execution::make_device_policy(q_ct1), temp_phi.begin(),
            temp_phi.end(), phis_begin);
}

/*!
 * Compute feature interaction contributions using the Shapley Taylor index on
 * the GPU, given a set of unique paths through a tree ensemble and a dataset.
 * Uses device memory proportional to the tree ensemble size.
 *
 * \exception std::invalid_argument Thrown when an invalid argument error
 *                                  condition occurs.
 * \tparam  PhiIteratorT      Thrust type iterator, may be thrust::device_ptr
 *                            for device memory, or stl iterator/raw pointer for
 *                            host memory. Value type must be floating point.
 * \tparam  PathIteratorT     Thrust type iterator, may be thrust::device_ptr
 *                            for device memory, or stl iterator/raw pointer for
 *                            host memory.
 * \tparam  DatasetT          User-specified dataset container.
 * \tparam  DeviceAllocatorT  Optional thrust style allocator.
 *
 * \param X           Thin wrapper over a dataset allocated in device memory. X
 *                    should be trivially copyable as a kernel parameter (i.e.
 *                    contain only pointers to actual data) and must implement
 *                    the methods NumRows()/NumCols()/GetElement(size_t row_idx,
 *                    size_t col_idx) as __device__ functions. GetElement may
 *                    return NaN where the feature value is missing.
 * \param begin       Iterator to paths, where separate paths are delineated by
 *                    PathElement.path_idx. Each unique path should contain 1
 *                    root with feature_idx = -1 and zero_fraction = 1.0. The
 *                    ordering of path elements inside a unique path does not
 *                    matter - the result will be the same. Paths may contain
 *                    duplicate features. See the PathElement class for more
 *                    information.
 * \param end         Path end iterator.
 * \param num_groups  Number of output groups. In multiclass classification the
 *                    algorithm outputs feature contributions per output class.
 * \param phis_begin  Begin iterator for output phis.
 * \param phis_end    End iterator for output phis.
 */
template <typename DeviceAllocatorT = dpct::deprecated::usm_device_allocator<int>,
//template <typename DeviceAllocatorT = thrust::device_allocator<int>,
          typename DatasetT, typename PathIteratorT, typename PhiIteratorT>
void GPUTreeShapTaylorInteractions(DatasetT X, PathIteratorT begin,
                                   PathIteratorT end, size_t num_groups,
                                   PhiIteratorT phis_begin,
                                   PhiIteratorT phis_end) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  using phis_type = typename std::iterator_traits<PhiIteratorT>::value_type;
  static_assert(std::is_floating_point<phis_type>::value,
                "Phis type must be floating point");

  if (X.NumRows() == 0 || X.NumCols() == 0 || end - begin <= 0) return;

  if (size_t(phis_end - phis_begin) <
      X.NumRows() * (X.NumCols() + 1) * (X.NumCols() + 1) * num_groups) {
    throw std::invalid_argument(
        "phis_out must be at least of size X.NumRows() * (X.NumCols() + 1)  * "
        "(X.NumCols() + 1) * "
        "num_groups");
  }

  using size_vector = detail::RebindVector<size_t, DeviceAllocatorT>;
  using double_vector = detail::RebindVector<double, DeviceAllocatorT>;
  using path_vector = detail::RebindVector<
      typename std::iterator_traits<PathIteratorT>::value_type,
      DeviceAllocatorT>;
  using split_condition =
      typename std::iterator_traits<PathIteratorT>::value_type::split_type;

  // Compute the global bias
  double_vector temp_phi(phis_end - phis_begin, 0.0);
  path_vector device_paths(begin, end);
  double_vector bias(num_groups, 0.0);
  detail::ComputeBias<path_vector, double_vector, DeviceAllocatorT,
                      split_condition>(device_paths, &bias);
  auto d_bias = bias.data().get();
  auto d_temp_phi = temp_phi.data().get();
  oneapi::dpl::for_each_n(oneapi::dpl::execution::make_device_policy(q_ct1),
                          dpct::make_counting_iterator(0llu),
                          X.NumRows() * num_groups, [=](size_t idx) {
        size_t group = idx % num_groups;
        size_t row_idx = idx / num_groups;
        d_temp_phi[IndexPhiInteractions(row_idx, num_groups, group, X.NumCols(),
                                        X.NumCols(), X.NumCols())] +=
            d_bias[group];
                          });

  path_vector deduplicated_paths;
  size_vector device_bin_segments;
  detail::PreprocessPaths<DeviceAllocatorT, split_condition>(
      &device_paths, &deduplicated_paths, &device_bin_segments);

  detail::ComputeShapTaylorInteractions(X, device_bin_segments,
                                        deduplicated_paths, num_groups,
                                        temp_phi.data().get());
  std::copy(oneapi::dpl::execution::make_device_policy(q_ct1), temp_phi.begin(),
            temp_phi.end(), phis_begin);
}

/*!
 * Compute feature contributions on the GPU given a set of unique paths through a tree ensemble
 * and a dataset. Uses device memory proportional to the tree ensemble size. This variant
 * implements the interventional tree shap algorithm described here:
 * https://drafts.distill.pub/HughChen/its_blog/
 *
 * It requires a background dataset R.
 *
 * \exception std::invalid_argument Thrown when an invalid argument error condition occurs.
 * \tparam  DeviceAllocatorT  Optional thrust style allocator.
 * \tparam  DatasetT          User-specified dataset container.
 * \tparam  PathIteratorT     Thrust type iterator, may be thrust::device_ptr for device memory, or
 *                            stl iterator/raw pointer for host memory.
 *
 * \param X           Thin wrapper over a dataset allocated in device memory. X should be trivially
 *                    copyable as a kernel parameter (i.e. contain only pointers to actual data) and
 *                    must implement the methods NumRows()/NumCols()/GetElement(size_t row_idx,
 *                    size_t col_idx) as __device__ functions. GetElement may return NaN where the
 *                    feature value is missing.
 * \param R           Background dataset.
 * \param begin       Iterator to paths, where separate paths are delineated by
 *                    PathElement.path_idx. Each unique path should contain 1 root with feature_idx =
 *                    -1 and zero_fraction = 1.0. The ordering of path elements inside a unique path
 *                    does not matter - the result will be the same. Paths may contain duplicate
 *                    features. See the PathElement class for more information.
 * \param end         Path end iterator.
 * \param num_groups  Number of output groups. In multiclass classification the algorithm outputs
 *                    feature contributions per output class.
 * \param phis_begin  Begin iterator for output phis.
 * \param phis_end    End iterator for output phis.
 */
template <typename DeviceAllocatorT = dpct::deprecated::usm_device_allocator<int>,
//template <typename DeviceAllocatorT = thrust::device_allocator<int>,
          typename DatasetT, typename PathIteratorT, typename PhiIteratorT>
void GPUTreeShapInterventional(DatasetT X, DatasetT R, PathIteratorT begin,
                               PathIteratorT end, size_t num_groups,
                               PhiIteratorT phis_begin, PhiIteratorT phis_end) {
  if (X.NumRows() == 0 || X.NumCols() == 0 || end - begin <= 0) return;

  if (size_t(phis_end - phis_begin) <
      X.NumRows() * (X.NumCols() + 1) * num_groups) {
    throw std::invalid_argument(
        "phis_out must be at least of size X.NumRows() * (X.NumCols() + 1) * "
        "num_groups");
  }

  using size_vector = detail::RebindVector<size_t, DeviceAllocatorT>;
  using double_vector = detail::RebindVector<double, DeviceAllocatorT>;
  using path_vector = detail::RebindVector<
      typename std::iterator_traits<PathIteratorT>::value_type,
      DeviceAllocatorT>;
  using split_condition =
      typename std::iterator_traits<PathIteratorT>::value_type::split_type;

  double_vector temp_phi(phis_end - phis_begin, 0.0);
  path_vector device_paths(begin, end);

  path_vector deduplicated_paths;
  size_vector device_bin_segments;
  detail::PreprocessPaths<DeviceAllocatorT, split_condition>(
      &device_paths, &deduplicated_paths, &device_bin_segments);
  detail::ComputeShapInterventional(X, R, device_bin_segments,
                                    deduplicated_paths, num_groups,
                                    temp_phi.data().get());
  std::copy(
      oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()),
      temp_phi.begin(), temp_phi.end(), phis_begin);
}
}  // namespace gpu_treeshap
