# Change Log: Fix GPU max_parts Bug

**Date:** July 7, 2025  
**Issue:** GPU implementation returning incorrect partition indices when `return_parts=True`  
**Severity:** High - Functional correctness bug  
**Files Modified:** `libs/ccc_cuda_ext/coef.cu`

## Issue Description

### Problem
The GPU implementation of CCC with `return_parts=True` was returning incorrect partition indices in the `max_parts` output. While the correlation coefficients were computed correctly, the indices identifying which partitions maximized the correlation were wrong.

### Symptoms
- Test `test_ccc_gpu_with_numerical_input` failing with assertion error
- GPU returning `[2, 8]` while CPU correctly returned `[8, 5]` for the same data
- Issue occurred specifically when ARI matrix contained mostly negative values with one small positive maximum

### Root Cause Analysis
The bug was in the `findMaxAriKernel` CUDA kernel's reduction logic:

1. **Two-step reduction problem**: The kernel used separate reductions for finding maximum value and corresponding index
2. **Race condition**: The association between ARI values and their indices was lost during the reduction process
3. **Floating-point comparison issues**: The kernel's approach to finding indices of threads with maximum values was flawed
4. **Edge case vulnerability**: Failed when dealing with mostly negative ARI values and one small positive maximum

## Technical Fix

### Changes Made
**File:** `libs/ccc_cuda_ext/coef.cu` - `findMaxAriKernel` function

#### Before (Problematic Implementation)
```cpp
// Two separate reductions - prone to race conditions
T max_block_val = cub::BlockReduce<T, 128>(temp_storage_val).Reduce(in.val, cub::Max());
uint64_t selected_idx = UINT64_MAX;
if (in.val == max_block_val) {
    selected_idx = in.idx;
}
uint64_t min_idx = cub::BlockReduce<uint64_t, 128>(temp_storage_idx).Reduce(selected_idx, custom_reducer);
```

#### After (Fixed Implementation)
```cpp
// Single atomic ArgMax reduction - maintains value-index association
typedef cub::KeyValuePair<int, T> KeyValuePairT;
KeyValuePairT thread_data;
thread_data.key = i;      // Local index
thread_data.value = val;  // ARI value

KeyValuePairT aggregate = BlockReduceT(temp_storage).Reduce(thread_data, cub::ArgMax());
```

### Key Technical Improvements

1. **Atomic ArgMax Reduction**: Replaced two-step reduction with CUB's `ArgMax` that atomically finds both maximum value and its index

2. **Proper Data Structure**: Used `cub::KeyValuePair<int, T>` to maintain tight coupling between values and indices

3. **Improved NaN Handling**: Enhanced NaN detection using shared memory communication across threads

4. **Race Condition Elimination**: Single reduction operation eliminates timing-dependent bugs

## Testing and Validation

### Test Cases Verified
- ✅ `test_ccc_gpu_with_numerical_input` - Previously failing, now passes
- ✅ All existing `return_parts` tests continue to pass
- ✅ Edge case with mostly negative ARIs and single positive maximum
- ✅ Simple cases with all positive ARI values
- ✅ GPU vs CPU output matching for `max_parts`

### Performance Impact
- **Positive**: Single reduction is more efficient than two separate reductions
- **Memory**: Slightly reduced shared memory usage
- **Correctness**: 100% accuracy improvement for `max_parts` results

## Future Considerations

### Code Review Guidelines
1. **Reduction Operations**: Always prefer atomic reductions (like `ArgMax`, `ArgMin`) over multi-step reductions when maintaining associations between data
2. **Floating-Point Comparisons**: Be extremely careful with exact equality comparisons (`==`) in CUDA kernels, especially after reductions
3. **Thread Synchronization**: Ensure proper `__syncthreads()` placement when using shared memory communication

### Testing Requirements
1. **Edge Cases**: Always test with datasets containing mostly negative correlations with small positive maximum
2. **Value-Index Association**: Verify that returned indices actually correspond to the reported maximum values
3. **Cross-Implementation Validation**: Compare GPU and CPU results for identical inputs, not just similar statistical properties

### Performance Monitoring
1. **Regression Testing**: Include performance benchmarks to ensure reduction optimizations don't introduce slowdowns
2. **Memory Usage**: Monitor shared memory usage as more complex reductions may require larger temporary storage

### Similar Patterns to Watch
- Any CUDA kernel performing multi-step reductions where data association matters
- Kernels using separate value and index reductions
- Functions returning both computed values and the parameters that produced them

## Dependencies Updated
- **Build System**: Rebuilt CUDA extension after kernel modification
- **Testing**: No test dependencies changed

## Migration Notes
- No API changes - fix is transparent to users
- Existing code using `return_parts=True` will now receive correct results
- No performance regressions expected

---
**Reviewed by:** [haoyu-zc]  
**Deployment Status:** ✅ Complete  
**Rollback Plan:** Revert `findMaxAriKernel` to previous implementation if unforeseen issues arise