#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('/home/haoyu/_database/projs/ccc-gpu/libs')

from ccc.coef.impl import cdist_parts_basic, compute_ccc, get_parts, ccc

# Test what the CPU implementation returns for various edge cases
print("Testing CPU behavior:")

# Case 1: Normal partitions - should work fine
print("\n=== Case 1: Normal partitions ===")
x_parts = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y_parts = np.array([[0, 1, 0, 1], [1, 1, 0, 0]])
result = cdist_parts_basic(x_parts, y_parts)
print(f"cdist_parts_basic result:\n{result}")
ccc_result = compute_ccc(x_parts, y_parts, cdist_parts_basic)
print(f"compute_ccc result: {ccc_result}")

# Case 2: Partitions with -1 values (should be skipped)
print("\n=== Case 2: Partitions with -1 values ===")
x_parts = np.array([[-1, -1, -1, -1], [0, 1, 0, 1]])
y_parts = np.array([[0, 1, 0, 1], [1, 1, 0, 0]])
result = cdist_parts_basic(x_parts, y_parts)
print(f"cdist_parts_basic result:\n{result}")
ccc_result = compute_ccc(x_parts, y_parts, cdist_parts_basic)
print(f"compute_ccc result: {ccc_result}")

# Case 3: All partitions with -1 values
print("\n=== Case 3: All partitions with -1 values ===")
x_parts = np.array([[-1, -1, -1, -1], [-1, -1, -1, -1]])
y_parts = np.array([[-1, -1, -1, -1], [-1, -1, -1, -1]])
result = cdist_parts_basic(x_parts, y_parts)
print(f"cdist_parts_basic result:\n{result}")
ccc_result = compute_ccc(x_parts, y_parts, cdist_parts_basic)
print(f"compute_ccc result: {ccc_result}")

# Case 4: Test with actual CCC function using constant data
print("\n=== Case 4: CCC with constant data ===")
x = np.array([1.0, 1.0, 1.0, 1.0])
y = np.array([2.0, 2.0, 2.0, 2.0])
result = ccc(x, y, return_parts=True)
print(f"CCC result: {result[0]}")
print(f"Max parts: {result[1]}")
print(f"Parts shape: {result[2].shape}")
print(f"X parts:\n{result[2][0]}")
print(f"Y parts:\n{result[2][1]}")

# Case 5: Test with mixed data
print("\n=== Case 5: CCC with mixed data ===")
x = np.array([1.0, 1.0, 1.0, 1.0])
y = np.array([1.0, 2.0, 3.0, 4.0])
result = ccc(x, y, return_parts=True)
print(f"CCC result: {result[0]}")
print(f"Max parts: {result[1]}")
print(f"X parts:\n{result[2][0]}")
print(f"Y parts:\n{result[2][1]}")