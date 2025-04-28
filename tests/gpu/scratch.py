import numpy as np

from ccc.sklearn.metrics import adjusted_rand_index

# n_samples = 5000
#
# in1 = np.random.randint(0, 100, size=(n_samples,), dtype=np.int32)
# in2 = np.random.randint(0, 100, size=(n_samples,), dtype=np.int32)
#
# output_gpu = adjusted_rand_score(in1, in2)
# output_cpu = adjusted_rand_index(in1, in2)
#
# print(output_gpu)
# print(output_cpu)

output_cpu = adjusted_rand_index(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
print(output_cpu)

output_cpu = adjusted_rand_index(np.array([1, 1, 0, 0]), np.array([2, 1, 2, 0]))
print(output_cpu)
