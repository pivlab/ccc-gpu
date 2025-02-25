import numpy as np

from cuml.metrics.cluster.adjusted_rand_index import adjusted_rand_score
from ccc.sklearn.metrics import adjusted_rand_index

n_samples = 5000

in1 = np.random.randint(0, 100, size=(n_samples,), dtype=np.int32)
in2 = np.random.randint(0, 100, size=(n_samples,), dtype=np.int32)

output_gpu = adjusted_rand_score(in1, in2)
output_cpu = adjusted_rand_index(in1, in2)

print(output_gpu)
print(output_cpu)
