n_features = 10000
n_partitions = 9
n_objects = 1000

int_size = 1
d_parts_size = n_features * n_partitions * n_objects * int_size / 1024**3

n_feature_comp = n_features * (n_features - 1) // 2
n_aris = n_feature_comp * n_partitions * n_partitions
float_size = 4
d_cm_values_size = n_aris * float_size / 1024**3

print(f"n_features: {n_features}")
print(f"n_partitions: {n_partitions}")
print(f"n_objects: {n_objects}")
print(f"int_size: {int_size} bytes")
print(f"d_parts_size: {d_parts_size} GB")

print(f"n_feature_comp: {n_feature_comp}")
print(f"n_aris: {n_aris}")
print(f"float_size: {float_size} bytes")
print(f"d_cm_values_size: {d_cm_values_size} GB")
