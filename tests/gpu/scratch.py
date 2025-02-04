from sklearn.metrics.cluster import adjusted_rand_score

print(adjusted_rand_score([0, 0, 1, 1], [2, 1, 2, 0]))
print(adjusted_rand_score([1, 1, 0, 0], [0, 0, 1, 2]))
print(adjusted_rand_score([0, 1, 0, 1], [0, 0, 1, 2]))
