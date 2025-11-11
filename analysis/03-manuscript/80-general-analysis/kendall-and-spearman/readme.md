# Kendall vs Spearman Correlation Comparison

This analysis compares Kendall's tau and Spearman's rho correlation coefficients to justify focusing exclusively on Spearman's correlation in the manuscript.

## Notebooks

### 1. `correlation_comparison_random_data.ipynb`
Generates 796 synthetic datasets across diverse scenarios:
- Linear correlations with varying strengths and sample sizes
- Linear relationships with different noise levels
- Nonlinear monotonic transformations (exponential, logarithmic, polynomial, etc.)
- Data with varying percentages of ties
- Edge cases and replicates for robustness testing

**Key Finding**: Pearson r = 0.9941, Spearman r = 0.9991 between Kendall and Spearman measures

### 2. `correlation_comparison_gtex.ipynb`
Validates synthetic findings using real GTEx v8 whole blood gene expression data (5,000 randomly sampled genes, 12.5M gene pairs).

**Key Findings**:
- Pearson r = 0.9972, Spearman r = 0.9998 between Kendall and Spearman measures
- Kendall is ~539x slower computationally (35.95 min vs 4 sec)

## Conclusion

Both rank-based correlation methods are highly redundant (>0.99 correlation). Since Spearman's rho is computationally efficient and provides equivalent information, the manuscript focuses on Spearman's correlation alone.
