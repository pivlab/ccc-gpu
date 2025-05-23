Preprocessing for the GTEx data was done in notebooks in the _glibio folder.

## Compute correlations
### CCC-GPU
Commands:
```bash
nohup python ./compute_correlations_cli.py --method ccc_gpu > logs/gtex-var_pc_log2-ccc_gpu.terminal.log 2> logs/gtex-var_pc_log2-ccc_gpu.progress.log < /dev/null &
```

### Pearson
Commands:
```bash
nohup python ./compute_correlations_cli.py --method pearson > logs/gtex-var_pc_log2-pearson.terminal.log 2> logs/gtex-var_pc_log2-pearson.progress.log < /dev/null &
```

### Spearman
Commands:
```bash
nohup python ./compute_correlations_cli.py --method spearman > logs/gtex-var_pc_log2-spearman.terminal.log 2> logs/gtex-var_pc_log2-spearman.progress.log < /dev/null &
```

In total, 54 tissues will be processed, and three logs will be generated, for each of the three methods:
- `gtex-var_pc_log2-{method}.terminal.log`: Terminal output.
- `gtex-var_pc_log2-{method}.progress.log`: Progress output.
- `gtex-var_pc_log2-{method}.root.log`: Root output.
