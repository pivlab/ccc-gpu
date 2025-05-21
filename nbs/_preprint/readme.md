Preprocessing for the GTEx data was done in notebooks in the _glibio folder.

## Compute correlations
Commands:
```bash
nohup python 10_compute_correlations/05-01-gtex-var_pc_log2-ccc-gpu.py > 05-01-gtex-var_pc_log2-ccc-gpu.terminal.log 2> 05-01-gtex-var_pc_log2-ccc-gpu.progress.log &
```

In total, 54 tissues were processed, and three logs were generated:
- `05-01-gtex-var_pc_log2-ccc-gpu.terminal.log`: Terminal output.
- `05-01-gtex-var_pc_log2-ccc-gpu.progress.log`: Progress output.
- `05-01-gtex-var_pc_log2-ccc-gpu.root.log`: Root output.
