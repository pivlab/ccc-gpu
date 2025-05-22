Preprocessing for the GTEx data was done in notebooks in the _glibio folder.

## Compute correlations
### Pearson
Commands:
```bash
nohup python ./compute_correlations_cli.py --method pearson > logs/gtex-var_pc_log2-pearson.terminal.log 2> logs/gtex-var_pc_log2-pearson.progress.log < /dev/null &
```

In total, 54 tissues were processed, and three logs were generated:
- `gtex-var_pc_log2-pearson.terminal.log`: Terminal output.
- `gtex-var_pc_log2-pearson.progress.log`: Progress output.
- `gtex-var_pc_log2-pearson.root.log`: Root output.

### Spearman
Commands:
```bash
nohup python 10_compute_correlations/compute_correlations_cli.py --method spearman > logs/gtex-var_pc_log2-spearman.terminal.log 2> logs/gtex-var_pc_log2-spearman.progress.log &
```
