# Result Files

Place CIFAR-10 experiment summaries here as CSV files.

Expected schema for `scripts/plot_cifar10_results.py`:

```csv
method,prior,seed,accuracy
ResNet18,clean,0,0.0000
ResNet18,corrupted,0,0.0000
ResNet18,decontaminated,0,0.0000
```

Use real measured values only. The plotting script aggregates repeated seeds
with mean accuracy and standard error.
