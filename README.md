# Learning Data Distribution from Queries

## Basic Usage

### Single Table

```bash
python PWLLattice.py --dataset wine2 --query-size 100000 --max-conditions 2 --epochs 1000
```

### LPALG (PGM)
```bash
python LPALG.py --dataset wine --query-size 5 --num-conditions 2
```

## 代码说明

- LPALG.py          ----- Baseline of LPALG (PGM)
- PWLLattice.py     ----- PWL-Lattice 网络
- query_func.py     ----- query related functions (rewrite by numpy)
- models.py         ----- models
