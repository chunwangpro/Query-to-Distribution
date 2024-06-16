# Learning Data Distribution from Queries

## Basic Usage

### Single Table

- 1-input model for "<="

```bash
python PWLLattice.py --dataset wine2 --query-size 100000 --min-conditions 1 --max-conditions 2 --epochs 2000 --bs 10000 --loss MSE
```

- 2-input model for "<, <=, >, >=, ="

```bash
python PWLLatticeCopula_legacy.py --dataset wine3 --query-size 1000 --num-conditions 2 --epochs 1000
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
