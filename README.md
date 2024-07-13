# Learning Data Distribution from Queries

## Basic Usage

### Single Table

- 1-input model for "<="

```bash
python SingleTable.py --model 1-input --dataset wine3 --query-size 100000 --min-conditions 1 --max-conditions 3 --lattice-size 2
```

- 2-input model for "<, <=, >, >=, ="

```bash
python SingleTable.py --model 2-input --dataset wine2 --query-size 1000 --min-conditions 1 --max-conditions 2 --lattice-size 2
```

### LPALG (PGM)
```bash
python LPALG.py --dataset wine --query-size 5 --num-conditions 2
```

## Code Description

- LPALG.py           ----- Baseline of LPALG (PGM)
- SingleTable.py     ----- Single table generator
- query_func.py      ----- query related functions (rewrite by numpy version)
- process_dataset.py ----- load and sort dataframe columns by unique number
- models.py          ----- models
