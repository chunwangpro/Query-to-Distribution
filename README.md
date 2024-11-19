# Learning Data Distribution from Queries

## Basic Usage

### Single Table



```bash
python SingleTable.py --dataset wine3 --query-size 10000 --min-conditions 1 --max-conditions 3 --cdf res
```


### Baseline

- LPALG (PGM)

```bash
python LPALG.py --dataset wine --query-size 5 --num-conditions 2
```

## Code Description

- LPALG.py           ----- Baseline of LPALG (PGM)
- SingleTable.py     ----- Single table generation
- util.py            ----- query related functions (rewrite by numpy version)
- dataset.py         ----- load and sort dataframe columns by unique number
- preprocessing.py   ----- build train set
- model.py           ----- models
- generator.py       ----- generation methods, by row, by col
