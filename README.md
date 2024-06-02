# Learning Data Distribution from Queries
## Basic Usage
### Single Table

#### LPALG
```bash
python query2distributeLPALG.py --dataset wine --device cpu --query-size 5 --num-conditions 2 --lr 2e-4
```

#### 代码说明 ipynb

- LPALGv1.0     ---- 第一版稳定运行 Baseline
- LPALGv2.0     ---- 第二版稳定运行 Baseline
- Lattice           ----- 师姐提供的 Lattice 代码
- Lattice_wine_2    ----- 可视化代码
- Lattice_v1.0 ----- 第一版可以稳定运行
  - 只采用 input - PWL - lattice 的网络
- Lattice_wine_2-Copy1    -----  一些改进
