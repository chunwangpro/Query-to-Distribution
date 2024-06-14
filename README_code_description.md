# 代码说明


## LPALG(PGM)
- LPALG.py                   --- Baseline of LPALG (PGM)
- LPALGv1.0.ipynb            --- Baseline of LPALG (PGM)的第一版
- LPALGv2.0.ipynb            --- Baseline of LPALG (PGM)的第二版，与最终版（LPALG.py）一致


## query related functions
- query_func.py              --- query related functions (rewrite by numpy)
- common.py / datasets.py / estimators.py  --- legacy vertion using CSVTable and Oracle


## models
- models.py                  --- models
- LatticeCDF.py              --- legacy version(...)


## PWLLattice (1-input, <, <=)
- PWLLattice.py              --- PWL-Lattice 网络(速度更快的numpy版本)
- PWLLattice_legacy.py       --- PWL-Lattice 网络(较早的common.CSVTable版本)
- PWLLattice_test.ipynb      --- PWL-Lattice 网络的测试文件(用%reload加载要测试的文件)


## Plotting
- draw_copula.py             --- 绘制 二维 Copula function, 上下界
- draw_cdf_comparison.ipynb  --- 绘制二维和三维分布的 PDF、CDF
- PWLLattice_legacy.py       --- 绘制模型output单调递增/二维立体Lattice/二维平面Lattice/query对区间网格的覆盖率
