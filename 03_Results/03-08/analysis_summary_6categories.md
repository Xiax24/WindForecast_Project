# 风切变和昼夜条件下预测策略性能分析报告

## 1. 数据概况
- 总数据量: 10233 条
- Low Shear Day: 3161 条 (30.9%)
- Low Shear Night: 2507 条 (24.5%)
- Moderate Shear Day: 1640 条 (16.0%)
- Moderate Shear Night: 2278 条 (22.3%)
- High Shear Day: 324 条 (3.2%)
- High Shear Night: 323 条 (3.2%)

## 2. 各风切变和昼夜条件下的最优策略
### Low Shear Day
- 最优策略: Fusion-M2 (RMSE: 27.7395)
- 最差策略: G-M1-70m (RMSE: 30.8386)
- 性能差距: 11.2%

### Low Shear Night
- 最优策略: Fusion-M2 (RMSE: 32.6031)
- 最差策略: G-M1-70m (RMSE: 36.2219)
- 性能差距: 11.1%

### Moderate Shear Day
- 最优策略: Fusion-M2 (RMSE: 30.0020)
- 最差策略: G-M1-10m (RMSE: 32.1463)
- 性能差距: 7.1%

### Moderate Shear Night
- 最优策略: Fusion-M2 (RMSE: 32.0296)
- 最差策略: G-M1-70m (RMSE: 35.3497)
- 性能差距: 10.4%

### High Shear Day
- 最优策略: Fusion-M2 (RMSE: 25.2622)
- 最差策略: E-M1-10m (RMSE: 29.7758)
- 性能差距: 17.9%

### High Shear Night
- 最优策略: Fusion-M2 (RMSE: 27.7706)
- 最差策略: E-M2-70m (RMSE: 31.2231)
- 性能差距: 12.4%

## 3. 昼夜对比分析
### 各风切变条件下的昼夜差异:
#### Low Shear
- 白天最优: Fusion-M2 (RMSE: 27.7395)
- 夜间最优: Fusion-M2 (RMSE: 32.6031)
- 昼夜差异: 4.8636

#### Moderate Shear
- 白天最优: Fusion-M2 (RMSE: 30.0020)
- 夜间最优: Fusion-M2 (RMSE: 32.0296)
- 昼夜差异: 2.0276

#### High Shear
- 白天最优: Fusion-M2 (RMSE: 25.2622)
- 夜间最优: Fusion-M2 (RMSE: 27.7706)
- 昼夜差异: 2.5084

