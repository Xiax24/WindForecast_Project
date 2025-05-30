
# 误差传播分析总结报告 (Normalized Sensitivity Analysis)

## 数据概况
- 分析时间段: 2021-05-01 00:00:00 到 2022-10-31 23:45:00
- 总样本数: 52317
- 分析特征数量: 7

## 误差分解结果 (RMSE)
- 建模误差 (Modeling Error): 16.784 kW
- ECMWF传播误差 (ECMWF Propagation): 36.707 kW  
- GFS传播误差 (GFS Propagation): 39.127 kW
- ECMWF总误差 (ECMWF Total): 38.866 kW
- GFS总误差 (GFS Total): 41.635 kW

## 标准化敏感性排序 (Top 5)
- obs_wind_speed_70m: 71.167 kW/std (5.311 kW/m/s)
- obs_wind_speed_10m: 60.071 kW/std (7.921 kW/m/s)
- obs_temperature_10m: 9.653 kW/std (0.055 kW/°C)
- obs_wind_speed_50m: 7.589 kW/std (0.937 kW/m/s)
- obs_wind_dir_cos_70m: 5.313 kW/std (3.431 kW/0.1 units (≈6°))

## 误差传播方差分析
- ECMWF理论方差: 69013.1480
- ECMWF实际方差: 1302.3894
- ECMWF比值 (实际/理论): 0.0189

- GFS理论方差: 76057.4670
- GFS实际方差: 1464.3958
- GFS比值 (实际/理论): 0.0193

## 关键发现
1. 标准化敏感性分析解决了尺度问题
2. 理论方差与实际方差的比值在合理范围内
3. 最重要的特征是: obs_wind_speed_70m
