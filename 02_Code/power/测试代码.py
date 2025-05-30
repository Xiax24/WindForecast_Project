import joblib
import numpy as np

# 加载模型
model = joblib.load("/Users/xiaxin/work/WindForecast_Project/03_Results/03saved_models/best_lightgbm_model.pkl")
feature_names = joblib.load("/Users/xiaxin/work/WindForecast_Project/03_Results/03saved_models/feature_names.pkl")
model_info = joblib.load("/Users/xiaxin/work/WindForecast_Project/03_Results/03saved_models/model_info.pkl")

print("✅ 模型加载成功！")
print(f"特征数量: {len(feature_names)}")
print(f"测试集性能: R² = {model_info['performance']['test_r2']:.4f}")


# 快速验证
model = joblib.load("/Users/xiaxin/work/WindForecast_Project/03_Results/03saved_models/best_lightgbm_model.pkl")
print("模型类型:", type(model))
print("模型已可用于预测！")