#!/usr/bin/env python3
"""
权重对比试验：测试不同权重分配策略的效果
"""

def run_weight_comparison_experiments(data_path, base_save_dir, indices_path):
    """运行权重对比试验"""
    
    print("=" * 80)
    print("🔬 权重对比试验：测试不同融合权重策略")
    print("=" * 80)
    
    # 定义不同的权重策略
    weight_strategies = [
        {
            'name': 'Fusion-Equal',
            'description': '四模型等权重融合 (各25%)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            'weights': [0.25, 0.25, 0.25, 0.25]  # 等权重
        },
        {
            'name': 'Fusion-Original',
            'description': '原始权重 (EC主导)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            'weights': [0.4, 0.2, 0.3, 0.1]  # 原始权重
        },
        {
            'name': 'Fusion-Performance',
            'description': '基于单模型性能的权重',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            # 基于你的试验结果：G-M2-70m(32.65) > G-M2-10m(32.93) > E-M2-10m(33.17) > E-M2-70m(33.84)
            # RMSE越小性能越好，权重应该越高
            'weights': [0.25, 0.15, 0.30, 0.30]  # GFS-70m最好给最高权重
        },
        {
            'name': 'Fusion-10m-Focus',
            'description': '偏向10m高度 (10m占70%)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            'weights': [0.35, 0.15, 0.35, 0.15]  # 10m占70%
        },
        {
            'name': 'Fusion-EC-Focus',
            'description': 'EC主导策略 (EC占70%)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            'weights': [0.45, 0.25, 0.20, 0.10]  # EC占70%
        },
        {
            'name': 'Fusion-GFS-Focus',
            'description': 'GFS主导策略 (GFS占70%)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            'weights': [0.15, 0.15, 0.35, 0.35]  # GFS占70%
        }
    ]
    
    # 存储所有结果
    comparison_results = []
    
    for i, exp_config in enumerate(weight_strategies, 1):
        print(f"\n权重策略 {i}/6: {exp_config['name']}")
        print(f"权重: {exp_config['weights']}")
        print(f"说明: {exp_config['description']}")
        
        try:
            # 运行试验
            save_dir = os.path.join(base_save_dir, 'weight_comparison', exp_config['name'])
            metrics = run_experiment(data_path, save_dir, indices_path, exp_config)
            
            # 记录结果
            result = {
                'strategy': exp_config['name'],
                'description': exp_config['description'],
                'weights': exp_config['weights'],
                'ec_10m_weight': exp_config['weights'][0],
                'ec_70m_weight': exp_config['weights'][1],
                'gfs_10m_weight': exp_config['weights'][2],
                'gfs_70m_weight': exp_config['weights'][3],
                'ec_total_weight': exp_config['weights'][0] + exp_config['weights'][1],
                'gfs_total_weight': exp_config['weights'][2] + exp_config['weights'][3],
                'm10_total_weight': exp_config['weights'][0] + exp_config['weights'][2],
                'm70_total_weight': exp_config['weights'][1] + exp_config['weights'][3],
                'RMSE': metrics.get('RMSE'),
                'Correlation': metrics.get('Correlation')
            }
            
            if 'error' in metrics:
                result['error'] = metrics['error']
            
            comparison_results.append(result)
            
            print(f"  结果: RMSE={result['RMSE']:.4f}, 相关系数={result['Correlation']:.4f}")
            
        except Exception as e:
            print(f"  ❌ 失败: {str(e)}")
            result = {
                'strategy': exp_config['name'],
                'description': exp_config['description'],
                'weights': exp_config['weights'],
                'RMSE': None,
                'Correlation': None,
                'error': str(e)
            }
            comparison_results.append(result)
    
    # 分析和汇总结果
    analyze_weight_comparison_results(comparison_results, base_save_dir)
    
    return comparison_results

def analyze_weight_comparison_results(results, base_save_dir):
    """分析权重对比结果"""
    
    import pandas as pd
    import os
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存完整结果
    os.makedirs(os.path.join(base_save_dir, 'weight_comparison'), exist_ok=True)
    df.to_csv(os.path.join(base_save_dir, 'weight_comparison', 'weight_comparison_results.csv'), index=False)
    
    # 筛选有效结果并排序
    df_valid = df[df['RMSE'].notna()].copy()
    df_valid = df_valid.sort_values('RMSE')
    
    print(f"\n{'='*80}")
    print(f"📊 权重策略对比结果 (按RMSE排序)")
    print(f"{'='*80}")
    print(f"{'排名':<4} {'策略名称':<20} {'RMSE':<10} {'相关系数':<8} {'权重分配':<25} {'说明'}")
    print(f"-" * 100)
    
    for i, (_, row) in enumerate(df_valid.iterrows(), 1):
        weights_str = f"[{row['ec_10m_weight']:.2f},{row['ec_70m_weight']:.2f},{row['gfs_10m_weight']:.2f},{row['gfs_70m_weight']:.2f}]"
        print(f"{i:<4} {row['strategy']:<20} {row['RMSE']:<10.4f} {row['Correlation']:<8.4f} {weights_str:<25} {row['description']}")
    
    # 分析不同权重策略的效果
    print(f"\n🔍 权重策略分析:")
    
    if len(df_valid) > 0:
        best_strategy = df_valid.iloc[0]
        worst_strategy = df_valid.iloc[-1]
        
        print(f"  🏆 最佳策略: {best_strategy['strategy']}")
        print(f"     权重: EC-10m({best_strategy['ec_10m_weight']:.2f}), EC-70m({best_strategy['ec_70m_weight']:.2f})")
        print(f"          GFS-10m({best_strategy['gfs_10m_weight']:.2f}), GFS-70m({best_strategy['gfs_70m_weight']:.2f})")
        print(f"     RMSE: {best_strategy['RMSE']:.4f}")
        
        print(f"  📉 最差策略: {worst_strategy['strategy']}")
        print(f"     RMSE: {worst_strategy['RMSE']:.4f}")
        
        improvement = (worst_strategy['RMSE'] - best_strategy['RMSE']) / worst_strategy['RMSE'] * 100
        print(f"  📈 最优vs最差改善: {improvement:.2f}%")
        
        # 等权重策略的表现
        equal_weight = df_valid[df_valid['strategy'] == 'Fusion-Equal']
        if len(equal_weight) > 0:
            equal_result = equal_weight.iloc[0]
            equal_rank = df_valid[df_valid['strategy'] == 'Fusion-Equal'].index[0] + 1
            print(f"  ⚖️ 等权重策略表现:")
            print(f"     排名: 第{equal_rank}名 (共{len(df_valid)}个)")
            print(f"     RMSE: {equal_result['RMSE']:.4f}")
            
            if equal_result['RMSE'] == best_strategy['RMSE']:
                print(f"     🎯 等权重策略就是最优策略!")
            else:
                gap = (equal_result['RMSE'] - best_strategy['RMSE']) / best_strategy['RMSE'] * 100
                print(f"     📊 与最优策略差距: {gap:.2f}%")
    
    # EC vs GFS 权重效果分析
    print(f"\n🔬 EC vs GFS 权重效果分析:")
    
    for _, row in df_valid.iterrows():
        ec_weight = row['ec_total_weight'] 
        gfs_weight = row['gfs_total_weight']
        print(f"  {row['strategy']:<20} EC:{ec_weight:.2f} GFS:{gfs_weight:.2f} → RMSE:{row['RMSE']:.4f}")
    
    # 10m vs 70m 权重效果分析  
    print(f"\n🌪️ 10m vs 70m 权重效果分析:")
    
    for _, row in df_valid.iterrows():
        m10_weight = row['m10_total_weight']
        m70_weight = row['m70_total_weight'] 
        print(f"  {row['strategy']:<20} 10m:{m10_weight:.2f} 70m:{m70_weight:.2f} → RMSE:{row['RMSE']:.4f}")
    
    # 保存分析报告
    with open(os.path.join(base_save_dir, 'weight_comparison', 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write("权重策略对比分析报告\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. 策略排名:\n")
        for i, (_, row) in enumerate(df_valid.iterrows(), 1):
            f.write(f"{i}. {row['strategy']}: RMSE={row['RMSE']:.4f}, 权重={row['weights']}\n")
        
        f.write(f"\n2. 主要发现:\n")
        if len(df_valid) > 0:
            best = df_valid.iloc[0]
            f.write(f"- 最优策略: {best['strategy']} (RMSE: {best['RMSE']:.4f})\n")
            f.write(f"- 最优权重: {best['weights']}\n")
            
            equal_weight = df_valid[df_valid['strategy'] == 'Fusion-Equal']
            if len(equal_weight) > 0:
                equal_rmse = equal_weight.iloc[0]['RMSE']
                f.write(f"- 等权重表现: RMSE={equal_rmse:.4f}\n")
                
                if equal_rmse == best['RMSE']:
                    f.write(f"- 结论: 等权重策略已经是最优的!\n")
                else:
                    gap = (equal_rmse - best['RMSE']) / best['RMSE'] * 100
                    f.write(f"- 权重优化带来的改善: {gap:.2f}%\n")




# 主函数调用示例
if __name__ == "__main__":
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    BASE_SAVE_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/weight_comparison_experiments"
    INDICES_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/third_part_experiments/train_test_split.json"
    
    # 注意：需要先导入原始代码中的 run_experiment 函数
    # from test import run_experiment
    
    results = run_weight_comparison_experiments(DATA_PATH, BASE_SAVE_DIR, INDICES_PATH)