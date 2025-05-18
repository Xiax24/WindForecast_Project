WindForecast_Project/
├── 01_Data/
│   ├── raw/                      # 原始数据
│   │   ├── obs/                  # 观测数据
│   │   │   ├── changma/          # 昌马测风塔数据
│   │   │   ├── sanlijijingzi/    # 三十里井子测风塔数据
│   │   │   └── kuangqu/          # 矿区测风塔数据
│   │   └── wrf/                  # WRF模拟输出
│   │       ├── gfs_driven/       # GFS驱动的WRF输出
│   │       └── ec_driven/        # EC驱动的WRF输出
│   ├── processed/                # 处理后的数据
│   │   ├── aligned/              # 时间对齐后的数据
│   │   ├── cleaned/              # 清洗后的数据
│   │   └── features/             # 提取的物理特征
│   └── metadata/                 # 数据描述文件
│       └── data_inventory.xlsx   # 数据清单
│
├── 02_Code/
│   ├── preprocessing/            # 数据预处理脚本
│   │   ├── data_loading.py
│   │   ├── quality_control.py
│   │   └── feature_extraction.py
│   ├── analysis/                 # 分析脚本
│   │   ├── stability_analysis.py
│   │   ├── time_scale_analysis.py
│   │   ├── error_decomposition.py
│   │   └── case_analysis.py
│   ├── ml_methods/               # 机器学习方法
│   │   ├── clustering.py
│   │   ├── shap_analysis.py
│   │   └── prediction_model.py
│   ├── visualization/            # 可视化脚本
│   │   ├── time_series_plots.py
│   │   ├── profile_plots.py
│   │   └── contribution_plots.py
│   └── utils/                    # 工具函数
│       ├── metrics.py
│       └── helpers.py
│
├── 03_Results/
│   ├── figures/                  # 图表结果
│   │   ├── exploratory/          # 探索性分析图表
│   │   ├── stability/            # 稳定度分析图表
│   │   ├── time_scales/          # 时间尺度分析图表
│   │   ├── error_analysis/       # 误差分析图表
│   │   ├── ml_results/           # 机器学习结果图表
│   │   └── paper_figures/        # 论文用高质量图表
│   ├── tables/                   # 表格结果
│   ├── models/                   # 训练的模型
│   └── case_studies/             # 案例研究结果
│
├── 04_Literature/
│   ├── papers/                   # 论文PDF
│   ├── notes/                    # 文献笔记
│   └── references.bib            # BibTeX参考文献文件
│
├── 05_Manuscript/
│   ├── drafts/                   # 论文草稿
│   ├── figures/                  # 最终论文图表
│   ├── submissions/              # 投稿版本
│   ├── reviews/                  # 审稿意见
│   └── final/                    # 最终稿件
│
├── 06_Presentations/
│   ├── progress/                 # 进展汇报
│   └── conference/               # 会议报告
│
├── 07_Documentation/
│   ├── research_log.md           # 研究日志
│   ├── meeting_notes/            # 会议记录
│   ├── workflows/                # 工作流程文档
│   └── data_descriptions/        # 数据说明文档
│
├── environment.yml               # Conda环境配置
├── README.md                     # 项目说明
└── .gitignore                    # Git忽略配置
