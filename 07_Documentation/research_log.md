# 风电场风速预报研究日志

## 2025年5月18日 - 项目启动

### 今日完成工作

1. **项目框架建立**
   - 创建了完整的项目文件夹结构，包括数据、代码、结果等主要目录
   - 设置了项目README.md，明确了研究目标和方法
   - 创建了requirements.txt记录项目依赖

2. **开发环境配置**
   - 创建了名为"windforecast"的conda环境
   - 安装了核心分析库：pandas, numpy, scikit-learn等
   - 解决了wrf-python在M系列Mac上的安装问题，采用替代方案

3. **Git版本控制设置**
   - 初始化了本地Git仓库
   - 创建了.gitignore文件，排除了数据文件和临时文件
   - 准备连接到GitHub远程仓库

### 遇到的问题与解决方案

1. **wrf-python安装问题**
   - 问题：conda无法在M系列Mac上找到wrf-python包
   - 解决方案：决定使用netCDF4和xarray直接处理WRF输出文件，或考虑使用x86_64环境

2. **项目范围界定**
   - 问题：研究内容广泛，需要明确优先级
   - 解决方案：决定首先聚焦于稳定度分类和边界层结构分析

### 观察与思考

- 项目框架设置比预期花费了更多时间，但这是值得的长期投资（倒也不至于，就是自己摸鱼了🦑了一段时间）
- 需要进一步了解数据格式和结构，特别是WRF输出的组织方式
- 考虑采用增量开发策略，先实现基础分析功能，再逐步添加高级方法

### 下一步计划

1. **数据探索（明天）**
   - 加载样本数据，了解数据结构和格式
   - 创建基本可视化，观察风速时间序列特征
   - 评估数据质量和缺失情况

2. **基础函数开发（明天-后天）**
   - 实现数据加载和预处理功能
   - 编写风切变参数计算函数
   - 开发初步的稳定度分类方法

3. **本周目标**
   - 完成数据预处理流程
   - 实现基本的稳定度分析
   - 生成初步的GFS vs EC误差统计

### 资源与参考

项目结构参考：[Standard ML Project Structure](https://github.com/example/project)

稳定度分类方法参考：
- #### 支持风切变参数(α)稳定度分类的核心文献
- Irwin, J.S. (1979). "A theoretical variation of the wind profile power-law exponent as a function of surface roughness and stability." Atmospheric Environment, 13(1), 191-194.
该文章建立了风切变指数α与大气稳定度之间的理论关系
提供了不同稳定度条件下α值的理论范围
- Wharton, S., & Lundquist, J.K. (2012). "Atmospheric stability affects wind turbine power collection." Environmental Research Letters, 7(1), 014005.
专门针对风能应用研究稳定度分类方法
验证了α值在风电场环境中作为稳定度指标的有效性
提供了稳定度与风电机组性能关系的实证分析

- Emeis, S. (2013). "Wind energy meteorology: atmospheric physics for wind power generation." Springer.
该专著3.2节详细讨论了风切变参数与稳定度的关系
提供了α值分类稳定度的标准阈值
讨论了该方法在风能领域的广泛应用
- Peña, A., Gryning, S.E., & Hasager, C.B. (2010). "Comparing mixing-length models of the diabatic wind profile over homogeneous terrain." Theoretical and Applied Climatology, 100(3-4), 325-335.
比较了不同稳定度条件下的风剖面模型
验证了α值变化与温度梯度变化的一致性
提供了风切变参数用于稳定度分类的实证支持
- Newman, J.F., & Klein, P.M. (2014). "The impacts of atmospheric stability on the accuracy of wind speed extrapolation methods." Resources, 3(1), 81-105.
研究了不同稳定度下风速外推方法的准确性
验证了风切变参数与Richardson数和Obukhov长度的相关性
支持在缺乏温度梯度数据时使用风切变参数作为替代

- #### 多因素辅助判断的支持文献

- Barthelmie, R.J., Palutikof, J.P., & Davies, T.D. (1993). "Estimation of sector roughness and the effect on prediction of the vertical wind speed profile." Boundary-Layer Meteorology, 66(1-2), 19-47.
讨论了结合多种因素(包括时间、风向等)提高稳定度分类准确性的方法
提供了综合判断框架的实证基础


- Gualtieri, G. (2019). "A comprehensive review on wind resource extrapolation models applied in wind energy." Renewable and Sustainable Energy Reviews, 102, 215-233.
综述了风能领域使用的各种稳定度分类方法
讨论了在数据有限情况下的替代方法和组合策略
支持在仅有单层温度数据时使用风切变参数为主的组合方法


- Risan, A., Lund, J.A., Chang, C.Y., & Sætran, L. (2018). "Wind in complex terrain - lidar measurements for evaluation of CFD simulations." Remote Sensing, 10(1), 59.
讨论了复杂地形中的稳定度估计方法
验证了风剖面特征作为稳定度指标的有效性
证明了在缺少温度梯度情况下风切变参数的适用性

- #### 方法优化和验证的关键文献

- Optis, M., Monahan, A., & Bosveld, F.C. (2016). "Limitations and breakdown of Monin–Obukhov similarity theory for wind profile extrapolation under stable stratification." Wind Energy, 19(6), 1053-1072.
详细讨论了稳定条件下风剖面特征
提供了优化风切变参数使用的指导
讨论了该方法的局限性和适用条件


- Argyle, P., & Watson, S. (2014). "Assessing the dependence of surface layer atmospheric stability on measurement height at offshore locations." Journal of Wind Engineering and Industrial Aerodynamics, 131, 88-99.
研究了测量高度对稳定度评估的影响
验证了70m/10m高度对作为计算α值的有效性
提供了提高α值方法准确性的实用建议


## 2025年5月20日 - 观测数据处理完成

### 今日完成工作

1. **观测数据清洗与处理**
   - 完成了3个风电场/测风塔的观测数据处理：
     - **昌马测风塔** (`changma`)：2021-05-01至2022-10-31，548天数据
     - **三十里井子风电场** (`sanlijijingzi`)：2021-06-01至2022-06-16，381天数据  
     - **矿区风电场** (`kuangqu`)：数据已处理完成
   - 实现了多高度层数据整合（10m, 30m, 50m, 70m）
   - 建立了标准化的宽格式数据结构

2. **数据质量控制系统**
   - 实现了多层次异常值检测：
     - 极值检查（风速>50m/s, 温度范围等）
     - 范围检查（风向0-360°等）
     - **僵值检测**：创新性地实现了连续5个相同数值的僵值识别和清理
   - 建立了缺失值标记识别系统（-99, NULL等多种格式）
   - 生成了详细的数据质量报告和统计摘要

3. **时间序列标准化**
   - 处理了复杂的时间格式（分列日期时间vs合并datetime）
   - 实现了智能重采样算法：
     - **风向**：圆形平均算法（解决0°/360°边界问题）
     - **风速变量**：按类型分别处理（平均/最大/最小/瞬时）
     - **气象变量**：标准算术平均
   - 生成了15分钟标准间隔数据产品

4. **文件组织与管理**
   - 建立了清晰的数据产品层次：
     - `cleaned_original/`：原始时间分辨率清洗数据
     - `cleaned-15min/`：15分钟标准间隔数据
   - 每个数据产品都包含详细的处理摘要和元数据

### 技术创新点

1. **僵值检测算法**
   - 基于时间序列分组，识别连续相同数值
   - 参数可调（默认5个连续值）
   - 适用于传感器故障导致的数据冻结问题

2. **多变量智能重采样**
   - 不同类型变量采用最适合的聚合方法
   - 风向使用圆形平均，避免角度计算错误
   - 保持了原始数据的物理意义

3. **分离式数据架构**
   - 原始清洗数据：保持最大信息量，用于详细分析
   - 15分钟标准数据：便于与NWP等外部数据对比

### 数据质量成果

**昌马测风塔（示例）**：
- 数据完整性：风速数据99.9%，风向数据99.9%
- 52,608个时间点（15分钟间隔后约17,536个）
- 4个高度层全部成功处理

**三十里井子风电场（示例）**：
- 包含了更丰富的风速指标：平均、最大、最小、瞬时、标准差
- 风向包含平均风向和瞬时风向
- 数据压缩比3:1（5分钟→15分钟）

### 遇到的挑战与解决方案

1. **数据格式不一致**
   - 挑战：不同风电场使用不同的时间列格式
   - 解决：开发了灵活的时间解析系统，支持分列和合并格式

2. **传感器异常数据**
   - 挑战：发现大量连续相同的异常数值（传感器卡住）
   - 解决：开发僵值检测算法，自动识别和清理

3. **风向数据处理**
   - 挑战：风向的圆形特性导致传统平均值计算错误
   - 解决：实现圆形平均算法，正确处理跨越0°/360°的情况

### 观察与分析

- **数据质量总体良好**：3个站点的主要变量完整性都超过99%
- **僵值问题较为常见**：所有站点都发现了传感器僵值，验证了检测算法的必要性
- **三十里井子数据最丰富**：包含了更多风速指标，有利于后续湍流分析

### 下一步计划

1. **数据探索分析（明天）**
   - 生成风况统计和可视化
   - 分析不同高度层的风切变特征
   - 评估3个站点的风资源质量

2. **与NWP数据对比（本周）**
   - 下载对应时期的GFS/EC数据
   - 建立时空匹配系统
   - 开始预报误差统计分析

3. **稳定度分析准备**
   - 利用多高度风速数据计算风切变参数α
   - 结合时间和季节信息进行稳定度初步分类
   - 为后续详细分析建立基础

