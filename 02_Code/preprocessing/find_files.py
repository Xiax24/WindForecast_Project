"""
文件查找诊断脚本
帮助找到矿区数据文件的位置
"""

import os
import glob

def find_kuangqu_files():
    """查找矿区相关的数据文件"""
    
    # 项目根目录
    project_root = "/Users/xiaxin/work/WindForecast_Project"
    
    print("="*60)
    print("查找矿区数据文件")
    print("="*60)
    
    # 1. 检查主要目录结构
    print("\n1. 检查主要目录结构:")
    data_dir = os.path.join(project_root, "01_Data")
    if os.path.exists(data_dir):
        print(f"✓ 数据目录存在: {data_dir}")
        
        # 列出01_Data下的子目录
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"  子目录: {subdirs}")
    else:
        print(f"✗ 数据目录不存在: {data_dir}")
        return
    
    # 2. 在整个01_Data目录树中搜索包含kuangqu的文件
    print("\n2. 搜索包含'kuangqu'的文件:")
    kuangqu_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if 'kuangqu' in file.lower() and file.endswith('.csv'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, project_root)
                kuangqu_files.append((full_path, relative_path))
                print(f"  找到: {relative_path}")
    
    if not kuangqu_files:
        print("  未找到包含'kuangqu'的CSV文件")
    
    # 3. 检查具体的子目录
    print("\n3. 检查各个子目录的CSV文件:")
    
    subdirs_to_check = ['raw', 'processed', 'cleaned']
    
    for subdir in subdirs_to_check:
        subdir_path = os.path.join(data_dir, subdir)
        print(f"\n  检查 {subdir} 目录:")
        
        if os.path.exists(subdir_path):
            # 递归查找CSV文件
            csv_pattern = os.path.join(subdir_path, "**", "*.csv")
            csv_files = glob.glob(csv_pattern, recursive=True)
            
            if csv_files:
                print(f"    找到 {len(csv_files)} 个CSV文件:")
                for csv_file in csv_files:
                    rel_path = os.path.relpath(csv_file, data_dir)
                    print(f"      {rel_path}")
            else:
                print(f"    该目录下没有CSV文件")
        else:
            print(f"    目录不存在: {subdir_path}")
    
    # 4. 如果找到了kuangqu文件，显示详细信息
    if kuangqu_files:
        print(f"\n4. 矿区文件详细信息:")
        for full_path, rel_path in kuangqu_files:
            print(f"\n  文件: {rel_path}")
            print(f"  完整路径: {full_path}")
            print(f"  文件大小: {os.path.getsize(full_path)} 字节")
            
            # 尝试读取前几行
            try:
                import pandas as pd
                df = pd.read_csv(full_path, nrows=5)
                print(f"  列名: {list(df.columns)}")
                print(f"  数据形状预览: 至少 {len(df)} 行, {len(df.columns)} 列")
            except Exception as e:
                print(f"  读取预览失败: {e}")
    
    # 5. 推荐处理路径
    print(f"\n5. 推荐的处理方案:")
    if kuangqu_files:
        # 选择最合适的文件
        best_file = None
        
        # 优先选择processed或cleaned目录下的文件
        for full_path, rel_path in kuangqu_files:
            if 'cleaned' in rel_path:
                best_file = (full_path, rel_path)
                break
            elif 'processed' in rel_path:
                best_file = (full_path, rel_path)
        
        if not best_file:
            best_file = kuangqu_files[0]
        
        print(f"  建议使用文件: {best_file[1]}")
        print(f"  完整路径: {best_file[0]}")
        
        # 生成修改后的代码路径
        file_dir = os.path.dirname(best_file[0])
        print(f"  数据目录路径: {file_dir}")
        
    else:
        print("  未找到矿区数据文件")
        print("  请检查:")
        print("    - 数据是否已经处理完成")
        print("    - 文件名是否正确")
        print("    - 文件是否在其他位置")

if __name__ == "__main__":
    find_kuangqu_files()