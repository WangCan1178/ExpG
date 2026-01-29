#!/usr/bin/env python3
"""
计算测试集的avg@n和pass@n
测试集来自final_task.jsonl中的task_id（不包括related_task）
计算每个子任务的avg@n和pass@n，以及标准差
支持传入采样次数n（默认为3），当n=1时只使用采样1的结果
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics


def load_test_task_ids(final_task_jsonl_path: str) -> set:
    """从final_task.jsonl中提取所有task_id（不包括related_task_ids）"""
    task_ids = set()
    with open(final_task_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task_data = json.loads(line)
            task_id = task_data.get('task_id')
            if task_id:
                task_ids.add(task_id)
    return task_ids


def load_score_file(score_file_path: str) -> Dict[str, bool]:
    """
    加载评分文件，返回task_id到valid的映射
    评分文件第一行是汇总信息，跳过
    """
    results = {}
    if not os.path.exists(score_file_path):
        return results
    
    with open(score_file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 第一行通常是汇总信息，跳过（通常包含accuracy字段）
                if line_idx == 0 and 'accuracy' in data:
                    continue
                # 检查是否有id和valid字段
                if 'id' in data and 'valid' in data:
                    task_id = data['id']
                    valid = data['valid']
                    results[task_id] = valid
            except json.JSONDecodeError:
                continue
    return results


def categorize_score_file(file_path: str, category: str) -> str:
    """
    将原始类别映射到四个大类别
    参考aggregate_scores.py的分类逻辑
    
    返回: multi_turn_base, multi_turn_other, live_nonlive_other, live_nonlive_base
    """
    path = os.path.normpath(file_path).lower()
    parts = [p for p in path.split(os.sep) if p]
    filename = os.path.basename(path)
    
    # multi_turn类别
    if 'multi_turn' in parts:
        if 'base' in filename:
            return 'multi_turn_base'
        else:
            return 'multi_turn_other'
    
    # live或non_live类别
    if 'live' in parts or 'non_live' in parts:
        if 'irrelevance' in filename or 'relevance' in filename:
            return 'live_nonlive_other'
        else:
            return 'live_nonlive_base'
    
    # 默认归类到live_nonlive_base
    return 'live_nonlive_base'


def find_all_score_files(base_dir: str, n: int = 3) -> Dict[str, List[str]]:
    """
    找到所有评分文件，按类别分组
    递归查找所有子目录中的评分文件
    返回格式: {category: [file_path_1, file_path_2, ..., file_path_n]}
    
    Args:
        base_dir: 基础目录
        n: 采样次数，只查找run_num在1到n之间的文件
    """
    score_files = defaultdict(list)
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return score_files
    
    # 当 n == 0 时，表示使用「不带采样后缀」的文件
    # 例如: BFCL_v4_live_irrelevance_score.json
    if n == 0:
        # 递归查找所有不带采样编号的评分文件（score/result 两种模式）
        score_files_found = list(base_path.rglob('BFCL_v4_*_score.json'))
        if not score_files_found:
            score_files_found = list(base_path.rglob('BFCL_v4_*_result.json'))
        
        for score_file in score_files_found:
            filename = score_file.stem  # 去掉.json
            
            # 处理 *_score 或 *_result 结尾的文件名
            if filename.endswith('_score'):
                core_name = filename[:-len('_score')]
            elif filename.endswith('_result'):
                core_name = filename[:-len('_result')]
            else:
                # 不符合预期命名，跳过
                continue
            
            # 去掉 BFCL_v4_ 前缀
            if core_name.startswith('BFCL_v4_'):
                category = core_name[len('BFCL_v4_'):]
            else:
                category = core_name
            
            score_files[category].append(str(score_file))
    else:
        # 递归查找所有带采样编号的评分文件（支持score和result两种文件名模式）
        score_files_found = list(base_path.rglob('BFCL_v4_*_score_*.json'))
        if not score_files_found:
            # 如果没有找到score文件，尝试result文件
            score_files_found = list(base_path.rglob('BFCL_v4_*_result_*.json'))
        
        # 生成允许的run_num列表（1到n）
        allowed_run_nums = [str(i) for i in range(1, n + 1)]
        
        for score_file in score_files_found:
            # 提取类别名称（去掉路径、前缀和后缀）
            # 例如: BFCL_v4_live_simple_score_1.json -> live_simple
            filename = score_file.stem  # 去掉.json
            
            # 尝试两种分割方式：_score_ 或 _result_
            parts = filename.split('_score_')
            if len(parts) != 2:
                parts = filename.split('_result_')
                if len(parts) != 2:
                    continue
            
            category = parts[0].replace('BFCL_v4_', '')  # 去掉BFCL_v4_前缀
            
            # 提取运行编号
            run_num = parts[1]
            if run_num in allowed_run_nums:
                score_files[category].append(str(score_file))
    
    # 对每个类别的文件列表进行排序
    for category in score_files:
        score_files[category].sort()
    
    return score_files


def calculate_avg_pass_atn(task_results: List[bool], n: int) -> Tuple[float, float, float]:
    """
    计算avg@n和pass@n，以及该task多次采样的标准差
    task_results: n次运行的结果列表（布尔值列表）
    n: 采样次数
    
    avg@n: n次运行的平均正确率
    pass@n: 至少通过1次的概率（如果有至少1次通过则为1.0，否则为0.0）
    sampling_std: 该task多次采样结果的标准差（反映该task的采样波动性，n=1时为0.0）
    """
    if not task_results:
        return 0.0, 0.0, 0.0
    
    # avg@n: 平均正确率
    avg_atn = sum(task_results) / len(task_results)
    
    # pass@n: 至少通过1次的概率
    pass_atn = 1.0 if any(task_results) else 0.0
    
    # 计算该task多次采样结果的标准差（将布尔值转为0/1）
    numeric_results = [1.0 if r else 0.0 for r in task_results]
    if len(numeric_results) > 1:
        sampling_std = statistics.stdev(numeric_results)
    else:
        sampling_std = 0.0
    
    return avg_atn, pass_atn, sampling_std


def find_final_task_jsonl(base_dir: str) -> Optional[str]:
    """
    尝试查找final_task.jsonl文件
    从根目录向上查找，或在整个项目目录中查找
    """
    base_path = Path(base_dir).resolve()
    
    # 1. 首先尝试在根目录的常见相对位置查找
    possible_paths = [
        # 相对于根目录的可能路径
        base_path / '..' / '..' / 'gorilla' / 'berkeley-function-call-leaderboard' / 'bfcl_eval' / 'data' / 'final_task.jsonl',
        base_path / '..' / '..' / '..' / 'gorilla' / 'berkeley-function-call-leaderboard' / 'bfcl_eval' / 'data' / 'final_task.jsonl',
        # 从根目录向上查找，直到找到 Toolmem 目录
        base_path / '..' / '..' / '..' / '..' / 'gorilla' / 'berkeley-function-call-leaderboard' / 'bfcl_eval' / 'data' / 'final_task.jsonl',
    ]
    
    for path in possible_paths:
        abs_path = path.resolve()
        if abs_path.exists():
            return str(abs_path)
    
    # 2. 从当前目录向上查找，直到找到 Toolmem 根目录
    current = base_path
    while current != current.parent:
        test_path = current / 'experiment' / 'gorilla' / 'berkeley-function-call-leaderboard' / 'bfcl_eval' / 'data' / 'final_task.jsonl'
        if test_path.exists():
            return str(test_path.resolve())
        current = current.parent
    
    # 3. 默认路径
    default_path = '/Users/xiaocan/Toolmem/experiment/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/final_task.jsonl'
    default_path = '/Users/xiaocan/Toolmem/experiment/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/all_task.jsonl'
    return default_path


def main():
    # 解析命令行参数
    n = 3  # 默认采样次数
    score_base_dir = None
    
    if len(sys.argv) > 1:
        score_base_dir = os.path.abspath(sys.argv[1])
    if len(sys.argv) > 2:
        try:
            n = int(sys.argv[2])
            # 约定: n=0 表示使用「不带采样后缀」的结果文件
            if n < 0:
                print(f"错误: 采样次数n必须大于等于0，当前值: {n}")
                sys.exit(1)
        except ValueError:
            print(f"错误: 采样次数n必须是整数，当前值: {sys.argv[2]}")
            sys.exit(1)
    
    if score_base_dir is None:
        # 默认路径
        score_base_dir = '/Users/xiaocan/Toolmem/experiment/gorilla/berkeley-function-call-leaderboard/result_new/score_m0/qwen3-8b'
        print(f"警告: 未指定根目录，使用默认路径: {score_base_dir}")
        print(f"用法: python {sys.argv[0]} <根目录路径> [采样次数n]")
        print(f"      采样次数n默认为3，当n=1时只使用采样1的结果")
        print()
    
    if not os.path.exists(score_base_dir):
        print(f"错误: 根目录不存在: {score_base_dir}")
        sys.exit(1)
    
    print(f"采样次数: {n} (n=0 表示使用不带采样后缀的结果文件)")
    
    # 自动查找final_task.jsonl
    print("正在查找 final_task.jsonl 文件...")
    final_task_jsonl_path = find_final_task_jsonl(score_base_dir)
    if not os.path.exists(final_task_jsonl_path):
        print(f"错误: 找不到final_task.jsonl文件: {final_task_jsonl_path}")
        print("请确保 final_task.jsonl 文件存在于以下位置之一：")
        print("  - experiment/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/final_task.jsonl")
        print("  - 或者相对于根目录的其他位置")
        sys.exit(1)
    
    print(f"使用评分目录: {score_base_dir}")
    print(f"使用测试集文件: {final_task_jsonl_path}")
    
    # 1. 加载测试集task_id
    print("正在加载测试集task_id...")
    test_task_ids = load_test_task_ids(final_task_jsonl_path)
    print(f"找到 {len(test_task_ids)} 个测试task_id")
    
    # 2. 找到所有评分文件
    print("\n正在查找评分文件...")
    score_files = find_all_score_files(score_base_dir, n)
    print(f"找到 {len(score_files)} 个测试类别")
    
    # 3. 为每个task_id收集n次运行的结果
    print("\n正在加载评分结果...")
    task_results = defaultdict(lambda: defaultdict(list))  # {task_id: {category: [result1, result2, ..., resultn]}}
    
    for category, file_paths in score_files.items():
        print(f"  处理类别: {category}")
        # n > 0: 需要至少 n 个评分文件；n = 0: 使用所有（通常是 1 个）文件
        if n > 0 and len(file_paths) < n:
            print(f"    警告: {category} 只有 {len(file_paths)} 个评分文件，需要{n}个")
            continue
        
        # 加载运行结果
        if n > 0:
            iter_files = list(enumerate(file_paths[:n], 1))
        else:
            # n=0 时，不区分采样编号，全部读入（通常只有 1 个）
            iter_files = list(enumerate(file_paths, 1))
        
        for run_idx, file_path in iter_files:
            results = load_score_file(file_path)
            print(f"    运行 {run_idx}: {len(results)} 个结果")
            for task_id, valid in results.items():
                # 只处理测试集中的task_id
                if task_id in test_task_ids:
                    task_results[task_id][category].append(valid)
    
    # 4. 计算每个task_id的avg@n和pass@n（按大类别合并）
    print("\n正在计算指标...")
    # 先按原始category计算，然后映射到大类别
    category_metrics = defaultdict(lambda: {
        'avg_atn_list': [], 
        'pass_atn_list': [], 
        'sampling_std_list': []  # 每个task多次采样的标准差列表
    })
    
    # 建立category到文件路径的映射（用于分类）
    category_to_file = {}
    for category, file_paths in score_files.items():
        if file_paths:
            category_to_file[category] = file_paths[0]  # 使用第一个文件路径来判断
    
    for task_id, category_results in task_results.items():
        for category, results in category_results.items():
            # n>0: 只处理有 n 次运行结果的；n=0: 只要有结果就计算
            if (n > 0 and len(results) == n) or (n == 0 and len(results) >= 1):
                avg_atn, pass_atn, sampling_std = calculate_avg_pass_atn(results, n)
                # 将原始category映射到大类别
                file_path = category_to_file.get(category, '')
                merged_category = categorize_score_file(file_path, category)
                category_metrics[merged_category]['avg_atn_list'].append(avg_atn)
                category_metrics[merged_category]['pass_atn_list'].append(pass_atn)
                category_metrics[merged_category]['sampling_std_list'].append(sampling_std)
    
    # 5. 计算每个大类别的统计信息
    print("\n" + "="*80)
    print(f"结果统计（按四个大类别，n={n}）")
    print("="*80)
    
    all_avg_atn = []
    all_pass_atn = []
    all_sampling_std = []
    
    # 按照固定顺序输出四个大类别
    category_order = ['multi_turn_base', 'multi_turn_other', 'live_nonlive_other', 'live_nonlive_base']
    for category in category_order:
        metrics = category_metrics[category]
        avg_list = metrics['avg_atn_list']
        pass_list = metrics['pass_atn_list']
        sampling_std_list = metrics['sampling_std_list']
        
        if not avg_list:
            continue
        
        # 计算平均值和标准差（不同task之间的标准差）
        avg_atn_mean = statistics.mean(avg_list)
        avg_atn_std = statistics.stdev(avg_list) if len(avg_list) > 1 else 0.0
        pass_atn_mean = statistics.mean(pass_list)
        pass_atn_std = statistics.stdev(pass_list) if len(pass_list) > 1 else 0.0
        
        # 计算多次采样标准差的平均值（反映模型多次采样的波动性）
        # 当n=1时，sampling_std始终为0.0，所以不需要显示
        avg_sampling_std = statistics.mean(sampling_std_list) if sampling_std_list else 0.0
        
        print(f"\n大类别: {category}")
        print(f"  样本数量: {len(avg_list)}")
        print(f"  avg@{n}: {avg_atn_mean:.4f} ± {avg_atn_std:.4f} (task间差异)")
        if n > 1:
            print(f"        平均采样波动性: {avg_sampling_std:.4f}")
        print(f"  pass@{n}: {pass_atn_mean:.4f} ± {pass_atn_std:.4f} (task间差异)")
        
        all_avg_atn.extend(avg_list)
        all_pass_atn.extend(pass_list)
        all_sampling_std.extend(sampling_std_list)
    
    # 输出其他可能存在的类别（如果有的话）
    for category in sorted(category_metrics.keys()):
        if category not in category_order:
            metrics = category_metrics[category]
            avg_list = metrics['avg_atn_list']
            pass_list = metrics['pass_atn_list']
            sampling_std_list = metrics['sampling_std_list']
            
            if not avg_list:
                continue
            
            avg_atn_mean = statistics.mean(avg_list)
            avg_atn_std = statistics.stdev(avg_list) if len(avg_list) > 1 else 0.0
            pass_atn_mean = statistics.mean(pass_list)
            pass_atn_std = statistics.stdev(pass_list) if len(pass_list) > 1 else 0.0
            
            # 计算多次采样标准差的平均值
            avg_sampling_std = statistics.mean(sampling_std_list) if sampling_std_list else 0.0
            
            print(f"\n类别: {category}")
            print(f"  样本数量: {len(avg_list)}")
            print(f"  avg@{n}: {avg_atn_mean:.4f} ± {avg_atn_std:.4f} (task间差异)")
            if n > 1:
                print(f"        平均采样波动性: {avg_sampling_std:.4f}")
            print(f"  pass@{n}: {pass_atn_mean:.4f} ± {pass_atn_std:.4f} (task间差异)")
            
            all_avg_atn.extend(avg_list)
            all_pass_atn.extend(pass_list)
            all_sampling_std.extend(sampling_std_list)
    
    # 6. 计算总体统计
    if all_avg_atn:
        print("\n" + "-"*80)
        print("总体统计（所有类别）")
        print("-"*80)
        overall_avg_atn_mean = statistics.mean(all_avg_atn)
        overall_avg_atn_std = statistics.stdev(all_avg_atn) if len(all_avg_atn) > 1 else 0.0
        overall_pass_atn_mean = statistics.mean(all_pass_atn)
        overall_pass_atn_std = statistics.stdev(all_pass_atn) if len(all_pass_atn) > 1 else 0.0
        
        # 计算总体多次采样标准差的平均值
        overall_avg_sampling_std = statistics.mean(all_sampling_std) if all_sampling_std else 0.0
        
        print(f"总样本数量: {len(all_avg_atn)}")
        print(f"总体 avg@{n}: {overall_avg_atn_mean:.4f} ± {overall_avg_atn_std:.4f} (task间差异)")
        if n > 1:
            print(f"          平均采样波动性: {overall_avg_sampling_std:.4f}")
        print(f"总体 pass@{n}: {overall_pass_atn_mean:.4f} ± {overall_pass_atn_std:.4f} (task间差异)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

