#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 Task2 的 pass@3 和 avg@3 指标。

使用方法:
python compute_pass_at_k.py --input-dir tool/new_test_res/qwen3-235b-a22b --files Task2_0 Task2_1 Task2_2
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# 导入 extract.py 中的工具函数
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract import (
    get_subtask_id,
    normalize_tool_field,
    extract_used_tools_from_action_res,
    compute_sample_success,
    SampleInfo,
)


@dataclass
class SampleResult:
    """单个样本的多次采样结果"""
    sample_id: str
    subtask: int
    gt_tools: Set[str]
    gt_tool_names_raw: List[str]
    samples: List[bool]  # 每次采样的 success 结果
    raw_samples: List[Dict[str, Any]]  # 原始采样数据


def load_samples_from_file(file_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    从文件中加载样本，按 sample_id 分组。
    返回: {sample_id: [sample1, sample2, ...]}，其中每个 sample 可能包含多次采样结果
    """
    if not file_path.exists():
        print(f"警告: 文件不存在: {file_path}")
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"警告: 文件 {file_path} 不是 JSON 数组")
        return {}
    
    # 按 sample_id 分组
    samples_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in data:
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get('id', ''))
        if sample_id:
            samples_by_id[sample_id].append(item)
    
    return samples_by_id


def compute_sample_success_for_item(item: Dict[str, Any]) -> Tuple[bool, int, Set[str], List[str]]:
    """
    计算单个样本的 success。
    返回: (success, subtask, used_tools, used_tool_names_raw)
    """
    sid = str(item.get("id", ""))
    st = get_subtask_id(item)
    if st is None:
        st = 1  # 默认使用 subtask 1
    
    gt_set, gt_raw = normalize_tool_field(item.get("tool"))
    # 对于 subtask3，groundtruth 应该为空
    if st == 3:
        gt_set = set()
        gt_raw = []
    
    used_set, used_raw = extract_used_tools_from_action_res(item.get("action_res"))
    
    # 创建 SampleInfo 来计算 success
    sample = SampleInfo(
        raw=item,
        sample_id=sid,
        subtask=st,
        used_tools=used_set,
        gt_tools=gt_set,
        used_tool_names_raw=used_raw,
        gt_tool_names_raw=gt_raw,
    )
    compute_sample_success(sample)
    
    return sample.success, st, used_set, used_raw


def compute_pass_at_k_and_avg(results: List[SampleResult], k: int = 3) -> Dict[int, Dict[str, float]]:
    """
    计算 pass@k 和 avg@k，以及它们的标准差。
    
    pass@k: 在 k 次采样中，至少有一次成功的比例
    avg@k: k 次采样的平均成功率
    
    返回: {subtask: {"pass@k": value, "avg@k": value, "pass@k_std": value, "avg@k_std": value, 
          "avg_sampling_std": value, "total": count}}
    """
    stats = {
        1: {"name": "Similar", "pass_values": [], "avg_values": [], "sampling_std_list": [], "total": 0},
        2: {"name": "Scenario", "pass_values": [], "avg_values": [], "sampling_std_list": [], "total": 0},
        3: {"name": "Reliability", "pass_values": [], "avg_values": [], "sampling_std_list": [], "total": 0},
        4: {"name": "Multi-tool", "pass_values": [], "avg_values": [], "sampling_std_list": [], "total": 0},
    }
    
    for result in results:
        st = result.subtask
        if st not in stats:
            continue
        
        stats[st]["total"] += 1
        
        # 只考虑有 k 次采样的样本
        if len(result.samples) >= k:
            samples_k = result.samples[:k]
            # pass@k: 至少有一次成功（0 或 1）
            pass_value = 1 if any(samples_k) else 0
            stats[st]["pass_values"].append(pass_value)
            # avg@k: 平均成功率（0 到 1 之间的值）
            avg_value = sum(samples_k) / len(samples_k)
            stats[st]["avg_values"].append(avg_value)
            
            # 计算该样本多次采样结果的标准差（将布尔值转为0/1）
            numeric_samples = [1.0 if s else 0.0 for s in samples_k]
            if len(numeric_samples) > 1:
                sampling_std = statistics.stdev(numeric_samples)
            else:
                sampling_std = 0.0
            stats[st]["sampling_std_list"].append(sampling_std)
    
    # 计算最终指标
    metrics = {}
    for st_id in (1, 2, 3, 4):
        stat = stats[st_id]
        total = stat["total"]
        pass_values = stat["pass_values"]
        avg_values = stat["avg_values"]
        sampling_std_list = stat["sampling_std_list"]
        total_samples = len(pass_values)
        
        if total_samples > 0:
            # 计算平均值
            pass_at_k = sum(pass_values) / total_samples
            avg_at_k = sum(avg_values) / total_samples
            
            # 计算标准差（不同样本之间的标准差）
            if total_samples > 1:
                pass_at_k_std = statistics.stdev(pass_values) if len(pass_values) > 1 else 0.0
                avg_at_k_std = statistics.stdev(avg_values) if len(avg_values) > 1 else 0.0
            else:
                pass_at_k_std = 0.0
                avg_at_k_std = 0.0
            
            # 计算平均采样波动性（所有样本多次采样标准差的平均值）
            avg_sampling_std = statistics.mean(sampling_std_list) if sampling_std_list else 0.0
        else:
            pass_at_k = 0.0
            avg_at_k = 0.0
            pass_at_k_std = 0.0
            avg_at_k_std = 0.0
            avg_sampling_std = 0.0
        
        metrics[st_id] = {
            "name": stat["name"],
            f"pass@{k}": pass_at_k,
            f"avg@{k}": avg_at_k,
            f"pass@{k}_std": pass_at_k_std,
            f"avg@{k}_std": avg_at_k_std,
            "avg_sampling_std": avg_sampling_std,
            "total": total,
            "total_with_k_samples": total_samples,
            "pass_count": sum(pass_values) if pass_values else 0,
        }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="计算 Task2 的 pass@3 和 avg@3 指标"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="new_test_res/qwen3-32b",
        help="输入目录路径",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=["Task2_0_mem", "Task2_1_mem", "Task2_2_mem"],
        help="要处理的基础文件名列表（不含扩展名和采样索引）",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="计算 pass@k 和 avg@k 的 k 值（默认 3）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出结果到 JSON 文件（可选）",
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 收集所有样本的多次采样结果
    all_samples: Dict[str, SampleResult] = {}
    
    # 对于每个基础文件名，尝试读取多个采样文件
    # 假设格式是: {base_name}_0.json, {base_name}_1.json, {base_name}_2.json
    # 每个文件包含所有样本的一次采样结果
    
    all_samples_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for base_name in args.files:
        print(f"\n处理文件: {base_name}")
        
        # 尝试读取分开的采样文件
        found_any = False
        for i in range(args.k):
            file_path = input_dir / f"{base_name}_{i}.json"
            if file_path.exists():
                found_any = True
                print(f"  读取采样 {i}: {file_path.name}")
                file_samples = load_samples_from_file(file_path)
                # 合并到 all_samples_by_id
                for sample_id, items in file_samples.items():
                    # 每个文件中的每个样本代表一次采样
                    for item in items:
                        all_samples_by_id[sample_id].append(item)
        
        # 如果没有找到分开的文件，尝试读取单个文件（可能是合并的结果）
        if not found_any:
            file_path = input_dir / f"{base_name}.json"
            if file_path.exists():
                print(f"  读取文件: {file_path.name}")
                file_samples = load_samples_from_file(file_path)
                # 如果文件中有多个相同 id 的样本，每个代表一次采样
                # 如果每个 id 只有一个样本，那么这个文件代表一次采样
                for sample_id, items in file_samples.items():
                    if len(items) > 1:
                        # 多个相同 id，每个是一次采样
                        all_samples_by_id[sample_id].extend(items)
                    else:
                        # 单个样本，代表一次采样
                        all_samples_by_id[sample_id].extend(items)
        
        if not found_any and not (input_dir / f"{base_name}.json").exists():
            print(f"  警告: 未找到文件 {base_name} 的任何采样结果")
    
    if not all_samples_by_id:
        print("\n错误: 未找到任何样本数据")
        return
        
    print(f"\n总共找到 {len(all_samples_by_id)} 个样本")
    
    # 处理每个样本的多次采样结果
    for sample_id, items in all_samples_by_id.items():
        if not items:
            continue
        
        # 计算每次采样的 success
        sample_results = []
        for item in items:
            try:
                success, subtask, used_tools, used_raw = compute_sample_success_for_item(item)
                sample_results.append((success, item))
            except Exception as e:
                print(f"  警告: 处理样本 {sample_id} 时出错: {e}")
                continue
        
        if not sample_results:
            continue
        
        # 获取第一个样本的信息（用于获取 subtask 和 gt_tools）
        first_item = items[0]
        st = get_subtask_id(first_item) or 1
        gt_set, gt_raw = normalize_tool_field(first_item.get("tool"))
        if st == 3:
            gt_set = set()
            gt_raw = []
        
        # 创建 SampleResult
        all_samples[sample_id] = SampleResult(
            sample_id=sample_id,
            subtask=st,
            gt_tools=gt_set,
            gt_tool_names_raw=gt_raw,
            samples=[success for success, _ in sample_results],
            raw_samples=[item for _, item in sample_results],
        )
    
    print(f"\n总共收集到 {len(all_samples)} 个样本的采样结果")
    
    # 统计每个样本的采样次数
    sample_counts = defaultdict(int)
    for result in all_samples.values():
        sample_counts[len(result.samples)] += 1
    print("\n采样次数分布:")
    for count, num_samples in sorted(sample_counts.items()):
        print(f"  {count} 次采样: {num_samples} 个样本")
    
    # 转换为列表
    results_list = list(all_samples.values())
    
    # 计算 pass@k 和 avg@k
    metrics = compute_pass_at_k_and_avg(results_list, k=args.k)
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"Pass@{args.k} 和 Avg@{args.k} 结果")
    print(f"{'='*80}")
    
    for st_id in (1, 2, 3, 4):
        metric = metrics[st_id]
        name = metric["name"]
        pass_at_k = metric[f"pass@{args.k}"]
        avg_at_k = metric[f"avg@{args.k}"]
        pass_at_k_std = metric[f"pass@{args.k}_std"]
        avg_at_k_std = metric[f"avg@{args.k}_std"]
        avg_sampling_std = metric.get("avg_sampling_std", 0.0)
        total = metric["total"]
        total_with_k = metric["total_with_k_samples"]
        
        print(f"\n{name} (Subtask{st_id}):")
        print(f"  Pass@{args.k}: {pass_at_k:.4f} ± {pass_at_k_std:.4f} (样本间差异) ({metric['pass_count']}/{total_with_k})")
        print(f"  Avg@{args.k}:  {avg_at_k:.4f} ± {avg_at_k_std:.4f} (样本间差异)")
        print(f"           平均采样波动性: {avg_sampling_std:.4f}")
        print(f"  总样本数: {total}")
        print(f"  有 {args.k} 次采样的样本数: {total_with_k}")
    
    # 计算总体平均和标准差（基于所有样本，使用样本标准差）
    overall_pass_values = []
    overall_avg_values = []
    overall_sampling_std_list = []
    
    # 收集所有样本的 pass@k 和 avg@k 值
    for result in results_list:
        if len(result.samples) >= args.k:
            samples_k = result.samples[:args.k]
            # pass@k: 至少有一次成功（0 或 1）
            pass_value = 1 if any(samples_k) else 0
            overall_pass_values.append(pass_value)
            # avg@k: 平均成功率（0 到 1 之间的值）
            avg_value = sum(samples_k) / len(samples_k)
            overall_avg_values.append(avg_value)
            
            # 计算该样本多次采样结果的标准差
            numeric_samples = [1.0 if s else 0.0 for s in samples_k]
            if len(numeric_samples) > 1:
                sampling_std = statistics.stdev(numeric_samples)
            else:
                sampling_std = 0.0
            overall_sampling_std_list.append(sampling_std)
    
    # 计算 overall 的平均值和样本标准差
    if overall_pass_values:
        overall_pass = sum(overall_pass_values) / len(overall_pass_values)
        overall_avg = sum(overall_avg_values) / len(overall_avg_values)
        overall_pass_std = statistics.stdev(overall_pass_values) if len(overall_pass_values) > 1 else 0.0
        overall_avg_std = statistics.stdev(overall_avg_values) if len(overall_avg_values) > 1 else 0.0
        
        # 计算总体平均采样波动性
        overall_avg_sampling_std = statistics.mean(overall_sampling_std_list) if overall_sampling_std_list else 0.0
    else:
        overall_pass = 0.0
        overall_avg = 0.0
        overall_pass_std = 0.0
        overall_avg_std = 0.0
        overall_avg_sampling_std = 0.0
    
    print(f"\n{'='*80}")
    print(f"Overall Average:")
    print(f"  Pass@{args.k}: {overall_pass:.4f} ± {overall_pass_std:.4f} (样本间差异)")
    print(f"  Avg@{args.k}:  {overall_avg:.4f} ± {overall_avg_std:.4f} (样本间差异)")
    print(f"           平均采样波动性: {overall_avg_sampling_std:.4f}")
    print(f"{'='*80}")
    
    # 保存结果到文件（如果指定）
    if args.output:
        output_data = {
            "k": args.k,
            "metrics": metrics,
            "overall": {
                f"pass@{args.k}": overall_pass,
                f"avg@{args.k}": overall_avg,
                f"pass@{args.k}_std": overall_pass_std,
                f"avg@{args.k}_std": overall_avg_std,
                "avg_sampling_std": overall_avg_sampling_std,
            },
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

