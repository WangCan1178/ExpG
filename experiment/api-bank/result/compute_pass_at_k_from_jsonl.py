#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 pass@3 和 avg@3 指标，基于 JSONL 格式的结果文件。

参考 compare_results.py 的判定方式（使用 correct 字段）和
compute_pass_at_k.py 的计算方式（计算 pass@k 和 avg@k）。

使用方法:
python compute_pass_at_k_from_jsonl.py --files run_results_8b_0.jsonl run_results_8b_1.jsonl run_results_8b_2.jsonl
"""

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SampleResult:
    """单个样本的多次采样结果"""
    file: str
    sample_id: int
    samples: List[bool]  # 每次采样的 success 结果（基于 correct 字段）
    raw_samples: List[Dict[str, Any]]  # 原始采样数据


def load_jsonl(file_path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    加载 JSONL 文件，返回以 (file, sample_id) 为 key 的字典。
    参考 compare_results.py 的加载方式。
    """
    data = {}
    if not file_path.exists():
        print(f"警告: 文件不存在: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    key = (record['file'], record['sample_id'])
                    data[key] = record
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"警告: 跳过无效记录: {e}")
                    continue
    
    return data


def is_valid_record(record: Dict[str, Any]) -> bool:
    """
    判断记录是否有效。
    参考 compare_results.py 的过滤逻辑：跳过 level-3。
    """
    if 'level-3' in str(record.get('file', '')):
        return False
    # if 'ToolSearcher' in str(record.get('ground_truth', {})):
        # return False
    return True


def get_success_from_record(record: Dict[str, Any]) -> Optional[bool]:
    """
    从记录中提取 success 结果。
    参考 compare_results.py 的判定方式：使用 correct 字段。
    只处理 API 类型的记录。
    """
    record_type = record.get('type', '')
    
    # 只处理 API 类型的记录
    if record_type != 'API':
        return None
    
    # 使用 correct 字段判断
    correct = record.get('correct')
    if correct is True:
        return True
    elif correct is False:
        return False
    else:
        # correct 字段不存在或为 None
        return None


def compute_pass_at_k_and_avg(
    results: List[SampleResult], 
    k: int = 3
) -> Dict[str, float]:
    """
    计算 pass@k 和 avg@k，以及它们的标准差。
    
    pass@k: 在 k 次采样中，至少有一次成功的比例
    avg@k: k 次采样的平均成功率
    
    参考 compute_pass_at_k.py 的计算方式。
    
    返回: {"pass@k": value, "avg@k": value, "pass@k_std": value, "avg@k_std": value, 
          "avg_sampling_std": value, "total": count}
    """
    pass_values = []
    avg_values = []
    sampling_std_list = []  # 每个样本多次采样的标准差列表
    total_samples = 0
    
    for result in results:
        # 只考虑有 k 次采样的样本
        if len(result.samples) >= k:
            total_samples += 1
            samples_k = result.samples[:k]
            
            # pass@k: 至少有一次成功（0 或 1）
            pass_value = 1 if any(samples_k) else 0
            pass_values.append(pass_value)
            
            # avg@k: 平均成功率（0 到 1 之间的值）
            avg_value = sum(samples_k) / len(samples_k)
            avg_values.append(avg_value)
            
            # 计算该样本多次采样结果的标准差（将布尔值转为0/1）
            numeric_samples = [1.0 if s else 0.0 for s in samples_k]
            if len(numeric_samples) > 1:
                sampling_std = statistics.stdev(numeric_samples)
            else:
                sampling_std = 0.0
            sampling_std_list.append(sampling_std)
    
    # 计算最终指标
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
    
    return {
        f"pass@{k}": pass_at_k,
        f"avg@{k}": avg_at_k,
        f"pass@{k}_std": pass_at_k_std,
        f"avg@{k}_std": avg_at_k_std,
        "avg_sampling_std": avg_sampling_std,
        "total": len(results),
        "total_with_k_samples": total_samples,
        "pass_count": sum(pass_values) if pass_values else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="计算 pass@3 和 avg@3 指标（基于 JSONL 格式）"
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="要处理的 JSONL 文件列表（多次采样结果）",
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
    
    # 加载所有文件
    all_data: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    
    for file_path_str in args.files:
        file_path = Path(file_path_str)
        print(f"加载文件: {file_path.name}")
        
        file_data = load_jsonl(file_path)
        print(f"  读取到 {len(file_data)} 条记录")
        
        # 将每条记录添加到对应的 (file, sample_id) 下
        for key, record in file_data.items():
            if is_valid_record(record):
                all_data[key].append(record)
    
    print(f"\n总共找到 {len(all_data)} 个唯一样本")
    
    # 处理每个样本的多次采样结果
    sample_results: List[SampleResult] = []
    
    for (file, sample_id), records in all_data.items():
        # 提取每次采样的 success 结果
        samples = []
        raw_samples = []
        
        for record in records:
            success = get_success_from_record(record)
            if success is not None:  # 只保留有效的 API 记录
                samples.append(success)
                raw_samples.append(record)
        
        if samples:  # 至少有一次有效采样
            sample_results.append(
                SampleResult(
                    file=file,
                    sample_id=sample_id,
                    samples=samples,
                    raw_samples=raw_samples,
                )
            )
    
    print(f"总共收集到 {len(sample_results)} 个样本的采样结果")
    
    # 统计每个样本的采样次数
    sample_counts = defaultdict(int)
    for result in sample_results:
        sample_counts[len(result.samples)] += 1
    print("\n采样次数分布:")
    for count, num_samples in sorted(sample_counts.items()):
        print(f"  {count} 次采样: {num_samples} 个样本")
    
    # 计算 pass@k 和 avg@k
    metrics = compute_pass_at_k_and_avg(sample_results, k=args.k)
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"Pass@{args.k} 和 Avg@{args.k} 结果")
    print(f"{'='*80}")
    
    pass_at_k = metrics[f"pass@{args.k}"]
    avg_at_k = metrics[f"avg@{args.k}"]
    pass_at_k_std = metrics[f"pass@{args.k}_std"]
    avg_at_k_std = metrics[f"avg@{args.k}_std"]
    avg_sampling_std = metrics.get("avg_sampling_std", 0.0)
    total = metrics["total"]
    total_with_k = metrics["total_with_k_samples"]
    pass_count = metrics["pass_count"]
    
    print(f"\nOverall:")
    print(f"  Pass@{args.k}: {pass_at_k:.4f} ± {pass_at_k_std:.4f} (样本间差异) ({pass_count}/{total_with_k})")
    print(f"  Avg@{args.k}:  {avg_at_k:.4f} ± {avg_at_k_std:.4f} (样本间差异)")
    print(f"           平均采样波动性: {avg_sampling_std:.4f}")
    print(f"  总样本数: {total}")
    print(f"  有 {args.k} 次采样的样本数: {total_with_k}")
    print(f"{'='*80}")
    
    # 保存结果到文件（如果指定）
    if args.output:
        output_data = {
            "k": args.k,
            "metrics": metrics,
            "files": args.files,
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

