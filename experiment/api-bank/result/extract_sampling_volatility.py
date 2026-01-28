#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取并比较原始文件和修正文件的平均采样波动性（avg_sampling_std）。

平均采样波动性：每个样本在多次采样中结果的标准差的平均值。
这个指标反映了模型在同一任务上多次采样时结果的一致性。
值越小，说明模型越稳定；值越大，说明模型输出越不稳定。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def compute_metrics(file_paths: List[Path], script_path: Path, k: int = 3) -> Dict:
    """计算指定文件的指标"""
    cmd = [
        sys.executable,
        str(script_path),
        "--files",
    ] + [str(f) for f in file_paths] + [
        "--k",
        str(k),
    ]
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        cmd_with_output = cmd + ["--output", str(tmp_path)]
        
        result = subprocess.run(
            cmd_with_output,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(file_paths[0].parent)
        )
        
        if tmp_path.exists():
            with open(tmp_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
                metrics = output_data.get("metrics", {})
            tmp_path.unlink()
            return metrics
        return {}
    except Exception as e:
        print(f"错误: {e}")
        return {}


def find_model_method_files(base_dir: Path, use_corr: bool = False) -> Dict[str, Dict[str, List[Path]]]:
    """自动发现所有模型和方法组合"""
    if use_corr:
        # 修正文件格式: qwen3-8b_0.jsonl.corr.jsonl
        pattern = "*.jsonl.corr.jsonl"
    else:
        # 原始文件格式: qwen3-8b_0.jsonl
        pattern = "*.jsonl"
        # 排除修正文件
        exclude_pattern = "*.corr.jsonl"
    
    results = defaultdict(lambda: defaultdict(list))
    
    for file_path in base_dir.glob(pattern):
        if not file_path.is_file():
            continue
        
        # 排除修正文件（当查找原始文件时）
        if not use_corr and file_path.name.endswith(".corr.jsonl"):
            continue
        
        file_name = file_path.name
        
        if use_corr:
            # 移除 .corr.jsonl 后缀，得到 qwen3-8b_0.jsonl
            base_name = file_name.replace(".corr.jsonl", "")
        else:
            base_name = file_name
        
        # 移除 .jsonl 后缀，得到 qwen3-8b_0
        base_name = base_name.replace(".jsonl", "")
        
        parts = base_name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        
        prefix, sample_idx_str = parts
        
        try:
            sample_idx = int(sample_idx_str)
        except ValueError:
            continue
        
        if prefix.startswith("run_results_8b_mem"):
            model = "qwen3_8b"
            method = "mem"
        else:
            model = prefix
            method = "baseline"
        
        if model == "qwen3-8b":
            model = "qwen3_8b"
        
        results[model][method].append((sample_idx, file_path))
    
    for model in results:
        for method in results[model]:
            results[model][method].sort(key=lambda x: x[0])
            results[model][method] = [f for _, f in results[model][method]]
    
    return dict(results)


def main():
    parser = argparse.ArgumentParser(
        description="提取并比较原始文件和修正文件的平均采样波动性"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="计算 pass@k 和 avg@k 的 k 值（默认 3）",
    )
    parser.add_argument(
        "--last-result-dir",
        type=str,
        default=None,
        help="last_result 文件夹路径",
    )
    args = parser.parse_args()
    
    if args.last_result_dir:
        last_result_dir = Path(args.last_result_dir)
    else:
        script_dir = Path(__file__).parent
        last_result_dir = script_dir / "last_result"
    
    if not last_result_dir.exists():
        print(f"错误: 找不到目录 {last_result_dir}")
        sys.exit(1)
    
    script_path = Path(__file__).parent / "compute_pass_at_k_from_jsonl.py"
    
    if not script_path.exists():
        print(f"错误: 找不到脚本 {script_path}")
        sys.exit(1)
    
    print(f"提取平均采样波动性 (avg_sampling_std)")
    print(f"{'='*100}\n")
    print(f"平均采样波动性说明:")
    print(f"  - 计算方式：对每个样本的 k 次采样结果计算标准差，然后对所有样本的标准差求平均")
    print(f"  - 值越小：模型在同一任务上多次采样时结果越一致（越稳定）")
    print(f"  - 值越大：模型输出越不稳定，多次采样结果差异越大")
    print(f"  - 取值范围：[0, 0.5]，其中 0 表示所有采样结果完全相同，0.5 表示完全随机\n")
    print(f"{'='*100}\n")
    
    # 计算原始文件
    print("正在计算原始文件...")
    orig_files = find_model_method_files(last_result_dir, use_corr=False)
    
    orig_results = {}
    for model in sorted(orig_files.keys()):
        orig_results[model] = {}
        for method in sorted(orig_files[model].keys()):
            file_paths = orig_files[model][method]
            metrics = compute_metrics(file_paths, script_path, args.k)
            orig_results[model][method] = metrics
    
    # 计算修正文件
    print("正在计算修正文件...")
    corr_files = find_model_method_files(last_result_dir, use_corr=True)
    
    corr_results = {}
    for model in sorted(corr_files.keys()):
        corr_results[model] = {}
        for method in sorted(corr_files[model].keys()):
            file_paths = corr_files[model][method]
            metrics = compute_metrics(file_paths, script_path, args.k)
            corr_results[model][method] = metrics
    
    # 打印对比结果
    print(f"\n{'='*100}")
    print(f"平均采样波动性对比 (k={args.k})")
    print(f"{'='*100}\n")
    
    header = f"{'模型':<25} {'方法':<15} {'原始文件':<20} {'修正文件':<20} {'差异':<15} {'Pass@3(原始)':<15}"
    print(header)
    print("-" * 110)
    
    all_models = set(orig_results.keys()) | set(corr_results.keys())
    
    for model in sorted(all_models):
        all_methods = set(orig_results.get(model, {}).keys()) | set(corr_results.get(model, {}).keys())
        
        for method in sorted(all_methods):
            orig_metrics = orig_results.get(model, {}).get(method, {})
            corr_metrics = corr_results.get(model, {}).get(method, {})
            
            orig_std = orig_metrics.get("avg_sampling_std", None)
            corr_std = corr_metrics.get("avg_sampling_std", None)
            orig_pass = orig_metrics.get(f"pass@{args.k}", None)
            
            orig_str = f"{orig_std:.6f}" if orig_std is not None else "N/A"
            corr_str = f"{corr_std:.6f}" if corr_std is not None else "N/A"
            
            if orig_std is not None and corr_std is not None:
                diff = corr_std - orig_std
                diff_str = f"{diff:+.6f}"
            else:
                diff_str = "N/A"
            
            pass_str = f"{orig_pass:.4f}" if orig_pass is not None else "N/A"
            
            print(
                f"{model:<25} {method:<15} {orig_str:<20} {corr_str:<20} {diff_str:<15} {pass_str:<15}"
            )
    
    print(f"\n{'='*100}")
    
    # 统计信息
    print("\n统计信息:")
    orig_stds = []
    corr_stds = []
    
    for model in all_models:
        for method in set(orig_results.get(model, {}).keys()) | set(corr_results.get(model, {}).keys()):
            orig_std = orig_results.get(model, {}).get(method, {}).get("avg_sampling_std")
            corr_std = corr_results.get(model, {}).get(method, {}).get("avg_sampling_std")
            if orig_std is not None:
                orig_stds.append(orig_std)
            if corr_std is not None:
                corr_stds.append(corr_std)
    
    if orig_stds:
        print(f"原始文件平均采样波动性:")
        print(f"  平均值: {sum(orig_stds)/len(orig_stds):.6f}")
        print(f"  最小值: {min(orig_stds):.6f}")
        print(f"  最大值: {max(orig_stds):.6f}")
    
    if corr_stds:
        print(f"修正文件平均采样波动性:")
        print(f"  平均值: {sum(corr_stds)/len(corr_stds):.6f}")
        print(f"  最小值: {min(corr_stds):.6f}")
        print(f"  最大值: {max(corr_stds):.6f}")


if __name__ == "__main__":
    main()
