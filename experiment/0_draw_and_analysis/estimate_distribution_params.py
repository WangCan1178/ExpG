#!/usr/bin/env python3
"""
从 jsonl 文件估计分布参数，用于拟合 scores 的成功率分布

用法:
    python estimate_distribution_params.py <input.jsonl>
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path

# scipy 是可选的
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: scipy 未安装，将使用简化的参数估计方法", file=sys.stderr)


def extract_success_rates(jsonl_path):
    """从 jsonl 文件提取所有成功率"""
    success_rates = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # 从 metadata 中获取 tool_call_results
                metadata = data.get('metadata', {})
                tool_call_results = metadata.get('tool_call_results', [])
                
                # 如果没有在 metadata 中，尝试直接在顶层
                if not tool_call_results:
                    tool_call_results = data.get('tool_call_results', [])
                
                for result in tool_call_results:
                    if 'scores' in result:
                        scores_array = result['scores']
                        if isinstance(scores_array, list) and len(scores_array) > 0:
                            scores = np.array(scores_array)
                            # 检查是否是二进制
                            unique_vals = np.unique(scores)
                            if len(unique_vals) <= 2 and all(v in [0, 1] for v in unique_vals):
                                success_rate = float(np.sum(scores) / len(scores))
                                success_rates.append(success_rate)
                            
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析错误: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"警告: 第 {line_num} 行处理错误: {e}", file=sys.stderr)
                continue
    
    return np.array(success_rates)


def estimate_mixed_distribution(success_rates):
    """估计混合分布参数（完全成功 + Beta分布）"""
    n_total = len(success_rates)
    n_perfect = np.sum(success_rates == 1.0)
    n_failed = np.sum(success_rates == 0.0)
    n_partial = np.sum((success_rates > 0) & (success_rates < 1))
    
    # 完全成功的概率
    p_success = n_perfect / n_total if n_total > 0 else 0.0
    
    # Beta 分布参数（只使用部分成功的数据）
    partial_rates = success_rates[(success_rates > 0) & (success_rates < 1)]
    
    if len(partial_rates) > 0 and HAS_SCIPY:
        try:
            # 使用 scipy 拟合 Beta 分布
            alpha, beta, _, _ = stats.beta.fit(partial_rates, floc=0, fscale=1)
        except:
            # 如果拟合失败，使用矩估计
            mean_partial = np.mean(partial_rates)
            var_partial = np.var(partial_rates)
            if var_partial > 0:
                alpha = mean_partial * (mean_partial * (1 - mean_partial) / var_partial - 1)
                beta = (1 - mean_partial) * (mean_partial * (1 - mean_partial) / var_partial - 1)
                alpha = max(0.1, alpha)  # 避免负值
                beta = max(0.1, beta)
            else:
                alpha, beta = 2.0, 1.2
    else:
        # 简化的矩估计
        if len(partial_rates) > 0:
            mean_partial = np.mean(partial_rates)
            var_partial = np.var(partial_rates)
            if var_partial > 0:
                alpha = mean_partial * (mean_partial * (1 - mean_partial) / var_partial - 1)
                beta = (1 - mean_partial) * (mean_partial * (1 - mean_partial) / var_partial - 1)
                alpha = max(0.1, alpha)
                beta = max(0.1, beta)
            else:
                alpha, beta = 2.0, 1.2
        else:
            alpha, beta = 2.0, 1.2
    
    return {
        'p_success': float(p_success),
        'alpha': float(alpha),
        'beta': float(beta),
        'n_total': n_total,
        'n_perfect': int(n_perfect),
        'n_failed': int(n_failed),
        'n_partial': n_partial,
        'mean_partial': float(np.mean(partial_rates)) if len(partial_rates) > 0 else None,
        'std_partial': float(np.std(partial_rates)) if len(partial_rates) > 0 else None,
    }


def estimate_beta_distribution(success_rates):
    """估计 Beta 分布参数（所有数据）"""
    if HAS_SCIPY:
        try:
            alpha, beta, _, _ = stats.beta.fit(success_rates, floc=0, fscale=1)
            return {'alpha': float(alpha), 'beta': float(beta)}
        except:
            pass
    
    # 矩估计
    mean = np.mean(success_rates)
    var = np.var(success_rates)
    if var > 0:
        alpha = mean * (mean * (1 - mean) / var - 1)
        beta = (1 - mean) * (mean * (1 - mean) / var - 1)
        alpha = max(0.1, alpha)
        beta = max(0.1, beta)
        return {'alpha': float(alpha), 'beta': float(beta)}
    else:
        return {'alpha': 2.0, 'beta': 1.2}


def generate_sample_mixed(p_success, alpha, beta, n=1):
    """使用混合分布生成样本"""
    samples = []
    for _ in range(n):
        if np.random.random() < p_success:
            samples.append(1.0)
        else:
            samples.append(np.random.beta(alpha, beta))
    return np.array(samples)


def main():
    parser = argparse.ArgumentParser(
        description='从 jsonl 文件估计分布参数'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='输入的 jsonl 文件路径'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='输出参数到 JSON 文件（可选）'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='测试生成的分布是否与原始数据相似'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"错误: 文件不存在: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"正在读取文件: {args.input_file}")
    success_rates = extract_success_rates(args.input_file)
    
    if len(success_rates) == 0:
        print("错误: 未找到任何 scores 数据！", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n数据统计:")
    print(f"  总样本数: {len(success_rates)}")
    print(f"  完全成功 (1.0): {np.sum(success_rates == 1.0)} ({np.sum(success_rates == 1.0)/len(success_rates)*100:.2f}%)")
    print(f"  完全失败 (0.0): {np.sum(success_rates == 0.0)} ({np.sum(success_rates == 0.0)/len(success_rates)*100:.2f}%)")
    print(f"  部分成功: {np.sum((success_rates > 0) & (success_rates < 1))} ({np.sum((success_rates > 0) & (success_rates < 1))/len(success_rates)*100:.2f}%)")
    print(f"  平均成功率: {np.mean(success_rates):.4f} ± {np.std(success_rates):.4f}")
    
    # 估计混合分布参数
    print(f"\n" + "="*60)
    print("混合分布参数估计（推荐）:")
    print("="*60)
    mixed_params = estimate_mixed_distribution(success_rates)
    print(f"  p_success (完全成功概率): {mixed_params['p_success']:.4f}")
    print(f"  alpha (Beta分布参数): {mixed_params['alpha']:.4f}")
    print(f"  beta (Beta分布参数): {mixed_params['beta']:.4f}")
    if mixed_params['mean_partial'] is not None:
        print(f"  部分成功的均值: {mixed_params['mean_partial']:.4f} ± {mixed_params['std_partial']:.4f}")
    
    # 估计 Beta 分布参数（简化版）
    print(f"\n" + "="*60)
    print("Beta 分布参数估计（简化版）:")
    print("="*60)
    beta_params = estimate_beta_distribution(success_rates)
    print(f"  alpha: {beta_params['alpha']:.4f}")
    print(f"  beta: {beta_params['beta']:.4f}")
    
    # 生成代码示例
    print(f"\n" + "="*60)
    print("Python 代码示例:")
    print("="*60)
    print(f"""
# 混合分布模型（推荐）
import numpy as np

def generate_success_rates_mixed(n_samples):
    p_success = {mixed_params['p_success']:.4f}
    alpha = {mixed_params['alpha']:.4f}
    beta = {mixed_params['beta']:.4f}
    
    rates = []
    for _ in range(n_samples):
        if np.random.random() < p_success:
            rates.append(1.0)
        else:
            rates.append(np.random.beta(alpha, beta))
    return np.array(rates)

# 使用示例
success_rates = generate_success_rates_mixed(1000)
""")
    
    # 测试生成的分布
    if args.test:
        print(f"\n" + "="*60)
        print("测试生成的分布:")
        print("="*60)
        generated = generate_sample_mixed(
            mixed_params['p_success'],
            mixed_params['alpha'],
            mixed_params['beta'],
            n=len(success_rates)
        )
        print(f"  原始数据:")
        print(f"    完全成功: {np.sum(success_rates == 1.0)/len(success_rates)*100:.2f}%")
        print(f"    平均: {np.mean(success_rates):.4f} ± {np.std(success_rates):.4f}")
        print(f"  生成数据:")
        print(f"    完全成功: {np.sum(generated == 1.0)/len(generated)*100:.2f}%")
        print(f"    平均: {np.mean(generated):.4f} ± {np.std(generated):.4f}")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        results = {
            'mixed_distribution': mixed_params,
            'beta_distribution': beta_params,
            'data_stats': {
                'n_total': len(success_rates),
                'mean': float(np.mean(success_rates)),
                'std': float(np.std(success_rates)),
                'min': float(np.min(success_rates)),
                'max': float(np.max(success_rates)),
            }
        }
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n参数已保存到: {output_path}")


if __name__ == '__main__':
    main()
