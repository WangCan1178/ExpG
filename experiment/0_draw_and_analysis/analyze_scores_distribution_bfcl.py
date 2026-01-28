#!/usr/bin/env python3
"""
分析 jsonl 文件中每个样本的 scores 数组分布特征
判断是否为长尾分布或正态分布，并与 test_sample_method.py 中的分布进行对比

用法:
    python analyze_scores_distribution.py <input.jsonl>
"""

import json
import sys
import argparse
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

# scipy 是可选的
try:
    from scipy import stats
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _calculate_skewness(data):
    """计算偏度（不依赖 scipy）"""
    if HAS_SCIPY:
        return stats.skew(data)
    else:
        # 手动计算偏度
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        n = len(data)
        skew = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
        return skew


def _calculate_kurtosis(data):
    """计算峰度（不依赖 scipy）"""
    if HAS_SCIPY:
        return stats.kurtosis(data)
    else:
        # 手动计算峰度（excess kurtosis）
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        n = len(data)
        kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        return kurt


def extract_scores_per_sample(jsonl_path):
    """从 jsonl 文件中提取每个样本的 scores 数组"""
    samples_scores = []  # 每个样本的 scores 数组列表
    
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
                
                # 收集该样本的所有 scores 数组
                sample_scores = []
                for result in tool_call_results:
                    if 'scores' in result:
                        scores_array = result['scores']
                        if isinstance(scores_array, list) and len(scores_array) > 0:
                            sample_scores.append(scores_array)
                
                if sample_scores:
                    samples_scores.append(sample_scores)
                            
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析错误: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"警告: 第 {line_num} 行处理错误: {e}", file=sys.stderr)
                continue
    
    return samples_scores


def test_normality(data):
    """测试数据是否符合正态分布"""
    if not HAS_SCIPY or len(data) < 3:
        return None
    
    try:
        # Shapiro-Wilk 检验（适用于小样本）
        if len(data) <= 5000:
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
        else:
            # 对于大样本，使用 Kolmogorov-Smirnov 检验
            stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            test_name = "Kolmogorov-Smirnov"
        
        return {'statistic': float(stat), 'p_value': float(p_value), 'test_name': test_name}
    except:
        return None


def fit_exponential_decay(data):
    """拟合指数衰减分布: y = a * exp(-b * x)"""
    if len(data) < 2:
        return None
    
    x = np.arange(len(data))
    y = np.array(data)
    
    # 确保数据非负
    if np.any(y < 0):
        y = y - np.min(y) + 1e-10
    
    try:
        # 使用对数线性回归拟合指数衰减
        # log(y) = log(a) - b*x
        log_y = np.log(y + 1e-10)
        coeffs = np.polyfit(x, log_y, 1)
        a = np.exp(coeffs[1])
        b = -coeffs[0]
        
        # 计算 R²
        y_pred = a * np.exp(-b * x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'a': float(a),
            'b': float(b),
            'r_squared': float(r_squared),
            'decay_rate': float(b)
        }
    except:
        return None


def analyze_single_scores_array(scores_array):
    """分析单个 scores 数组的分布特征"""
    if len(scores_array) == 0:
        return None
    
    scores = np.array(scores_array)
    unique_values = np.unique(scores)
    
    analysis = {
        'length': len(scores),
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'median': float(np.median(scores)),
        'unique_count': len(unique_values),
        'is_binary': len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values),
    }
    
    # 如果是二进制分布（0和1），进行特殊分析
    if analysis['is_binary']:
        ones_count = int(np.sum(scores))
        zeros_count = len(scores) - ones_count
        success_rate = ones_count / len(scores)
        
        analysis['binary'] = {
            'ones_count': ones_count,
            'zeros_count': zeros_count,
            'success_rate': float(success_rate),
        }
        
        # 二项分布检验：检查是否可以用固定的成功概率 p 来建模
        # 如果所有样本的成功率都接近某个值，说明可以用二项分布
        analysis['is_uniform_binary'] = success_rate == 0.0 or success_rate == 1.0 or abs(success_rate - 0.5) < 0.1
    else:
        # 对于非二进制数据，进行常规分析
        analysis['skewness'] = float(_calculate_skewness(scores))
        analysis['kurtosis'] = float(_calculate_kurtosis(scores))
        
        # 正态性检验（仅对非二进制数据有意义）
        if len(unique_values) > 2:
            normality = test_normality(scores)
            if normality:
                analysis['normality'] = normality
        
        # 指数衰减拟合（如果数据看起来像是有序的）
        is_decreasing = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        is_increasing = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
        
        if is_decreasing or is_increasing:
            exp_fit = fit_exponential_decay(scores if is_decreasing else scores[::-1])
            if exp_fit and exp_fit['r_squared'] > 0:  # 只保留有意义的拟合
                analysis['exponential_fit'] = exp_fit
    
    # 计算长尾特征：检查高值占比
    sorted_scores = np.sort(scores)[::-1]  # 降序
    top_20_percent = sorted_scores[:max(1, len(scores) // 5)]
    analysis['top_20_percent_mean'] = float(np.mean(top_20_percent))
    analysis['top_20_percent_ratio'] = float(np.sum(top_20_percent) / np.sum(scores)) if np.sum(scores) > 0 else 0
    
    # 计算熵（用于衡量分布的均匀性）
    # 对于二进制数据，使用伯努利熵
    if analysis['is_binary']:
        p = analysis['binary']['success_rate']
        if 0 < p < 1:
            entropy = -(p * np.log(p) + (1-p) * np.log(1-p))
        else:
            entropy = 0
        max_entropy = np.log(2)  # 二进制最大熵
        analysis['entropy'] = float(entropy)
        analysis['normalized_entropy'] = float(entropy / max_entropy) if max_entropy > 0 else 0
    else:
        # 将 scores 归一化为概率分布
        scores_normalized = scores / (np.sum(scores) + 1e-10)
        entropy = -np.sum(scores_normalized * np.log(scores_normalized + 1e-10))
        max_entropy = np.log(len(scores))
        analysis['entropy'] = float(entropy)
        analysis['normalized_entropy'] = float(entropy / max_entropy) if max_entropy > 0 else 0
    
    return analysis


def analyze_across_samples_distribution(samples_scores):
    """分析多个 scores 数组之间的分布（如成功率分布）"""
    # 提取每个 scores 数组的成功率（1的比例）
    success_rates = []
    for sample_scores_list in samples_scores:
        for scores_array in sample_scores_list:
            if len(scores_array) > 0:
                scores = np.array(scores_array)
                # 检查是否是二进制
                if len(np.unique(scores)) <= 2 and all(v in [0, 1] for v in np.unique(scores)):
                    success_rate = float(np.sum(scores) / len(scores))
                    success_rates.append(success_rate)
    
    if not success_rates:
        return None
    
    success_rates = np.array(success_rates)
    
    analysis = {
        'count': len(success_rates),
        'mean': float(np.mean(success_rates)),
        'std': float(np.std(success_rates)),
        'min': float(np.min(success_rates)),
        'max': float(np.max(success_rates)),
        'median': float(np.median(success_rates)),
        'skewness': float(_calculate_skewness(success_rates)),
        'kurtosis': float(_calculate_kurtosis(success_rates)),
    }
    
    # 正态性检验
    normality = test_normality(success_rates)
    if normality:
        analysis['normality'] = normality
    
    # 检查是否可以用指数衰减分布来建模
    # 将成功率按降序排列，看是否符合指数衰减
    sorted_rates = np.sort(success_rates)[::-1]
    exp_fit = fit_exponential_decay(sorted_rates)
    if exp_fit:
        analysis['exponential_fit'] = exp_fit
    
    # 长尾特征：检查高成功率占比
    top_20_percent = sorted_rates[:max(1, len(sorted_rates) // 5)]
    analysis['top_20_percent_mean'] = float(np.mean(top_20_percent))
    analysis['top_20_percent_ratio'] = float(np.sum(top_20_percent) / np.sum(sorted_rates)) if np.sum(sorted_rates) > 0 else 0
    
    return analysis


def compare_with_test_distribution(across_samples_analysis):
    """与 test_sample_method.py 中的指数衰减分布进行对比"""
    comparison = {}
    
    if across_samples_analysis and 'exponential_fit' in across_samples_analysis:
        exp_fit = across_samples_analysis['exponential_fit']
        comparison['decay_rate'] = exp_fit['decay_rate']
        comparison['r_squared'] = exp_fit['r_squared']
        comparison['test_method_decay_rate'] = 0.7
        comparison['decay_rate_difference'] = abs(exp_fit['decay_rate'] - 0.7)
        comparison['fit_quality'] = 'good' if exp_fit['r_squared'] > 0.7 else 'moderate' if exp_fit['r_squared'] > 0.5 else 'poor'
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='分析 jsonl 文件中每个样本的 scores 数组分布特征'
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
        help='输出统计结果的 JSON 文件路径（可选）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"错误: 文件不存在: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"正在读取文件: {args.input_file}")
    samples_scores = extract_scores_per_sample(args.input_file)
    
    print(f"\n统计信息:")
    print(f"  总样本数（包含 scores）: {len(samples_scores)}")
    total_arrays = sum(len(sample) for sample in samples_scores)
    print(f"  总 scores 数组数: {total_arrays}")
    
    if not samples_scores:
        print("\n警告: 未找到任何 scores 数据！", file=sys.stderr)
        sys.exit(1)
    
    # 分析每个 scores 数组
    print("\n正在分析每个 scores 数组的分布特征...")
    all_analyses = []
    for sample_idx, sample_scores_list in enumerate(samples_scores):
        for array_idx, scores_array in enumerate(sample_scores_list):
            analysis = analyze_single_scores_array(scores_array)
            if analysis:
                analysis['sample_idx'] = sample_idx
                analysis['array_idx'] = array_idx
                all_analyses.append(analysis)
    
    # 汇总统计
    print("\n" + "="*60)
    print("分布特征汇总:")
    print("="*60)
    
    # 二进制分布统计
    binary_analyses = [a for a in all_analyses if a.get('is_binary', False)]
    if binary_analyses:
        print(f"\n【二进制分布（0/1）特征】")
        print(f"  二进制数组数: {len(binary_analyses)} ({len(binary_analyses)/len(all_analyses)*100:.2f}%)")
        success_rates = [a['binary']['success_rate'] for a in binary_analyses]
        print(f"  平均成功率（1的比例）: {np.mean(success_rates):.4f} ± {np.std(success_rates):.4f}")
        print(f"  成功率范围: [{np.min(success_rates):.4f}, {np.max(success_rates):.4f}]")
        print(f"  完全成功 (100%): {sum(1 for r in success_rates if r == 1.0)} ({sum(1 for r in success_rates if r == 1.0)/len(success_rates)*100:.2f}%)")
        print(f"  完全失败 (0%): {sum(1 for r in success_rates if r == 0.0)} ({sum(1 for r in success_rates if r == 0.0)/len(success_rates)*100:.2f}%)")
        print(f"  混合 (0% < p < 100%): {sum(1 for r in success_rates if 0 < r < 1)} ({sum(1 for r in success_rates if 0 < r < 1)/len(success_rates)*100:.2f}%)")
    
    # 初始化变量
    normality_results = []
    p_values = []
    significant_norm = 0
    exp_fits = []
    decay_rates = []
    r_squareds = []
    top_20_ratios = []
    long_tail_count = 0
    skewnesses = []
    kurtoses = []
    
    # 非二进制分布统计
    non_binary_analyses = [a for a in all_analyses if not a.get('is_binary', False)]
    if non_binary_analyses:
        print(f"\n【非二进制分布特征】")
        print(f"  非二进制数组数: {len(non_binary_analyses)} ({len(non_binary_analyses)/len(all_analyses)*100:.2f}%)")
        
        # 正态性检验结果汇总
        normality_results = [a['normality'] for a in non_binary_analyses if 'normality' in a]
        if normality_results:
            p_values = [r['p_value'] for r in normality_results]
            significant_norm = sum(1 for p in p_values if p < 0.05)
            print(f"  正态性检验:")
            print(f"    检验的数组数: {len(normality_results)}")
            print(f"    拒绝正态性假设 (p < 0.05): {significant_norm} ({significant_norm/len(normality_results)*100:.2f}%)")
            print(f"    接受正态性假设 (p >= 0.05): {len(normality_results) - significant_norm} ({(len(normality_results) - significant_norm)/len(normality_results)*100:.2f}%)")
            print(f"    平均 p 值: {np.mean(p_values):.4f}")
    
        if decay_rates:
            print(f"\n【指数衰减拟合（非二进制数据）】")
            print(f"  可拟合的数组数: {len(decay_rates)}")
            print(f"  平均衰减率: {np.mean(decay_rates):.4f} ± {np.std(decay_rates):.4f}")
            print(f"  衰减率范围: [{np.min(decay_rates):.4f}, {np.max(decay_rates):.4f}]")
            print(f"  平均 R²: {np.mean(r_squareds):.4f}")
            print(f"  test_sample_method.py 使用的衰减率: 0.7")
            print(f"  与测试方法的衰减率差异: {abs(np.mean(decay_rates) - 0.7):.4f}")
    
    # 长尾特征汇总
    top_20_ratios = [a['top_20_percent_ratio'] for a in all_analyses if 'top_20_percent_ratio' in a]
    if top_20_ratios:
        print(f"\n【长尾特征】")
        print(f"  平均 top 20% 占比: {np.mean(top_20_ratios):.4f} ± {np.std(top_20_ratios):.4f}")
        print(f"  (如果 > 0.5，说明分布是长尾的)")
        long_tail_count = sum(1 for r in top_20_ratios if r > 0.5)
        print(f"  长尾分布数组数: {long_tail_count} ({long_tail_count/len(top_20_ratios)*100:.2f}%)")
    
    # 指数衰减拟合结果汇总（仅非二进制数据）
    exp_fits = [a['exponential_fit'] for a in non_binary_analyses if 'exponential_fit' in a]
    if exp_fits:
        decay_rates = [f['decay_rate'] for f in exp_fits if f['r_squared'] > 0]
        r_squareds = [f['r_squared'] for f in exp_fits if f['r_squared'] > 0]
    
    # 熵统计（衡量分布均匀性）
    entropies = [a['normalized_entropy'] for a in all_analyses if 'normalized_entropy' in a]
    if entropies:
        print(f"\n【分布均匀性（归一化熵）】")
        print(f"  平均归一化熵: {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")
        print(f"  (1.0 = 完全均匀, 0.0 = 完全集中)")
        uniform_count = sum(1 for e in entropies if e > 0.8)
        print(f"  接近均匀分布 (熵 > 0.8): {uniform_count} ({uniform_count/len(entropies)*100:.2f}%)")
    
    # 偏度和峰度（仅非二进制数据）
    if non_binary_analyses:
        skewnesses = [a['skewness'] for a in non_binary_analyses if 'skewness' in a]
        kurtoses = [a['kurtosis'] for a in non_binary_analyses if 'kurtosis' in a]
    if skewnesses:
        print(f"\n【偏度（Skewness，非二进制数据）】")
        print(f"  平均偏度: {np.mean(skewnesses):.4f} ± {np.std(skewnesses):.4f}")
        print(f"  (正偏度 = 右偏/长尾, 负偏度 = 左偏)")
        positive_skew = sum(1 for s in skewnesses if s > 0)
        print(f"  正偏度数组数: {positive_skew} ({positive_skew/len(skewnesses)*100:.2f}%)")
    
    if kurtoses:
        print(f"\n【峰度（Kurtosis，非二进制数据）】")
        print(f"  平均峰度: {np.mean(kurtoses):.4f} ± {np.std(kurtoses):.4f}")
        print(f"  (高峰度 = 尖峰分布, 低峰度 = 平缓分布)")
    
    # 分析多个 scores 数组之间的分布（重点关注长尾分布）
    print("\n" + "="*60)
    print("多个 Scores 数组之间的分布分析（长尾分布检测）:")
    print("="*60)
    
    across_samples_analysis = analyze_across_samples_distribution(samples_scores)
    if across_samples_analysis:
        # 长尾分布特征分析
        print(f"\n【长尾分布特征分析】")
        
        # 1. 偏度（长尾分布应该是正偏度）
        skewness = across_samples_analysis['skewness']
        print(f"  1. 偏度: {skewness:.4f}")
        if skewness > 0.5:
            print(f"     ✓ 正偏度，符合长尾分布特征（右偏/长尾）")
        elif skewness > 0:
            print(f"     ~ 轻微正偏度，可能具有长尾特征")
        else:
            print(f"     ✗ 负偏度，不符合长尾分布特征（左偏）")
        
        # 2. Top 20% 占比（帕累托原则：长尾分布中 top 20% 应该占大部分）
        top_20_ratio = across_samples_analysis['top_20_percent_ratio']
        print(f"\n  2. Top 20% 占比: {top_20_ratio:.4f} ({top_20_ratio*100:.2f}%)")
        if top_20_ratio > 0.5:
            print(f"     ✓ 符合长尾分布（top 20% 占比 > 50%，符合帕累托原则）")
        elif top_20_ratio > 0.4:
            print(f"     ~ 可能具有长尾特征（top 20% 占比 > 40%）")
        else:
            print(f"     ✗ 不符合长尾分布（top 20% 占比 < 50%）")
        
        # 3. 指数衰减拟合（长尾分布可以用指数衰减来拟合）
        if 'exponential_fit' in across_samples_analysis:
            exp_fit = across_samples_analysis['exponential_fit']
            r_squared = exp_fit['r_squared']
            decay_rate = exp_fit['decay_rate']
            print(f"\n  3. 指数衰减拟合:")
            print(f"     衰减率: {decay_rate:.4f}")
            print(f"     R²: {r_squared:.4f}")
            if r_squared > 0.7:
                print(f"     ✓ 拟合质量良好，可以用指数衰减分布建模（R² > 0.7）")
            elif r_squared > 0.5:
                print(f"     ~ 拟合质量中等，可能可以用指数衰减分布建模（0.5 < R² < 0.7）")
            else:
                print(f"     ✗ 拟合质量较差，不适合用指数衰减分布建模（R² < 0.5）")
        
        # 综合判断
        print(f"\n【综合判断】")
        long_tail_indicators = 0
        total_indicators = 3
        
        if skewness > 0:
            long_tail_indicators += 1
        if top_20_ratio > 0.4:
            long_tail_indicators += 1
        if 'exponential_fit' in across_samples_analysis and exp_fit['r_squared'] > 0.5:
            long_tail_indicators += 1
        
        if long_tail_indicators >= 2:
            print(f"  ✓ 结论: 符合长尾分布特征 ({long_tail_indicators}/{total_indicators} 个指标符合)")
        elif long_tail_indicators == 1:
            print(f"  ~ 结论: 可能具有长尾分布特征 ({long_tail_indicators}/{total_indicators} 个指标符合)")
        else:
            print(f"  ✗ 结论: 不符合长尾分布特征 ({long_tail_indicators}/{total_indicators} 个指标符合)")
        
        # 额外统计信息（简要）
        print(f"\n【简要统计】")
        print(f"  分析的 scores 数组数: {across_samples_analysis['count']}")
        print(f"  成功率范围: [{across_samples_analysis['min']:.4f}, {across_samples_analysis['max']:.4f}]")
        print(f"  平均成功率: {across_samples_analysis['mean']:.4f} ± {across_samples_analysis['std']:.4f}")
    
    # 与 test_sample_method.py 的对比（仅当符合长尾分布时）
    print("\n" + "="*60)
    print("与 test_sample_method.py 的分布对比:")
    print("="*60)
    
    comparison = compare_with_test_distribution(across_samples_analysis)
    if comparison:
        print(f"\n【指数衰减率对比】")
        print(f"  实际数据衰减率: {comparison['decay_rate']:.4f}")
        print(f"  test_sample_method.py 衰减率: {comparison['test_method_decay_rate']:.4f}")
        print(f"  差异: {comparison['decay_rate_difference']:.4f}")
        print(f"  拟合质量: {comparison['fit_quality']} (R² = {comparison['r_squared']:.4f})")
        
        if comparison['r_squared'] > 0.7 and comparison['decay_rate_difference'] < 0.2:
            print(f"\n  ✓ 结论: 可以用 test_sample_method.py 中的指数衰减分布来拟合")
            print(f"     - 衰减率接近 (差异 < 0.2)")
            print(f"     - 拟合质量良好 (R² > 0.7)")
        elif comparison['r_squared'] > 0.5:
            print(f"\n  ~ 结论: 可能可以用 test_sample_method.py 中的指数衰减分布来拟合")
            print(f"     - 拟合质量中等 (R² > 0.5)")
            print(f"     - 但衰减率差异: {comparison['decay_rate_difference']:.4f}")
        else:
            print(f"\n  ✗ 结论: 不适合用 test_sample_method.py 中的指数衰减分布来拟合")
            print(f"     - 拟合质量较差 (R² < 0.5)")
            print(f"     - 或者衰减率差异较大")
    else:
        print(f"\n  无法进行对比（数据不符合指数衰减分布）")
    
    # 保存结果
    results = {
        'summary': {
            'total_samples': len(samples_scores),
            'total_arrays': total_arrays,
            'analyzed_arrays': len(all_analyses),
        },
        'normality': {
            'total_tests': len(normality_results),
            'rejected_normality': significant_norm if normality_results else 0,
            'mean_p_value': float(np.mean(p_values)) if normality_results else None,
        },
        'exponential_fit': {
            'fittable_arrays': len(exp_fits) if exp_fits else 0,
            'mean_decay_rate': float(np.mean(decay_rates)) if exp_fits else None,
            'mean_r_squared': float(np.mean(r_squareds)) if exp_fits else None,
        },
        'long_tail': {
            'mean_top_20_ratio': float(np.mean(top_20_ratios)) if top_20_ratios else None,
            'long_tail_count': long_tail_count if top_20_ratios else 0,
        },
        'comparison': comparison,
        'across_samples_analysis': across_samples_analysis,
        'all_analyses': all_analyses[:100] if len(all_analyses) > 100 else all_analyses,  # 只保存前100个详细结果
    }
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
