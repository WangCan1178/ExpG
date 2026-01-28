#!/usr/bin/env python3
"""
分析 jsonl 文件中 tool_call_results 的 scores 分布

用法:
    python analyze_tool_scores.py <input.jsonl>
"""

import json
import sys
import argparse
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

# matplotlib 是可选的，仅用于绘图
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def extract_scores(jsonl_path):
    """从 jsonl 文件中提取所有 tool_call_results 的 scores"""
    all_scores = []  # 单个 score 值
    all_scores_arrays = []  # scores 数组
    sample_count = 0
    tool_call_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                sample_count += 1
                
                # 从 metadata 中获取 tool_call_results
                metadata = data.get('metadata', {})
                tool_call_results = metadata.get('tool_call_results', [])
                
                # 如果没有在 metadata 中，尝试直接在顶层
                if not tool_call_results:
                    tool_call_results = data.get('tool_call_results', [])
                
                for result in tool_call_results:
                    tool_call_count += 1
                    
                    # 提取单个 score
                    if 'score' in result:
                        score = result['score']
                        if isinstance(score, (int, float)):
                            all_scores.append(score)
                    
                    # 提取 scores 数组
                    if 'scores' in result:
                        scores_array = result['scores']
                        if isinstance(scores_array, list):
                            all_scores_arrays.append(scores_array)
                            
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析错误: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"警告: 第 {line_num} 行处理错误: {e}", file=sys.stderr)
                continue
    
    return all_scores, all_scores_arrays, sample_count, tool_call_count


def analyze_distribution(scores, scores_arrays):
    """分析 scores 的分布"""
    stats = {}
    
    # 单个 score 的统计
    if scores:
        scores_array = np.array(scores)
        stats['single_score'] = {
            'count': len(scores),
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
            'percentiles': {
                '25th': float(np.percentile(scores_array, 25)),
                '50th': float(np.percentile(scores_array, 50)),
                '75th': float(np.percentile(scores_array, 75)),
                '90th': float(np.percentile(scores_array, 90)),
                '95th': float(np.percentile(scores_array, 95)),
            },
            'value_counts': dict(Counter(scores)),
        }
    
    # scores 数组的统计
    if scores_arrays:
        # 展平所有 scores 数组
        flattened_scores = []
        for arr in scores_arrays:
            flattened_scores.extend(arr)
        
        if flattened_scores:
            flattened_array = np.array(flattened_scores)
            stats['scores_array'] = {
                'count': len(flattened_scores),
                'mean': float(np.mean(flattened_array)),
                'std': float(np.std(flattened_array)),
                'min': float(np.min(flattened_array)),
                'max': float(np.max(flattened_array)),
                'median': float(np.median(flattened_array)),
                'percentiles': {
                    '25th': float(np.percentile(flattened_array, 25)),
                    '50th': float(np.percentile(flattened_array, 50)),
                    '75th': float(np.percentile(flattened_array, 75)),
                    '90th': float(np.percentile(flattened_array, 90)),
                    '95th': float(np.percentile(flattened_array, 95)),
                },
                'value_counts': dict(Counter(flattened_scores)),
            }
        
        # 每个数组的统计（如果所有数组长度相同）
        array_lengths = [len(arr) for arr in scores_arrays]
        if len(set(array_lengths)) == 1:
            # 所有数组长度相同，可以按位置统计
            array_len = array_lengths[0]
            position_stats = {}
            for pos in range(array_len):
                pos_scores = [arr[pos] for arr in scores_arrays if pos < len(arr)]
                if pos_scores:
                    pos_array = np.array(pos_scores)
                    position_stats[f'position_{pos}'] = {
                        'mean': float(np.mean(pos_array)),
                        'std': float(np.std(pos_array)),
                        'min': float(np.min(pos_array)),
                        'max': float(np.max(pos_array)),
                    }
            stats['by_position'] = position_stats
    
    return stats


def plot_distribution(scores, scores_arrays, output_path=None):
    """绘制分布图"""
    if not HAS_MATPLOTLIB:
        print("警告: matplotlib 未安装，无法生成分布图", file=sys.stderr)
        return
    
    try:
        # 确定子图数量
        has_single = bool(scores)
        has_array = bool(scores_arrays)
        
        if not has_single and not has_array:
            print("警告: 没有数据可绘制", file=sys.stderr)
            return
        
        # 根据可用数据调整布局
        if has_single and has_array:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            ax_single_hist = axes[0, 0]
            ax_single_box = axes[0, 1]
            ax_array_hist = axes[1, 0]
            ax_array_box = axes[1, 1]
        elif has_single:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            ax_single_hist = axes[0]
            ax_single_box = axes[1]
            ax_array_hist = None
            ax_array_box = None
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            ax_single_hist = None
            ax_single_box = None
            ax_array_hist = axes[0]
            ax_array_box = axes[1]
        
        fig.suptitle('Tool Call Results Scores Distribution', fontsize=16)
        
        # 单个 score 的直方图
        if scores and ax_single_hist:
            # 计算合适的 bins
            unique_scores = sorted(set(scores))
            if len(unique_scores) <= 20:
                bins = len(unique_scores)
            else:
                bins = 50
            
            ax_single_hist.hist(scores, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
            ax_single_hist.set_xlabel('Score', fontsize=12)
            ax_single_hist.set_ylabel('Frequency', fontsize=12)
            ax_single_hist.set_title('Single Score Distribution', fontsize=14)
            ax_single_hist.grid(True, alpha=0.3)
        
        # 单个 score 的箱线图
        if scores and ax_single_box:
            ax_single_box.boxplot(scores, vert=True, patch_artist=True,
                                boxprops=dict(facecolor='steelblue', alpha=0.7))
            ax_single_box.set_ylabel('Score', fontsize=12)
            ax_single_box.set_title('Single Score Box Plot', fontsize=14)
            ax_single_box.grid(True, alpha=0.3)
        
        # scores 数组展平后的直方图
        if scores_arrays and ax_array_hist:
            flattened = []
            for arr in scores_arrays:
                flattened.extend(arr)
            if flattened:
                unique_scores = sorted(set(flattened))
                if len(unique_scores) <= 20:
                    bins = len(unique_scores)
                else:
                    bins = 50
                
                ax_array_hist.hist(flattened, bins=bins, edgecolor='black', alpha=0.7, color='orange')
                ax_array_hist.set_xlabel('Score', fontsize=12)
                ax_array_hist.set_ylabel('Frequency', fontsize=12)
                ax_array_hist.set_title('Flattened Scores Array Distribution', fontsize=14)
                ax_array_hist.grid(True, alpha=0.3)
        
        # scores 数组展平后的箱线图
        if scores_arrays and ax_array_box:
            flattened = []
            for arr in scores_arrays:
                flattened.extend(arr)
            if flattened:
                ax_array_box.boxplot(flattened, vert=True, patch_artist=True,
                                    boxprops=dict(facecolor='orange', alpha=0.7))
                ax_array_box.set_ylabel('Score', fontsize=12)
                ax_array_box.set_title('Flattened Scores Array Box Plot', fontsize=14)
                ax_array_box.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"分布图已保存到: {output_path}")
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        print(f"绘制分布图时出错: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='分析 jsonl 文件中 tool_call_results 的 scores 分布'
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
    parser.add_argument(
        '--plot',
        '-p',
        action='store_true',
        help='生成分布图（需要 matplotlib）'
    )
    parser.add_argument(
        '--plot-output',
        type=str,
        default=None,
        help='分布图输出路径（默认：<input_file>_scores_distribution.png）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"错误: 文件不存在: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"正在读取文件: {args.input_file}")
    all_scores, all_scores_arrays, sample_count, tool_call_count = extract_scores(args.input_file)
    
    print(f"\n统计信息:")
    print(f"  总样本数: {sample_count}")
    print(f"  总 tool_call_results 数: {tool_call_count}")
    print(f"  包含单个 score 的记录数: {len(all_scores)}")
    print(f"  包含 scores 数组的记录数: {len(all_scores_arrays)}")
    
    if not all_scores and not all_scores_arrays:
        print("\n警告: 未找到任何 scores 数据！", file=sys.stderr)
        sys.exit(1)
    
    # 分析分布
    stats = analyze_distribution(all_scores, all_scores_arrays)
    
    # 打印统计结果
    print("\n" + "="*60)
    print("分布统计结果:")
    print("="*60)
    
    if 'single_score' in stats:
        print("\n【单个 Score 统计】")
        s = stats['single_score']
        print(f"  数量: {s['count']}")
        print(f"  均值: {s['mean']:.4f}")
        print(f"  标准差: {s['std']:.4f}")
        print(f"  最小值: {s['min']}")
        print(f"  最大值: {s['max']}")
        print(f"  中位数: {s['median']:.4f}")
        print(f"  百分位数:")
        for p, v in s['percentiles'].items():
            print(f"    {p}: {v:.4f}")
        print(f"  值分布:")
        for score_val, count in sorted(s['value_counts'].items()):
            print(f"    {score_val}: {count} 次 ({count/s['count']*100:.2f}%)")
    
    if 'scores_array' in stats:
        print("\n【Scores 数组统计（展平后）】")
        s = stats['scores_array']
        print(f"  数量: {s['count']}")
        print(f"  均值: {s['mean']:.4f}")
        print(f"  标准差: {s['std']:.4f}")
        print(f"  最小值: {s['min']}")
        print(f"  最大值: {s['max']}")
        print(f"  中位数: {s['median']:.4f}")
        print(f"  百分位数:")
        for p, v in s['percentiles'].items():
            print(f"    {p}: {v:.4f}")
        print(f"  值分布:")
        for score_val, count in sorted(s['value_counts'].items()):
            print(f"    {score_val}: {count} 次 ({count/s['count']*100:.2f}%)")
    
    if 'by_position' in stats:
        print("\n【按位置统计（如果所有数组长度相同）】")
        for pos, pos_stats in stats['by_position'].items():
            print(f"  {pos}:")
            print(f"    均值: {pos_stats['mean']:.4f}")
            print(f"    标准差: {pos_stats['std']:.4f}")
            print(f"    范围: [{pos_stats['min']}, {pos_stats['max']}]")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n统计结果已保存到: {output_path}")
    
    # 生成分布图
    if args.plot:
        if not HAS_MATPLOTLIB:
            print("\n警告: matplotlib 未安装，无法生成分布图。请安装 matplotlib: pip install matplotlib", file=sys.stderr)
        else:
            plot_output = args.plot_output
            if not plot_output:
                plot_output = str(input_path.with_suffix('')) + '_scores_distribution.png'
            plot_distribution(all_scores, all_scores_arrays, plot_output)


if __name__ == '__main__':
    main()
