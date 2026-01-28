#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计多轮任务的平均轮数和步数
"""
import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_jsonl_file(file_path):
    """分析单个JSONL文件"""
    turns_list = []
    total_steps_list = []  # 总步数（所有轮次加起来）
    steps_per_turn_list = []  # 每轮平均步数
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                # 从result字段获取轮数和步数
                turns = None
                total_steps = None
                avg_steps_per_turn = None
                
                # result是一个列表的列表，外层列表长度代表轮数，内层列表长度代表每轮的步数
                if 'result' in data and isinstance(data['result'], list):
                    turns = len(data['result'])  # 轮数 = result列表的长度
                    # 总步数 = 所有轮次的步数之和
                    total_steps = sum(len(turn) for turn in data['result'] if isinstance(turn, list))
                    # 每轮平均步数 = 总步数 / 轮数
                    if turns > 0:
                        avg_steps_per_turn = total_steps / turns
                
                if turns is not None:
                    turns_list.append(turns)
                if total_steps is not None:
                    total_steps_list.append(total_steps)
                if avg_steps_per_turn is not None:
                    steps_per_turn_list.append(avg_steps_per_turn)
                    
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行JSON解析失败: {e}")
                continue
    
    return turns_list, total_steps_list, steps_per_turn_list

def main():
    folder_path = "/Users/xiaocan/Toolmem/experiment/gorilla/berkeley-function-call-leaderboard/result_old/result_filter_50/qwen3-8b-FC/multi_turn"
    
    all_turns = []
    all_total_steps = []  # 总步数（所有轮次加起来）
    all_steps_per_turn = []  # 每轮平均步数
    file_stats = {}
    
    # 遍历文件夹中的所有JSON文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            print(f"\n正在分析: {file_name}")
            
            turns, total_steps, steps_per_turn = analyze_jsonl_file(file_path)
            
            if turns:
                avg_turns = sum(turns) / len(turns)
                file_stats[file_name] = {
                    'turns': {
                        'count': len(turns),
                        'avg': avg_turns,
                        'min': min(turns),
                        'max': max(turns)
                    }
                }
                all_turns.extend(turns)
                print(f"  轮数统计: 任务数={len(turns)}, 平均={avg_turns:.2f}, 最小={min(turns)}, 最大={max(turns)}")
            
            if total_steps:
                avg_total_steps = sum(total_steps) / len(total_steps)
                if file_name not in file_stats:
                    file_stats[file_name] = {}
                file_stats[file_name]['total_steps'] = {
                    'count': len(total_steps),
                    'avg': avg_total_steps,
                    'min': min(total_steps),
                    'max': max(total_steps)
                }
                all_total_steps.extend(total_steps)
                print(f"  总步数统计(所有轮次加起来): 任务数={len(total_steps)}, 平均={avg_total_steps:.2f}, 最小={min(total_steps)}, 最大={max(total_steps)}")
            
            if steps_per_turn:
                avg_steps_per_turn = sum(steps_per_turn) / len(steps_per_turn)
                if file_name not in file_stats:
                    file_stats[file_name] = {}
                file_stats[file_name]['steps_per_turn'] = {
                    'count': len(steps_per_turn),
                    'avg': avg_steps_per_turn,
                    'min': min(steps_per_turn),
                    'max': max(steps_per_turn)
                }
                all_steps_per_turn.extend(steps_per_turn)
                print(f"  每轮平均步数: 任务数={len(steps_per_turn)}, 平均={avg_steps_per_turn:.2f}, 最小={min(steps_per_turn):.2f}, 最大={max(steps_per_turn):.2f}")
    
    # 汇总统计
    print("\n" + "="*60)
    print("总体统计:")
    print("="*60)
    
    if all_turns:
        avg_turns = sum(all_turns) / len(all_turns)
        print(f"平均轮数: {avg_turns:.2f}")
        print(f"  总任务数: {len(all_turns)}")
        print(f"  最小轮数: {min(all_turns)}")
        print(f"  最大轮数: {max(all_turns)}")
    
    if all_total_steps:
        avg_total_steps = sum(all_total_steps) / len(all_total_steps)
        print(f"\n平均总步数(所有轮次加起来): {avg_total_steps:.2f}")
        print(f"  总任务数: {len(all_total_steps)}")
        print(f"  最小总步数: {min(all_total_steps)}")
        print(f"  最大总步数: {max(all_total_steps)}")
    
    if all_steps_per_turn:
        avg_steps_per_turn = sum(all_steps_per_turn) / len(all_steps_per_turn)
        print(f"\n平均每轮步数: {avg_steps_per_turn:.2f}")
        print(f"  总任务数: {len(all_steps_per_turn)}")
        print(f"  最小每轮步数: {min(all_steps_per_turn):.2f}")
        print(f"  最大每轮步数: {max(all_steps_per_turn):.2f}")
    
    if not all_turns and not all_total_steps:
        print("未找到轮数或步数字段，请检查数据格式")

if __name__ == "__main__":
    main()

