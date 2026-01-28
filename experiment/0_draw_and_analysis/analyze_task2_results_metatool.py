#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析Task2结果文件，比较不同方法的性能
"""

import json
import os
from collections import defaultdict
from pathlib import Path

def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return None

def extract_predicted_tool(item):
    """从结果项中提取预测的工具"""
    # 检查action_res字段
    action_res = item.get('action_res', '')
    if action_res:
        # 尝试从action_res中提取工具名称
        # action_res可能是字符串，需要解析
        if isinstance(action_res, str):
            # 去除空白字符和换行
            tool = action_res.strip().strip('\n').strip()
            # 如果包含"None"，返回None
            if tool.lower() == 'none' or not tool:
                return None
            return tool
        elif isinstance(action_res, dict):
            # 如果是字典，尝试找到工具字段
            return action_res.get('tool') or action_res.get('tool_name')
    return None

def is_task_correct(predicted_tool, ground_truth_tool):
    """判断任务是否正确"""
    if predicted_tool is None and ground_truth_tool is None:
        return True
    if predicted_tool is None or ground_truth_tool is None:
        return False
    # 比较工具名称（忽略大小写和空白）
    return predicted_tool.strip().lower() == ground_truth_tool.strip().lower()

def analyze_method_results(data_dir, method_name, ground_truth_data):
    """分析某个方法的结果"""
    # 构建任务ID到正确答案的映射
    task_id_to_gt = {}
    for item in ground_truth_data:
        task_id = item.get('id')
        gt_tool = item.get('tool')
        if task_id:
            task_id_to_gt[task_id] = gt_tool
    
    # 存储每个任务的3次采样结果
    task_results = defaultdict(list)
    
    # 读取3次采样的结果
    for idx in [0, 1, 2]:
        file_name = f"Task2_{idx}_{method_name}.json" if method_name else f"Task2_{idx}.json"
        file_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在")
            continue
        
        data = load_json_file(file_path)
        if data is None:
            continue
        
        # 处理每个任务
        for item in data:
            task_id = item.get('id')
            if not task_id:
                continue
            
            predicted_tool = extract_predicted_tool(item)
            gt_tool = task_id_to_gt.get(task_id)
            
            if gt_tool is None:
                continue
            
            is_correct = is_task_correct(predicted_tool, gt_tool)
            task_results[task_id].append(is_correct)
    
    # 判断每个任务是否至少有一次做对
    correct_tasks = set()
    wrong_tasks = set()
    
    for task_id, results in task_results.items():
        if any(results):  # 至少有一次做对
            correct_tasks.add(task_id)
        else:
            wrong_tasks.add(task_id)
    
    return correct_tasks, wrong_tasks, task_results

def main():
    data_dir = "/Users/xiaocan/Toolmem/experiment/MetaTool/src/generation/tool/new_test_res/qwen3-8b"
    
    # 加载ground truth数据（使用Task2_0.json作为参考，因为它应该包含所有任务）
    ground_truth_file = os.path.join(data_dir, "Task2_0.json")
    ground_truth_data = load_json_file(ground_truth_file)
    
    if ground_truth_data is None:
        print("无法加载ground truth数据")
        return
    
    # 分析方法列表
    methods = {
        'baseline': '',  # 没有方法后缀的
        'm0': 'm0',
        'mem': 'mem',
        'few': 'fewshot'
    }
    
    # 分析每个方法
    method_results = {}
    for method_name, method_suffix in methods.items():
        print(f"\n正在分析方法: {method_name}")
        correct_tasks, wrong_tasks, task_results = analyze_method_results(
            data_dir, method_suffix, ground_truth_data
        )
        method_results[method_name] = {
            'correct': correct_tasks,
            'wrong': wrong_tasks,
            'task_results': task_results
        }
        print(f"  做对的任务数: {len(correct_tasks)}")
        print(f"  做错的任务数: {len(wrong_tasks)}")
    
    # 比较每个方法相对于baseline的结果
    baseline_correct = method_results['baseline']['correct']
    baseline_wrong = method_results['baseline']['wrong']
    
    print("\n" + "="*80)
    print("各方法相对于baseline的分析结果")
    print("="*80)
    
    for method_name in ['m0', 'mem', 'fewshot']:
        if method_name not in method_results:
            continue
        
        method_correct = method_results[method_name]['correct']
        method_wrong = method_results[method_name]['wrong']
        
        # 相对于baseline做对的任务（baseline做错，但该方法做对）
        relative_correct = method_correct - baseline_correct
        
        # 相对于baseline做错的任务（baseline做对，但该方法做错）
        relative_wrong = method_wrong - baseline_wrong
        
        print(f"\n方法: {method_name}")
        print(f"  总任务数: {len(method_correct | method_wrong)}")
        print(f"  做对的任务数: {len(method_correct)}")
        print(f"  做错的任务数: {len(method_wrong)}")
        print(f"\n  相对于baseline:")
        print(f"    额外做对的任务数: {len(relative_correct)}")
        print(f"    额外做错的任务数: {len(relative_wrong)}")
        
        if relative_correct:
            print(f"\n  额外做对的任务ID (前20个):")
            for i, task_id in enumerate(sorted(relative_correct)[:20]):
                print(f"    {task_id}")
            if len(relative_correct) > 20:
                print(f"    ... 还有 {len(relative_correct) - 20} 个任务")
        
        if relative_wrong:
            print(f"\n  额外做错的任务ID (前20个):")
            for i, task_id in enumerate(sorted(relative_wrong)[:20]):
                print(f"    {task_id}")
            if len(relative_wrong) > 20:
                print(f"    ... 还有 {len(relative_wrong) - 20} 个任务")

if __name__ == "__main__":
    main()


