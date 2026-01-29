#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理任务ID和函数的关系
第一步：提取每个任务id用到的function及其schema
第二步：找用到了相同function的其他任务id
第三步：剔除找到的其他任务id不足2个的任务
第四步：对每个类别最多保留50个任务（随机采样）
"""

import json
import os
import re
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Any
import hashlib


def normalize_function_schema(func: Dict[str, Any]) -> str:
    """将函数的schema标准化为字符串，用于比较"""
    # 提取函数的关键信息：name, description, parameters
    schema = {
        "name": func.get("name", ""),
        "description": func.get("description", ""),
        "parameters": func.get("parameters", {})
    }
    # 使用json序列化并排序，确保相同schema得到相同的hash
    schema_str = json.dumps(schema, sort_keys=True)
    return schema_str


def extract_category_from_task_id(task_id: str) -> str:
    """
    从任务ID提取测试类别。

    Args:
        task_id: 任务ID，例如 "live_relevance_7-7-0" 或 "simple_python_1"

    Returns:
        测试类别，例如 "live_relevance" 或 "simple_python"
    """
    return task_id.rsplit('_', 1)[0]


def get_function_signature(func: Dict[str, Any]) -> str:
    """获取函数的唯一签名（用于标识相同的函数）"""
    schema_str = normalize_function_schema(func)
    # 使用hash来生成一个较短的标识符
    return hashlib.md5(schema_str.encode()).hexdigest()[:16]


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件（每行一个JSON对象）"""
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {file_path}: {e}")
                    continue
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """保存数据到JSONL文件"""
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def filename_to_classname(filename: str) -> str:
    """
    将文件名转换为类名
    例如: "gorilla_file_system.json" -> "GorillaFileSystem"
         "twitter_api.json" -> "TwitterAPI"
    """
    # 去掉.json后缀，按_分割，每个单词首字母大写，然后拼接
    name_without_ext = filename[:-5] if filename.endswith('.json') else filename
    parts = name_without_ext.split('_')
    # 每个部分首字母大写
    class_name = ''.join(part.capitalize() for part in parts)
    return class_name


def load_multi_turn_tools(multi_turn_func_doc_dir: str) -> Dict[str, Dict]:
    """
    加载multi_turn_func_doc文件夹中的所有工具定义
    返回: {full_function_name: function_dict}
    例如: {"GorillaFileSystem.cd": {...}, "TwitterAPI.post_tweet": {...}}
    """
    tools = {}
    
    if not os.path.exists(multi_turn_func_doc_dir):
        return tools
    
    # 遍历multi_turn_func_doc目录下的所有json文件
    for filename in os.listdir(multi_turn_func_doc_dir):
        if not filename.endswith('.json'):
            continue
        
        # 从文件名转换为类名
        class_name = filename_to_classname(filename)
        
        file_path = os.path.join(multi_turn_func_doc_dir, filename)
        func_definitions = load_jsonl(file_path)
        
        for func_def in func_definitions:
            func_name = func_def.get('name')
            if func_name:
                # 构建完整的函数名：ClassName.function_name
                full_func_name = f"{class_name}.{func_name}"
                tools[full_func_name] = func_def
                # 同时存储短名称，方便匹配（可能多个类有同名函数，保留最后一个）
                tools[func_name] = func_def
    
    return tools


def parse_function_call(call_str: str) -> str:
    """
    从函数调用字符串中提取函数名
    例如: "cd(folder='document')" -> "cd"
         "GorillaFileSystem.mv(source='file', dest='dir')" -> "GorillaFileSystem.mv"
    """
    # 匹配函数调用格式：function_name(...)
    match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(', call_str.strip())
    if match:
        return match.group(1)
    return None


def step1_extract_functions(data_dir: str, possible_answer_dir: str, multi_turn_func_doc_dir: str, output_file: str):
    """
    第一步：提取每个任务id用到的function及其schema
    
    对于irrelevance任务（没有possible_answer），使用data文件中的function
    对于多轮任务（multi_turn），从multi_turn_func_doc加载工具定义，从ground_truth解析函数调用
    """
    print("开始第一步：提取每个任务id用到的function...")
    
    # 加载多轮任务的工具定义
    multi_turn_tools = load_multi_turn_tools(multi_turn_func_doc_dir)
    print(f"  加载了 {len([k for k in multi_turn_tools.keys() if '.' in k])} 个多轮工具定义")
    
    # 获取data目录下所有json文件（不包括子文件夹）
    data_files = [f for f in os.listdir(data_dir) 
                  if f.endswith('.json') and os.path.isfile(os.path.join(data_dir, f))]
    
    # 加载所有possible_answer文件，建立id到答案的映射
    possible_answer_map = {}
    if os.path.exists(possible_answer_dir):
        for filename in os.listdir(possible_answer_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(possible_answer_dir, filename)
                answers = load_jsonl(file_path)
                for answer in answers:
                    task_id = answer.get('id')
                    if task_id:
                        possible_answer_map[task_id] = answer
    
    # 存储每个任务id用到的函数
    task_functions = {}  # task_id -> List[function_dict]
    
    # 存储所有遇到的函数schema（用于去重同名不同schema的函数）
    function_signatures = {}  # signature -> function_dict
    
    # 遍历所有data文件
    for filename in data_files:
        if filename == 'config.json':  # 跳过config.json
            continue
            
        file_path = os.path.join(data_dir, filename)
        print(f"  处理文件: {filename}")
        
        tasks = load_jsonl(file_path)
        
        for task in tasks:
            task_id = task.get('id')
            if not task_id:
                continue
            
            # 检查是否是irrelevance任务或多轮任务
            is_irrelevance = 'irrelevance' in filename.lower()
            is_multi_turn = 'multi_turn' in filename.lower()
            
            # 获取用到的函数
            used_functions = []
            
            if is_multi_turn and task_id in possible_answer_map:
                # 多轮任务：从ground_truth解析函数调用，从multi_turn_func_doc找到对应的schema
                answer = possible_answer_map[task_id]
                ground_truth = answer.get('ground_truth', [])
                
                # 从task中获取path字段（可用函数的完整路径列表）
                path_list = task.get('path', [])
                # 建立短函数名到完整路径的映射
                short_to_full = {}
                for full_path in path_list:
                    if '.' in full_path:
                        class_name, func_name = full_path.rsplit('.', 1)
                        if func_name not in short_to_full:
                            short_to_full[func_name] = []
                        short_to_full[func_name].append(full_path)
                
                # 记录已使用的函数签名，避免重复添加
                used_signatures = set()
                used_func_names = set()
                
                # 遍历所有轮次的ground_truth
                for turn_calls in ground_truth:
                    if isinstance(turn_calls, list):
                        for call_str in turn_calls:
                            if not isinstance(call_str, str):
                                continue
                            
                            # 解析函数调用，获取函数名
                            func_name_short = parse_function_call(call_str)
                            if not func_name_short:
                                continue
                            
                            # 如果已经处理过这个函数，跳过
                            if func_name_short in used_func_names:
                                continue
                            
                            # 找到对应的完整函数路径和定义
                            func_def = None
                            
                            # 先尝试直接匹配完整路径
                            if func_name_short in multi_turn_tools:
                                func_def = multi_turn_tools[func_name_short]
                            
                            # 如果没找到，尝试通过短名称映射
                            if not func_def and func_name_short in short_to_full:
                                for full_path in short_to_full[func_name_short]:
                                    if full_path in multi_turn_tools:
                                        func_def = multi_turn_tools[full_path]
                                        break
                            
                            if func_def:
                                signature = get_function_signature(func_def)
                                if signature not in used_signatures:
                                    used_functions.append(func_def)
                                    used_signatures.add(signature)
                                    used_func_names.add(func_name_short)
            
            elif is_irrelevance or task_id not in possible_answer_map:
                # irrelevance任务或没有答案，使用data文件中的function
                functions = task.get('function', [])
                for func in functions:
                    if func:
                        used_functions.append(func)
            else:
                # 从possible_answer中提取用到的函数
                answer = possible_answer_map[task_id]
                ground_truth = answer.get('ground_truth', [])
                
                # ground_truth是一个列表，每个元素是 {function_name: {param: value}}
                # 需要从原始task的function列表中找到对应的函数定义
                # 注意：同一个任务可能有多个同名但不同schema的函数，需要都保留
                original_functions_by_name = defaultdict(list)  # func_name -> List[func_dict]
                for func in task.get('function', []):
                    func_name = func.get('name')
                    if func_name:
                        original_functions_by_name[func_name].append(func)
                
                # 记录已使用的函数签名，避免重复添加
                used_signatures = set()
                
                for gt_item in ground_truth:
                    # gt_item是字典，key是函数名，value是参数
                    if isinstance(gt_item, dict):
                        for func_name, params in gt_item.items():
                            # 在原始函数列表中查找所有同名函数（可能有多版本schema）
                            if func_name in original_functions_by_name:
                                # 添加所有同名函数（包括不同schema的版本）
                                # 因为同名不同schema的函数应视作不同的函数
                                for func in original_functions_by_name[func_name]:
                                    signature = get_function_signature(func)
                                    if signature not in used_signatures:
                                        used_functions.append(func)
                                        used_signatures.add(signature)
            
            # 为每个函数创建唯一标识符（考虑schema）
            function_list_with_schema = []
            for func in used_functions:
                signature = get_function_signature(func)
                # 存储函数签名和完整schema
                function_signatures[signature] = {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                    "signature": signature
                }
                function_list_with_schema.append({
                    "signature": signature,
                    "schema": func  # 完整的函数schema
                })
            
            task_functions[task_id] = function_list_with_schema
    
    # 生成中间文件1
    output_data = []
    for task_id, functions in task_functions.items():
        output_data.append({
            "task_id": task_id,
            "functions": functions,
            "function_count": len(functions)
        })
    
    save_jsonl(output_data, output_file)
    print(f"第一步完成，共处理 {len(task_functions)} 个任务")
    print(f"中间文件1已保存到: {output_file}")
    
    return task_functions, function_signatures


def step2_find_related_tasks(task_functions: Dict[str, List], output_file: str):
    """
    第二步：对于每个任务id，找用到了相同function的其他所有任务id，按工具分类存储
    
    如果一个任务用了3个工具a、b、c，那么输出：
    {
        "task_id": "...",
        "functions": [
            {
                "signature": "sig_a",
                "function_name": "...",
                "related_task_ids": [...],  # 使用工具a的其他任务
                "related_count": N
            },
            ...
        ]
    }
    """
    print("\n开始第二步：找用到了相同function的其他任务id（按工具分类）...")
    
    # 建立函数签名到任务id的映射
    signature_to_tasks = defaultdict(set)  # signature -> Set[task_id]
    
    for task_id, functions in task_functions.items():
        for func_info in functions:
            signature = func_info.get('signature')
            if signature:
                signature_to_tasks[signature].add(task_id)
    
    # 对于每个任务，找到用到了相同函数的所有其他任务，按工具分类
    task_related_tasks_by_func = {}  # task_id -> List[{signature, function_name, related_task_ids, related_count}]
    
    for task_id, functions in task_functions.items():
        functions_with_related = []
        
        # 遍历该任务使用的所有函数
        for func_info in functions:
            signature = func_info.get('signature')
            schema = func_info.get('schema', {})
            function_name = schema.get('name', '')
            
            if signature:
                # 找到所有使用相同函数的任务（排除自己）
                tasks_with_same_func = signature_to_tasks[signature] - {task_id}
                related_task_ids = sorted(list(tasks_with_same_func))
                
                functions_with_related.append({
                    "signature": signature,
                    "function_name": function_name,
                    "related_task_ids": related_task_ids,
                    "related_count": len(related_task_ids)
                })
        
        task_related_tasks_by_func[task_id] = functions_with_related
    
    # 生成中间文件2
    output_data = []
    for task_id, functions_info in task_related_tasks_by_func.items():
        # 计算所有相关任务的并集（用于统计）
        all_related = set()
        for func_info in functions_info:
            all_related.update(func_info['related_task_ids'])
        
        output_data.append({
            "task_id": task_id,
            "functions": functions_info,
            "total_related_count": len(all_related)
        })
    
    save_jsonl(output_data, output_file)
    print(f"第二步完成，共处理 {len(task_related_tasks_by_func)} 个任务")
    print(f"中间文件2已保存到: {output_file}")
    
    return task_related_tasks_by_func


def step3_filter_minimum_tasks(task_related_tasks_by_func: Dict[str, List], output_file: str):
    """
    第三步：剔除所有工具的相关任务数不足2个的任务
    
    要求：一个任务使用的所有工具，每个工具都必须有>=2个其他任务也在使用。
    也就是说，如果一个任务用了3个工具a、b、c，那么：
    - 使用工具a的其他任务数必须>=2
    - 使用工具b的其他任务数必须>=2
    - 使用工具c的其他任务数必须>=2
    只有所有工具都满足条件，这个任务才被保留。
    """
    print("\n开始第三步：剔除所有工具的相关任务数不足2个的任务...")
    
    filtered_tasks = {}
    
    for task_id, functions_info in task_related_tasks_by_func.items():
        # 检查该任务使用的所有工具，是否每个工具的相关任务数都>=2
        all_meet_requirement = True
        
        for func_info in functions_info:
            if func_info['related_count'] < 2:
                all_meet_requirement = False
                break
        
        # 只有所有工具都满足条件，才保留这个任务
        if all_meet_requirement and len(functions_info) > 0:
            filtered_tasks[task_id] = functions_info
    
    # 生成中间文件3（格式与step2相同）
    output_data = []
    for task_id, functions_info in filtered_tasks.items():
        # 计算所有相关任务的并集（用于统计）
        all_related = set()
        for func_info in functions_info:
            all_related.update(func_info['related_task_ids'])
        
        output_data.append({
            "task_id": task_id,
            "functions": functions_info,
            "total_related_count": len(all_related)
        })
    
    save_jsonl(output_data, output_file)
    print(f"第三步完成，保留 {len(filtered_tasks)} 个任务（原 {len(task_related_tasks_by_func)} 个）")
    print(f"中间文件3已保存到: {output_file}")
    
    return filtered_tasks


def step4_sample_tasks(input_file: str, output_file: str, max_per_category: int = 50):
    """
    第四步：对于每个类别，如果任务数量超过max_per_category，随机选择max_per_category个任务

    Args:
        input_file: step3_filtered_tasks.jsonl文件路径
        output_file: 输出文件路径
        max_per_category: 每个类别最大任务数
    """
    print(f"\n开始第四步：对每个类别最多保留 {max_per_category} 个任务...")

    # 读取step3的输出
    tasks = load_jsonl(input_file)

    # 按类别分组
    category_to_tasks = defaultdict(list)

    for task in tasks:
        task_id = task['task_id']
        category = extract_category_from_task_id(task_id)
        category_to_tasks[category].append(task)

    # 对于每个类别，如果超过max_per_category，随机选择
    sampled_tasks = []

    for category, tasks_in_category in category_to_tasks.items():
        if len(tasks_in_category) > max_per_category:
            sampled = random.sample(tasks_in_category, max_per_category)
            print(f"类别 {category}: 从 {len(tasks_in_category)} 个任务中随机选择 {max_per_category} 个")
        else:
            sampled = tasks_in_category
            print(f"类别 {category}: 保留所有 {len(tasks_in_category)} 个任务")

        sampled_tasks.extend(sampled)

    # 保存到新文件
    save_jsonl(sampled_tasks, output_file)
    print(f"第四步完成，共保留 {len(sampled_tasks)} 个任务")
    print(f"中间文件4已保存到: {output_file}")

    return sampled_tasks


def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    possible_answer_dir = base_dir / "data" / "possible_answer"
    multi_turn_func_doc_dir = base_dir / "data" / "multi_turn_func_doc"
    
    # 输出文件路径
    output_dir = base_dir / "scripts" / "intermediate_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file1 = output_dir / "step1_task_functions.jsonl"
    output_file2 = output_dir / "step2_related_tasks.jsonl"
    output_file3 = output_dir / "step3_filtered_tasks.jsonl"
    output_file4 = output_dir / "step4_sampled_tasks.jsonl"
    
    # 执行三步处理
    task_functions, function_signatures = step1_extract_functions(
        str(data_dir), 
        str(possible_answer_dir),
        str(multi_turn_func_doc_dir),
        str(output_file1)
    )
    
    task_related_tasks_by_func = step2_find_related_tasks(
        task_functions,
        str(output_file2)
    )
    
    filtered_tasks = step3_filter_minimum_tasks(
        task_related_tasks_by_func,
        str(output_file3)
    )

    sampled_tasks = step4_sample_tasks(
        str(output_file3),
        str(output_file4),
        max_per_category=50
    )

    print("\n所有处理完成！")
    print(f"第三步保留 {len(filtered_tasks)} 个任务，这些任务的related_task_ids数量 >= 2")
    print(f"第四步采样后最终保留 {len(sampled_tasks)} 个任务，每个类别最多50个")


if __name__ == "__main__":
    main()

