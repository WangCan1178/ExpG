#!/usr/bin/env python3
"""
脚本用于处理数据集，划分训练集和测试集，并运行评估流程。

主要流程：
1. 读取文件，划分测试集和训练集
2. 在训练集上运行生成和评估
3. 从评估结果中提取工具调用信息，计算hash并用hash代替工具名称
4. 根据method参数进行相应处理（mem/m0/draft）
5. 对测试集进行生成和评估

"""
import traceback
import json
import subprocess
import sys
import os
import re
import hashlib
import requests
import threading
import random
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 添加项目路径以便导入 bfcl_eval 模块
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from bfcl_eval.constants.eval_config import PROJECT_ROOT, TEST_IDS_TO_GENERATE_PATH, RESULT_PATH, SCORE_PATH
from bfcl_eval.utils import extract_test_category_from_id, is_multi_turn, load_file, populate_test_cases_with_predefined_functions, calculate_tool_hash
from bfcl_eval.constants.category_mapping import NON_LIVE_CATEGORY, LIVE_CATEGORY, MULTI_TURN_CATEGORY


def extract_category_from_task_id(task_id: str) -> str:
    """
    从任务 ID 提取测试类别。
    
    Args:
        task_id: 任务 ID，例如 "live_relevance_7-7-0" 或 "simple_python_1"
    
    Returns:
        测试类别，例如 "live_relevance" 或 "simple_python"
    """
    return extract_test_category_from_id(task_id, remove_prereq=False)


def collect_all_task_ids(entry: dict) -> set[str]:
    """
    从一条数据中收集所有相关的任务 ID。
    
    Args:
        entry: 从输入文件读取的一条数据
    
    Returns:
        包含所有任务 ID 的集合（包括主 task_id 和所有 related_task_ids）
    """
    task_ids = {entry["task_id"]}
    
    # 收集所有 functions 中的 related_task_ids
    # 对于超过5个 related_task_ids 的 functions，随机保留5个
    for func in entry.get("functions", []):
        related_ids = func.get("related_task_ids", [])
        if len(related_ids) > 10:
            # 随机选择5个
            related_ids = random.sample(related_ids, 10)
        task_ids.update(related_ids)
    
    return task_ids


def group_task_ids_by_category(task_ids: set[str]) -> dict[str, list[str]]:
    """
    按测试类别对任务 ID 进行分组。
    
    Args:
        task_ids: 任务 ID 集合
    
    Returns:
        按类别分组的字典，格式为 {category: [task_id1, task_id2, ...]}
    """
    category_to_ids = defaultdict(list)
    
    for task_id in task_ids:
        category = extract_category_from_task_id(task_id)
        category_to_ids[category].append(task_id)
    
    # 对每个类别的 ID 列表进行排序
    for category in category_to_ids:
        category_to_ids[category].sort()
    
    return dict(category_to_ids)


def generate_test_case_ids_file(category_to_ids: dict[str, list[str]], output_path: Path):
    """
    生成 test_case_ids_to_generate.json 文件。
    
    Args:
        category_to_ids: 按类别分组的任务 ID 字典
        output_path: 输出文件路径
    """
    # 确保所有可能的类别都存在（即使为空列表）
    all_categories = [
        "simple_python", "simple_java", "simple_javascript",
        "irrelevance", "parallel", "multiple", "parallel_multiple",
        "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
        "live_irrelevance", "live_relevance",
        "multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param",
        "multi_turn_long_context", "multi_turn_composite",
        "memory_kv", "memory_vector", "memory_rec_sum",
        "web_search_base", "web_search_no_snippet",
        "format_sensitivity"
    ]
    
    result = {}
    for category in all_categories:
        result[category] = category_to_ids.get(category, [])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


def run_generation(model: str, additional_args: list[str] = None, result_dir: Path = None, method: str = None):
    """
    运行生成命令。

    Args:
        model: 模型名称
        additional_args: 额外的命令行参数列表
        result_dir: 结果目录路径，如果为None则使用RESULT_PATH
        method: 方法名称（例如："mem", "m0", "draft"），如果为None则不添加--method参数
    """
    if result_dir is None:
        result_dir = RESULT_PATH
    
    cmd = [
        sys.executable, "-u", "-m", "bfcl_eval", "generate",
        "--model", model,
        "--run-ids",
        "--result-dir", str(result_dir)
    ]
    
    if method:
        cmd.extend(["--method", method])

    if additional_args:
        cmd.extend(additional_args)

    print(f"🚀 运行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ 生成完成")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ 生成失败: {e}")
        return False


def run_evaluation_command(model: str, result_dir: Path = None, method: str = None):
    """
    运行评估命令。
    
    Args:
        model: 模型名称
        result_dir: 结果目录路径，如果为None则使用RESULT_PATH
        method: 方法名称，用于确定score_dir的后缀
    """
    if result_dir is None:
        result_dir = RESULT_PATH
    
    # 根据method获取对应的score_dir
    score_dir = get_score_dir_for_method(method)
    
    cmd = [
        sys.executable, "-m", "bfcl_eval", "evaluate",
        "--partial-eval",
        "--score-dir", str(score_dir),
        "--result-dir", str(result_dir),
        "--model", model
    ]

    print(f"🔍 运行评估命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ 评估完成")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ 评估失败: {e}")
        return False


def get_result_dir_for_method(method: str = None) -> Path:
    """
    根据method获取对应的result目录。
    
    Args:
        method: 方法名称（"mem"/"m0"/"draft"），如果为空则返回RESULT_PATH
    
    Returns:
        result目录路径
    """
    if not method:
        return RESULT_PATH
    
    # 根据method添加后缀
    method_suffix = method.lower()
    result_dir_name = f"result_{method_suffix}"
    return PROJECT_ROOT / result_dir_name


def get_score_dir_for_method(method: str = None) -> Path:
    """
    根据method获取对应的score目录。
    
    Args:
        method: 方法名称（"mem"/"m0"/"draft"），如果为空则返回SCORE_PATH
    
    Returns:
        score目录路径
    """
    if not method:
        return SCORE_PATH
    
    # 根据method添加后缀
    method_suffix = method.lower()
    score_dir_name = f"score_{method_suffix}"
    return PROJECT_ROOT / score_dir_name


# calculate_tool_hash 已移至 bfcl_eval.utils 中，使用统一的实现


def get_tool_config_from_dataset(task_id: str, tool_name: str) -> dict | None:
    """
    从原始数据集中获取工具的config。
    
    Args:
        task_id: 任务ID
        tool_name: 工具名称（简写形式）
    
    Returns:
        工具config字典，如果找不到返回None
    """
    try:
        # 确定任务类别
        category = extract_category_from_task_id(task_id)

        # 构建原始数据集文件路径
        data_file = PROJECT_ROOT / "bfcl_eval" / "data" / f"BFCL_v4_{category}.json"
        if not data_file.exists():
            return None

        # 加载原始数据集
        dataset = load_file(data_file)

        # 查找对应的任务
        for item in dataset:
            if item.get("id") == task_id:
                # 对于多轮任务，需要先填充 function 字段
                if is_multi_turn(task_id):
                    # 使用 populate_test_cases_with_predefined_functions 填充多轮任务的函数文档
                    test_cases = populate_test_cases_with_predefined_functions([item])
                    if test_cases:
                        item = test_cases[0]
                
                # 找到了对应的任务，统一从function字段提取工具信息
                function_data = item.get("function", [])
                if isinstance(function_data, list):
                    for func in function_data:
                        if isinstance(func, dict) and "name" in func:
                            doc_tool_name = func["name"]
                            # 支持完整名称和简写名称匹配
                            if doc_tool_name == tool_name or doc_tool_name.endswith(f".{tool_name}"):
                                # 提取schema（去掉name字段）
                                schema = {k: v for k, v in func.items() if k != "name"}
                                return schema if schema else None
                break

    except Exception as e:
        print(f"⚠️ 提取工具config时出错: {e}")

    return None


# 以下函数从 run_filtered_tasks_fc.py 复制并修改

def is_structured_tool_result(model_result_raw) -> bool:
    """检查model_result_raw是否为FC模型的结构化输出"""
    if isinstance(model_result_raw, dict):
        return True
    if isinstance(model_result_raw, list) and model_result_raw:
        return all(isinstance(item, dict) for item in model_result_raw if item is not None)
    
    # 如果是字符串，尝试解析为JSON，看是否是结构化的dict或list[dict]格式
    if isinstance(model_result_raw, str):
        raw_str = model_result_raw.strip()
        if not raw_str:
            return False
        # 尝试解析为JSON
        try:
            parsed = json.loads(raw_str)
            if isinstance(parsed, dict):
                return True
            if isinstance(parsed, list) and parsed:
                return all(isinstance(item, dict) for item in parsed if item is not None)
        except (json.JSONDecodeError, ValueError):
            pass
    
    return False


def parse_structured_tool_input(raw_input):
    """解析FC模型结构化输出中的参数"""
    if raw_input is None:
        return {}

    if isinstance(raw_input, (dict, list)):
        return raw_input

    if isinstance(raw_input, (int, float, bool)):
        return raw_input

    if isinstance(raw_input, str):
        raw_str = raw_input.strip()
        if not raw_str:
            return {}
        try:
            parsed = json.loads(raw_str)
            return parsed
        except json.JSONDecodeError:
            return {"value": raw_input}

    return {"value": raw_input}


def build_tool_call_result(tool_name: str, context: str, tool_input, output: str, config_path: Path, task_id: str | None):
    """统一构造工具调用结果"""
    normalized_output = "Success" if output == "" else output
    # 根据 output 判断 success：如果包含 [Error] 则失败，否则成功
    is_success = not str(normalized_output).startswith("[Error]")
    tool_call_result = {
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool_name": tool_name,
        "context": context,
        "input": tool_input if tool_input is not None else {},
        "output": str(normalized_output),
        "config": str(config_path) if config_path else "",
        "token_cost": 10,
        "success": is_success,
        "time_cost": 0.3,
    }
    if task_id:
        tool_call_result["source_task_id"] = task_id
    return tool_call_result


def split_parallel_tool_calls(model_result_raw: str) -> list[str]:
    """拆分并行任务中的多个工具调用"""
    if not model_result_raw or not isinstance(model_result_raw, str):
        return []

    if not model_result_raw.startswith('[') or not model_result_raw.endswith(']'):
        return [model_result_raw]

    content = model_result_raw[1:-1].strip()
    if not content:
        return []

    if ',' not in content:
        return [f"[{content}]"]

    tool_calls = []
    current_call = ""
    paren_count = 0
    bracket_count = 0
    in_string = False
    string_char = ""

    for i, char in enumerate(content):
        if char in ['"', "'"] and (i == 0 or content[i-1] != '\\'):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = ""
        elif not in_string:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
            elif char == ',' and paren_count == 0 and bracket_count == 0:
                tool_call = current_call.strip()
                if tool_call:
                    tool_calls.append(f"[{tool_call}]")
                current_call = ""
                continue
        current_call += char

    if current_call.strip():
        tool_calls.append(f"[{current_call.strip()}]")

    return tool_calls if tool_calls else [model_result_raw]


def get_error_for_tool_index(error, tool_index: int) -> str:
    """
    为特定索引的工具获取对应的错误信息
    只有明确指向该索引的错误才返回错误，否则返回"Success"
    
    错误格式说明（来自ast_checker.py的parallel_function_checker_no_order）：
    - 对于并行任务，error是一个列表
    - 第一个元素是总领错误信息，格式："Could not find a matching function among index {considered_indices} of model output for index {i} of possible answers."
    - 其中 considered_indices 是未被匹配的索引列表（已匹配的索引不在其中，说明它们成功了）
    - 后续元素是字典，格式为：{"Model Result Index {index}": {"sub_error": [...]}}
    - 只有明确在总领错误信息的 considered_indices 中或字典中出现的索引才表示该工具出错
    """
    if error == "Success":
        return "Success"
    
    if isinstance(error, list):
        # 首先查找是否有明确指定该索引的字典错误
        # 格式：{"Model Result Index {index}": {"sub_error": [...]}}
        has_dict_error = False
        dict_error_message = ""
        for item in error:
            if isinstance(item, dict):
                for key, value in item.items():
                    # 检查key中是否包含当前索引（如 "Model Result Index 1"）
                    # 提取key中的索引数字
                    index_match = re.search(r'Index\s+(\d+)', key, re.IGNORECASE)
                    if index_match:
                        error_index = int(index_match.group(1))
                        if error_index == tool_index:
                            # 找到了该索引的错误
                            has_dict_error = True
                            if isinstance(value, dict) and "sub_error" in value:
                                sub_errors = value["sub_error"]
                                if isinstance(sub_errors, list):
                                    dict_error_message = "[Error] " + "; ".join(str(e) for e in sub_errors)
                                else:
                                    dict_error_message = "[Error] " + str(sub_errors)
                            else:
                                dict_error_message = "[Error] " + str(value)
                            break
        
        if has_dict_error:
            return dict_error_message
        
        # 检查总领错误信息中的 considered_indices
        # 格式："Could not find a matching function among index [1] of model output for index 1 of possible answers."
        error_messages = []
        for item in error:
            if isinstance(item, str):
                error_messages.append(item)
        
        if error_messages:
            # 查找总领错误信息，提取 considered_indices
            for msg in error_messages:
                # 匹配 "among index [1]" 或 "among index [0, 1]" 等模式
                # 提取方括号中的索引列表
                match = re.search(r'among index\s+(\[[\d\s,]+\]|\d+)', msg, re.IGNORECASE)
                if match:
                    indices_str = match.group(1)
                    # 解析索引列表
                    if indices_str.startswith('[') and indices_str.endswith(']'):
                        # 提取方括号中的数字
                        indices_content = indices_str[1:-1]
                        # 分割并转换为整数列表
                        try:
                            error_indices = [int(x.strip()) for x in indices_content.split(',') if x.strip()]
                            if tool_index in error_indices:
                                # 当前索引在错误列表中，返回错误
                                # 如果有字典错误，优先返回字典错误，否则返回总领错误
                                if has_dict_error:
                                    return dict_error_message
                                else:
                                    return "[Error] " + msg
                        except ValueError:
                            pass
                    else:
                        # 单个数字
                        try:
                            error_index = int(indices_str.strip())
                            if error_index == tool_index:
                                if has_dict_error:
                                    return dict_error_message
                                else:
                                    return "[Error] " + msg
                        except ValueError:
                            pass
            
            # 如果没有找到总领错误信息中的索引匹配，检查是否是全局错误
            # 全局错误关键词：语法错误、解码错误等会影响所有工具
            global_error_keywords = ["Invalid syntax", "Failed to decode", "decoder_failed", "syntax", "Wrong number"]
            is_global_error = any(keyword in msg for msg in error_messages for keyword in global_error_keywords)
            
            if is_global_error:
                # 全局错误，所有工具都受影响
                return "[Error] " + "; ".join(error_messages)
            else:
                # 错误提到了其他索引，当前工具没有错误（已成功匹配）
                return "Success"
        
        # 如果都是字典但没有当前索引的错误，说明当前工具没有错误
        return "Success"
    
    elif isinstance(error, str):
        return "[Error] " + error if error != "Success" else "Success"
    
    return "[Error] " + str(error)


def extract_function_name(model_result_raw: str) -> str | None:
    """从model_result_raw中提取函数名称"""
    match = re.match(r'^\[(\w+(?:\.\w+)*)\(', model_result_raw)
    if match:
        return match.group(1)
    return None


def extract_context(data: dict, task_id: str = None) -> str:
    """从数据中提取上下文（question部分）"""
    try:
        prompt = data.get('prompt', {})
        question_data = prompt.get('question', [])
        
        if question_data and isinstance(question_data, list):
            # 提取所有user角色的内容
            context_parts = []
            for item_list in question_data:
                if isinstance(item_list, list):
                    for item in item_list:
                        if isinstance(item, dict) and item.get('role') == 'user':
                            content = item.get('content', '')
                            if content:
                                context_parts.append(f"user: {content}")
            
            return "\n".join(context_parts) + "\n" if context_parts else ""
        
        return ""
    except Exception:
        return ""


def extract_tool_input(model_result_raw: str) -> dict:
    """从model_result_raw中提取工具输入参数"""
    try:
        # 移除开头的[和结尾的]
        if model_result_raw.startswith('[') and model_result_raw.endswith(']'):
            func_call = model_result_raw[1:-1]
            
            # 提取参数部分
            param_str = func_call[func_call.find('(')+1:func_call.rfind(')')]
            
            # 简单解析参数
            params = {}
            
            # 如果没有匹配到，尝试其他方式
            if not re.search(r'\w+\s*=', param_str):
                # 可能是位置参数，跳过
                return params
                
            # 分割参数字符串
            param_parts = []
            current_part = ""
            paren_count = 0
            bracket_count = 0
            in_string = False
            string_char = ""
            
            for i, char in enumerate(param_str):
                if char in ['"', "'"] and (i == 0 or param_str[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = ""
                elif not in_string:
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                    elif char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                    elif char == ',' and paren_count == 0 and bracket_count == 0:
                        param_parts.append(current_part.strip())
                        current_part = ""
                        continue
                current_part += char
            
            if current_part.strip():
                param_parts.append(current_part.strip())
            
            # 处理每个参数
            for part in param_parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 处理值类型
                    if value.startswith('"') and value.endswith('"') and len(value) > 1:
                        params[key] = value[1:-1]  # 去掉引号
                    elif value.startswith("'") and value.endswith("'") and len(value) > 1:
                        params[key] = value[1:-1]  # 去掉引号
                    elif value.isdigit():
                        params[key] = int(value)
                    elif value.replace('.', '').replace('-', '').isdigit():
                        try:
                            params[key] = float(value)
                        except:
                            params[key] = value
                    elif value.lower() in ['true', 'false']:
                        params[key] = value.lower() == 'true'
                    elif value.startswith('[') and value.endswith(']'):
                        # 简单处理数组
                        try:
                            # 尝试解析为数组
                            array_content = value[1:-1]
                            if ',' in array_content:
                                items = [item.strip() for item in array_content.split(',')]
                                processed_items = []
                                for item in items:
                                    if item.startswith('"') and item.endswith('"'):
                                        processed_items.append(item[1:-1])
                                    elif item.startswith("'") and item.endswith("'"):
                                        processed_items.append(item[1:-1])
                                    elif item.isdigit():
                                        processed_items.append(int(item))
                                    else:
                                        processed_items.append(item)
                                params[key] = processed_items
                            else:
                                params[key] = [array_content]
                        except:
                            params[key] = value
                    elif value.startswith('{') and value.endswith('}'):
                        # 字典类型，保持原样
                        params[key] = value
                    else:
                        params[key] = value
                    
            return params
        return {}
    except Exception as e:
        print(f"解析工具输入时出错: {e}")
        return {}


def extract_output(data: dict) -> str:
    """提取工具输出（错误信息）"""
    # 检查是否有效
    valid = data.get('valid', False)
    error = data.get('error', '')
    
    if valid and error == "Success":
        # 成功的情况，返回空字符串
        return ""
    elif isinstance(error, list):
        # 错误是列表的情况
        return "[Error] " + "; ".join(error)
    elif error and error != "Success":
        # 错误是字符串的情况
        return "[Error] " + str(error)
    else:
        # 默认情况，返回空字符串
        return "Success"


def extract_tool_info_from_structured_result(model_result_raw, data: dict, config_path: Path, task_id: str | None):
    """解析FC模型结构化的model_result_raw（list[dict] 或 dict）"""
    raw_calls = model_result_raw
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]

    if not isinstance(raw_calls, list):
        return None

    structured_calls: list[tuple[str, dict]] = []
    for entry in raw_calls:
        if not isinstance(entry, dict):
            continue
        for tool_name, raw_input in entry.items():
            if not tool_name:
                continue
            tool_input = parse_structured_tool_input(raw_input)
            structured_calls.append((tool_name, tool_input))

    if not structured_calls:
        return None

    context = extract_context(data, task_id)
    is_parallel = len(structured_calls) > 1
    error_field = data.get('error', '')
    tool_call_results = []

    for idx, (tool_name, tool_input) in enumerate(structured_calls):
        if is_parallel:
            output = get_error_for_tool_index(error_field, idx)
        else:
            output = extract_output(data)
        
        tool_call_result = build_tool_call_result(
            tool_name=tool_name,
            context=context,
            tool_input=tool_input,
            output=output,
            config_path=config_path,
            task_id=task_id
        )
        
        # 对于单轮非并行任务，根据 valid 字段判定 success
        if not is_parallel:
            valid = data.get('valid', False)
            tool_call_result["success"] = valid
        
        tool_call_results.append(tool_call_result)

    if len(tool_call_results) == 1:
        return tool_call_results[0]
    return tool_call_results


def extract_single_tool_info(data: dict, model_result_raw: str, tool_index: int = 0, is_parallel: bool = False, config_path: Path = None, task_id: str = None) -> dict | None:
    """从单条数据中提取单个工具调用信息"""
    try:
        # 提取工具名称
        tool_name = extract_function_name(model_result_raw)
        
        # 如果提取不出工具名称，返回None
        if not tool_name:
            return None
            
        # 提取上下文（只保留question部分）
        context = extract_context(data, task_id)
        
        # 提取工具输入
        tool_input = extract_tool_input(model_result_raw)
        
        # 提取工具输出（错误信息）- 为并行任务提供索引
        error = data.get('error', '')

        if is_parallel and isinstance(error, list) and len(error)>1:
            output = get_error_for_tool_index(error, tool_index)
        else:
            output = extract_output(data)
        
        # success字段：只要能解码出工具名称就可以为true
        success = True if 'error' not in str(output).lower() else False
        
        # 构建工具调用结果
        tool_call_result = build_tool_call_result(
            tool_name=tool_name,
            context=context,
            tool_input=tool_input,
            output=output,
            config_path=config_path,
            task_id=task_id
        )
        
        tool_call_result["success"] = success
        
        return tool_call_result
        
    except Exception as e:
        print(f"提取单个工具信息时出错: {e}", data)
        return None


def extract_tool_info(data: dict, config_path: Path, task_id: str = None) -> list[dict] | dict | None:
    """从单条数据中提取工具调用信息（支持并行任务）"""
    try:

        model_result_raw = data.get('model_result_raw', '')
        # 处理字符串格式
        if not model_result_raw:
            return None
        
        # 如果是字符串，检查是否为空或 '[]'
        if isinstance(model_result_raw, str):
            if not model_result_raw.strip() or model_result_raw.strip() == '[]':
                return None
        else:
            return None
        # 拆分并行工具调用
        tool_call_strings = split_parallel_tool_calls(model_result_raw)
        
        if not tool_call_strings:
            return None
        
        is_parallel = len(tool_call_strings) > 1

        # 如果只有一个工具调用，保持原有逻辑
        if not is_parallel:
            return extract_single_tool_info(data, tool_call_strings[0], is_parallel=False, config_path=config_path, task_id=task_id)
        
        # 多个工具调用，为每个工具单独提取
        tool_call_results = []
        for index, tool_call_str in enumerate[str](tool_call_strings):
            tool_info = extract_single_tool_info(data, tool_call_str, tool_index=index, is_parallel=True, config_path=config_path, task_id=task_id)
            if tool_info:
                tool_call_results.append(tool_info)

        return tool_call_results if tool_call_results else None
        
    except Exception as e:
        print(f"提取工具信息时出错: {e}", data)
        import traceback
        traceback.print_exc()
        return None


def parse_multiple_tool_calls(assistant_content: str) -> list:
    """
    解析 assistant 内容中的多个工具调用
    支持字符串格式和列表格式
    
    Args:
        assistant_content: 助手的内容，可能是单个或多个工具调用
    
    Returns:
        工具调用列表，每个元素是一个工具调用（字符串或字典）
    """
    tool_calls = []
    
    try:
        if isinstance(assistant_content, str):
            # 字符串格式：可能是 "[func1(...), func2(...)]" 或单个函数调用
            # 使用 split_parallel_tool_calls 来拆分
            tool_calls = split_parallel_tool_calls(assistant_content)
        
        elif isinstance(assistant_content, list):
            # 列表格式：可能是 [{"func1": {...}}, {"func2": {...}}]
            # 或者 [{"func1": "{\"param\": \"value\"}"}, {"func2": "{\"param\": \"value\"}"}]
            for item in assistant_content:
                if isinstance(item, dict):
                    # 每个字典是一个工具调用
                    tool_calls.append(item)
                else:
                    # 其他格式，直接添加
                    tool_calls.append(item)
        
        else:
            # 其他格式，作为单个工具调用
            tool_calls = [assistant_content]
        
        return tool_calls if tool_calls else []
        
    except Exception as e:
        print(f"解析多个工具调用时出错: {e}")
        # 如果解析失败，尝试作为单个工具调用
        return [assistant_content] if assistant_content else []


def extract_tool_name_from_content(assistant_content) -> str | None:
    """
    从助手内容中提取工具名称
    
    Args:
        assistant_content: 助手的内容，可能是工具调用（字符串、列表或字典）
    
    Returns:
        工具名称，如果提取失败返回None
    """
    try:
        if isinstance(assistant_content, str):
            # 字符串格式：可能是 "[function_name(...)]" 或直接是函数调用
            tool_name = extract_function_name(f"[{assistant_content}]" if not assistant_content.startswith('[') else assistant_content)
            return tool_name if tool_name else None
        elif isinstance(assistant_content, dict):
            # 字典格式：{"function_name": "{\"param\": \"value\"}"} 或 {"function_name": {...}}
            tool_name = list(assistant_content.keys())[0] if assistant_content else None
            return tool_name if tool_name else None
        elif isinstance(assistant_content, list):
            # 列表格式：可能是 [{"function_name": "{\"param\": \"value\"}"}, ...]
            if len(assistant_content) > 0:
                first_item = assistant_content[0]
                if isinstance(first_item, dict):
                    # 获取第一个工具调用的名称
                    tool_name = list(first_item.keys())[0] if first_item else None
                    return tool_name if tool_name else None
        return None
    except:
        return None


def format_tool_call_for_react(tool_call_content):
    """将工具调用内容格式化为ReAct格式的字符串"""
    if isinstance(tool_call_content, str):
        return tool_call_content
    elif isinstance(tool_call_content, dict):
        # 字典格式：{"function_name": "{\"param\": \"value\"}"} 或 {"function_name": {...}}
        tool_name = list(tool_call_content.keys())[0] if tool_call_content else ""
        param_str = tool_call_content.get(tool_name, "")
        if isinstance(param_str, str):
            try:
                params = json.loads(param_str)
                return f"{tool_name}({json.dumps(params, ensure_ascii=False)})"
            except:
                return f"{tool_name}({param_str})"
        else:
            return f"{tool_name}({json.dumps(param_str, ensure_ascii=False)})"
    elif isinstance(tool_call_content, list):
        # 列表格式，转换为字符串
        if len(tool_call_content) > 0:
            first_item = tool_call_content[0]
            if isinstance(first_item, dict):
                tool_name = list(first_item.keys())[0] if first_item else ""
                param_str = first_item.get(tool_name, "")
                if isinstance(param_str, str):
                    try:
                        params = json.loads(param_str)
                        return f"{tool_name}({json.dumps(params, ensure_ascii=False)})"
                    except:
                        return f"{tool_name}({param_str})"
                else:
                    return f"{tool_name}({json.dumps(param_str, ensure_ascii=False)})"
    return str(tool_call_content)


def extract_tool_from_multi_turn_step(assistant_content, tool_output, conversation_history, tool_name, config_path, task_id=None):
    """
    从多轮对话的单个步骤中提取工具调用信息
    
    Args:
        assistant_content: 助手的内容，可能是工具调用（字符串或列表）
        tool_output: 工具执行结果（JSON字符串）
        conversation_history: 该工具调用之前的对话历史（格式：["user: ...", "assistant: ...", "tool_result: ...", ...]）
        tool_name: 工具名称
        config_path: config文件路径
        task_id: 任务ID
    
    Returns:
        工具调用结果字典，如果提取失败返回None
    """
    try:
        # 如果没有传入工具名称，从内容中提取
        if not tool_name:
            tool_name = extract_tool_name_from_content(assistant_content)
        
        if not tool_name:
            return None
        
        # 解析助手内容，提取工具输入参数
        tool_call_str = None
        tool_input = {}
        
        if isinstance(assistant_content, str):
            # 字符串格式：可能是 "[function_name(...)]" 或直接是函数调用
            tool_call_str = assistant_content
            tool_input = extract_tool_input(f"[{assistant_content}]" if not assistant_content.startswith('[') else assistant_content)
        
        elif isinstance(assistant_content, dict):
            # 字典格式：{"function_name": "{\"param\": \"value\"}"} 或 {"function_name": {...}}
            param_str = assistant_content.get(tool_name, {})
            # 参数可能是JSON字符串或字典
            if isinstance(param_str, str):
                try:
                    tool_input = json.loads(param_str)
                except:
                    tool_input = {}
            elif isinstance(param_str, dict):
                tool_input = param_str
            else:
                tool_input = {}
            
            # 构造工具调用字符串用于格式化
            tool_call_str = f"{tool_name}({json.dumps(tool_input, ensure_ascii=False)})"
        
        elif isinstance(assistant_content, list):
            # 列表格式：可能是 [{"function_name": "{\"param\": \"value\"}"}, ...]
            # 或者 [{"function_name": {...}}, ...]
            if len(assistant_content) > 0:
                first_item = assistant_content[0]
                if isinstance(first_item, dict):
                    # 获取第一个工具调用的参数
                    param_str = first_item.get(tool_name, {})
                    # 参数可能是JSON字符串或字典
                    if isinstance(param_str, str):
                        try:
                            tool_input = json.loads(param_str)
                        except:
                            tool_input = {}
                    elif isinstance(param_str, dict):
                        tool_input = param_str
                    else:
                        tool_input = {}
                    
                    # 构造工具调用字符串用于格式化
                    tool_call_str = f"{tool_name}({json.dumps(tool_input, ensure_ascii=False)})"
        
        # 解析工具输出
        output = ""
        if tool_output:
            try:
                if isinstance(tool_output, str):
                    output_parsed = json.loads(tool_output)
                else:
                    output_parsed = tool_output
                output = json.dumps(output_parsed, ensure_ascii=False)
            except:
                output = str(tool_output)
        
        # 构建上下文（ReAct格式）
        # conversation_history 已经包含了该轮之前的对话历史和用户查询
        context = "\n".join(conversation_history) if conversation_history else ""
        
        # 构造工具调用结果
        tool_call_result = {
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool_name": tool_name,
            "context": context,
            "input": tool_input,
            "output": output,
            "config": str(config_path) if config_path else "",
            "token_cost": 10,
            "success": True,
            "time_cost": 0.3,
        }

        if task_id:
            tool_call_result["source_task_id"] = task_id
        
        return tool_call_result
        
    except Exception as e:
        print(f"从步骤中提取工具信息时出错: {e}")
        return None


def extract_multi_turn_tool_info(data: dict, config_path: Path, task_id: str = None) -> list[dict]:
    """
    从多轮对话数据中提取所有工具调用信息
    遍历 inference_log，找到所有使用工具的位置，提取工具信息
    返回工具调用结果列表
    
    上下文格式：只包含本次工具调用之前的内容，格式为：
    - user: <用户消息>
    - assistant: <工具调用>
    - tool_result: <工具执行结果>
    - assistant: <返回给用户的对话>
    """
    tool_call_results = []
    
    try:
        inference_log = data.get('inference_log', [])
        if not inference_log:
            return []
        
        # 累积所有轮次的对话历史（用于构建上下文）
        # 格式：["user: ...", "assistant: ...", "tool_result: ...", "assistant: ..."]
        conversation_history = []
        
        # 遍历 inference_log 中的每个元素
        i = 0
        while i < len(inference_log):
            turn_data = inference_log[i]
            
            # 如果是列表，可能是 state_info（状态信息），跳过
            if isinstance(turn_data, list):
                i += 1
                continue
            
            # 如果是字典，包含 begin_of_turn_query 和 step_X
            if isinstance(turn_data, dict):
                # 获取该轮的用户查询
                begin_of_turn_query = turn_data.get('begin_of_turn_query', [])
                
                # 提取用户消息
                user_message = ""
                if begin_of_turn_query:
                    user_items = [item.get('content', '') for item in begin_of_turn_query 
                                 if isinstance(item, dict) and item.get('role') == 'user']
                    if user_items:
                        user_message = user_items[0]
                        # 将用户查询添加到对话历史中
                        conversation_history.append(f"user: {user_message}")
                
                # 遍历该轮的所有步骤（step_0, step_1, ...）
                step_keys = [key for key in turn_data.keys() if key.startswith('step_')]
                step_keys.sort(key=lambda x: int(x.split('_')[1]))  # 按步骤编号排序
                
                # 遍历该轮的所有步骤
                for step_key in step_keys:
                    step_data = turn_data.get(step_key, [])
                    if not isinstance(step_data, list):
                        continue
                    
                    # 在步骤中查找工具调用和助手回复
                    assistant_content = None
                    tool_outputs = []  # 可能有多个工具输出
                    assistant_text = None  # 助手的文本回复（非工具调用）
                    
                    # 遍历 step_data，收集所有信息
                    for item in step_data:
                        if isinstance(item, dict):
                            role = item.get('role', '')
                            
                            if role == 'assistant':
                                # 助手响应
                                content = item.get('content', '')
                                assistant_content = content
                            
                            elif role == 'tool':
                                # 工具执行结果（可能有多个）
                                tool_outputs.append(item.get('content', ''))
                    
                    # 如果找到了工具输出，说明这一步使用了工具
                    if tool_outputs and assistant_content is not None:
                        # 解析 assistant 内容中的多个工具调用
                        tool_calls = parse_multiple_tool_calls(assistant_content)
                        
                        # 按顺序匹配每个工具调用和对应的工具结果
                        for idx, tool_call in enumerate(tool_calls):
                            # 获取对应的工具输出（按索引顺序）
                            tool_output = tool_outputs[idx] if idx < len(tool_outputs) else None
                            
                            if tool_output is None:
                                continue
                            
                            # 提取工具名称
                            tool_name = extract_tool_name_from_content(tool_call)
                            
                            # 使用当前对话历史作为上下文（不包含本次工具调用）
                            tool_info = extract_tool_from_multi_turn_step(
                                tool_call, 
                                tool_output,
                                conversation_history.copy(),  # 使用副本，避免修改原列表
                                tool_name,
                                config_path,
                                task_id
                            )
                            
                            if tool_info:
                                tool_call_results.append(tool_info)
                            
                            # 将工具调用和结果添加到对话历史中（供后续工具使用）
                            if tool_call:
                                # 格式化工具调用
                                tool_call_str = format_tool_call_for_react(tool_call)
                                conversation_history.append(f"assistant: {tool_call_str}")
                            
                            if tool_output:
                                # 解析工具输出（可能是JSON字符串）
                                try:
                                    if isinstance(tool_output, str):
                                        tool_output_parsed = json.loads(tool_output)
                                    else:
                                        tool_output_parsed = tool_output
                                    conversation_history.append(f"tool_result: {json.dumps(tool_output_parsed, ensure_ascii=False)}")
                                except:
                                    conversation_history.append(f"tool_result: {tool_output}")
                    
                    # 如果助手有文本回复（非工具调用），也添加到历史中
                    elif assistant_content and isinstance(assistant_content, str) and not tool_outputs:
                        # 检查是否是工具调用格式（包含括号）
                        if not (isinstance(assistant_content, str) and '(' in assistant_content and ')' in assistant_content):
                            conversation_history.append(f"assistant: {assistant_content}")
            
            i += 1
        
        return tool_call_results
        
    except Exception as e:
        print(f"提取多轮对话工具信息时出错: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_tool_calls_from_results(model_name: str, task_ids: set[str], config_path: Path, result_dir: Path = None, n_sample: int = None) -> list[dict]:
    """
    从评估结果中提取工具调用信息
    
    Args:
        model_name: 模型名称
        task_ids: 任务ID集合
        config_path: config文件路径
        result_dir: 结果目录，如果为None则使用RESULT_PATH或SCORE_PATH
        n_sample: 采样次数，如果指定则从所有采样文件中提取（从采样1到采样n_sample）
    """
    tool_call_results = []

    categories = set()
    for tid in task_ids:
        category = extract_category_from_task_id(tid)
        categories.add(category)

    for category in categories:
        if category in MULTI_TURN_CATEGORY:
            subdir = "multi_turn"
            is_multi_turn = True
        elif category in LIVE_CATEGORY:
            subdir = "live"
            is_multi_turn = False
        elif category in NON_LIVE_CATEGORY:
            subdir = "non_live"
            is_multi_turn = False
        else:
            print(f"⚠️ 未知类别 {category}，跳过工具调用提取")
            continue

        model_subdir = model_name.replace("/", "_")
        if is_multi_turn:
            model_dir = RESULT_PATH / model_subdir / subdir        # 只匹配采样1的文件（后缀为1的文件），即文件名以 "_result_1.json" 结尾
            result_files = [f for f in model_dir.glob("BFCL_v4_*_result_1.json") 
                            if f.name.endswith("_result_1.json")]
        else:
            model_dir = SCORE_PATH / model_subdir / subdir
            # 只匹配采样1的文件（后缀为1的文件），即文件名以 "_result_1.json" 结尾
            result_files = [f for f in model_dir.glob("BFCL_v4_*_score_1.json") 
                            if f.name.endswith("_score_1.json")]
        
        if not model_dir.exists():
            print(f"⚠️ 结果目录不存在: {model_dir}")
            continue


        if not result_files:
            print(f"{category}: ⚠️ 未找到结果文件在: {model_dir}")
            continue

        category_task_ids = {tid for tid in task_ids if extract_category_from_task_id(tid) == category}
        for result_file in result_files:
            try:
                results = load_file(result_file)
                total_in_file = len(results)
                matched_count = 0
                for data in results:
                    data_id = data.get("id")
                    # 只提取 task_ids 中指定的任务，过滤掉结果文件中的其他任务
                    if data_id in category_task_ids:
                        matched_count += 1
                        if is_multi_turn:
                            tool_infos = extract_multi_turn_tool_info(data, config_path, data_id)
                            if tool_infos:
                                for tool_info in tool_infos:
                                    tool_info["source_task_id"] = data_id
                                tool_call_results.extend(tool_infos)
                        else:
                            tool_infos = extract_tool_info(data, config_path, data_id)
                            if tool_infos:
                                if isinstance(tool_infos, list):
                                    for tool_info in tool_infos:
                                        tool_info["source_task_id"] = data_id
                                    tool_call_results.extend(tool_infos)
                                else:
                                    tool_infos["source_task_id"] = data_id
                                    tool_call_results.append(tool_infos)
                
                # 输出过滤统计信息
                if total_in_file > 0:
                    filtered_count = total_in_file - matched_count
                    if filtered_count > 0:
                        print(f"   文件 {result_file.name}: 总 {total_in_file} 条，匹配 {matched_count} 条，过滤 {filtered_count} 条（不在 task_ids 中）")
                            
            except Exception as e:
                print(f"⚠️ 读取结果文件 {result_file} 时出错: {e}")

    return tool_call_results


def split_dataset(input_file: Path) -> tuple[set[str], set[str]]:
    """
    划分数据集为测试集和训练集。
    
    Args:
        input_file: 输入文件路径
    
    Returns:
        (test_task_ids, train_task_ids) 元组
    """
    test_task_ids = set()
    train_task_ids = set()
    
    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return test_task_ids, train_task_ids
    
    print(f"📖 读取文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line or not line.strip():
                continue
            try:
                entry = json.loads(line)
                task_id = entry.get("task_id")
                if not task_id:
                    continue
                
                # 测试集：所有task_id
                test_task_ids.add(task_id)
                
                # 训练集：只包括related_task_ids，不包括task_id本身
                all_related_ids = collect_all_task_ids(entry)
                # 移除task_id，只保留related_task_ids
                train_task_ids.update(all_related_ids - {task_id})
                
            except json.JSONDecodeError:
                continue
    
    print(f"📊 测试集: {len(test_task_ids)} 个任务")
    print(f"📊 训练集: {len(train_task_ids)} 个任务（仅包括related_task_ids，不包括task_id）")
    
    return test_task_ids, train_task_ids


def merge_train_results_to_test_sample1(model: str, test_result_dir: Path, train_result_dir: Path = None):
    """
    将训练集的结果合并到测试集结果的采样1文件中。
    
    Args:
        model: 模型名称
        test_result_dir: 测试集结果目录
        train_result_dir: 训练集结果目录，如果为None则使用RESULT_PATH
    """
    if train_result_dir is None:
        train_result_dir = RESULT_PATH
    
    model_subdir = model.replace("/", "_")
    test_model_dir = test_result_dir / model_subdir
    train_model_dir = train_result_dir / model_subdir
    
    if not train_model_dir.exists():
        print(f"⚠️ 训练集结果目录不存在: {train_model_dir}")
        return
    
    if not test_model_dir.exists():
        print(f"⚠️ 测试集结果目录不存在: {test_model_dir}")
        return
    
    print(f"📝 合并训练集结果到测试集采样1文件...")
    print(f"   训练集目录: {train_model_dir}")
    print(f"   测试集目录: {test_model_dir}")
    
    # 遍历所有子目录（non_live, live, multi_turn等）
    for subdir in ["non_live", "live", "multi_turn"]:
        train_subdir = train_model_dir / subdir
        test_subdir = test_model_dir / subdir
        
        if not train_subdir.exists():
            continue
        
        # 查找训练集的所有结果文件（应该是采样1的文件，即后缀为1的文件）
        # 只匹配后缀为1的文件（采样1）
        train_result_files = [
            f for f in train_subdir.glob("BFCL_v4_*_result_1.json")
            if f.name.endswith("_result_1.json")
        ]
        
        for train_file in train_result_files:
            # 获取测试集对应的采样1文件（也是后缀为1的文件）
            test_file = test_subdir / train_file.name
            
            if not test_file.exists():
                print(f"⚠️ 测试集文件不存在，跳过: {test_file}")
                continue
            
            try:
                # 读取训练集结果
                train_results = load_file(train_file)
                
                # 读取测试集采样1结果
                test_results = load_file(test_file)
                
                # 获取测试集中已有的ID集合
                existing_ids = {entry.get("id") for entry in test_results}
                
                # 只添加训练集中不在测试集中的结果
                new_results = [entry for entry in train_results if entry.get("id") not in existing_ids]
                
                if new_results:
                    # 合并结果
                    merged_results = test_results + new_results
                    
                    # 按ID排序
                    merged_results.sort(key=lambda x: x.get("id", ""))
                    
                    # 写回文件
                    with open(test_file, 'w', encoding='utf-8') as f:
                        for entry in merged_results:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    
                    print(f"✅ 合并 {len(new_results)} 条训练集结果到 {test_file.name}")
                else:
                    print(f"ℹ️  {test_file.name} 中已包含所有训练集结果，无需合并")
                    
            except Exception as e:
                print(f"⚠️ 合并文件时出错 {train_file} -> {test_file}: {e}")


def extract_and_map_tools_to_hash(model: str, task_ids: set[str], output_config_path: Path, result_dir: Path = None, n_sample: int = None, skip_config_build: bool = False) -> tuple[list[dict], dict[str, str]]:
    """
    从评估结果中提取工具调用信息，计算hash，并用hash代替工具名称。
    
    Args:
        model: 模型名称
        task_ids: 任务ID集合（可以是训练集、测试集或两者的并集）
        output_config_path: 输出config.json文件路径
        result_dir: 结果目录，如果为None则使用RESULT_PATH
        n_sample: 采样次数，如果指定则从所有采样文件中提取（从采样1到采样n_sample）
        skip_config_build: 如果为True，跳过config构建过程，只从已有config中加载并映射
    
    Returns:
        (tool_call_results_with_hash, tool_name_to_hash) 元组
        - tool_call_results_with_hash: 工具调用结果列表（工具名称已替换为hash）
        - tool_name_to_hash: 工具名称到hash的映射字典 {tool_name: hash}
    """
    # print(f"\n🔍 步骤3: 从评估结果中提取工具调用信息并计算hash")
    # print("-" * 80)
    
    # 初始化变量
    tool_name_to_hash = {}
    tool_hash_to_config = {}
    
    # 如果 skip_config_build 为 True，且 config 已存在，则只加载已有 config 并映射
    if skip_config_build and output_config_path.exists():
        print(f"📖 检测到 config.json 已存在，跳过 config 构建过程，只提取工具调用并映射...")
        # 加载已有的 config，构建 tool_name 到 hash 的映射
        try:
            with open(output_config_path, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
                tool_hash_to_config.update(existing_config)
                # 从 hash 中提取 tool_name
                for tool_hash in existing_config.keys():
                    if len(tool_hash) > 33:  # 至少 tool_name(1) + '_' + hash(32)
                        last_underscore_idx = tool_hash.rfind('_')
                        if last_underscore_idx > 0 and last_underscore_idx < len(tool_hash) - 1:
                            potential_hash = tool_hash[last_underscore_idx + 1:]
                            if len(potential_hash) == 32 and all(c in '0123456789abcdef' for c in potential_hash.lower()):
                                tool_name = tool_hash[:last_underscore_idx]
                                if tool_name not in tool_name_to_hash:
                                    tool_name_to_hash[tool_name] = tool_hash
                print(f"📖 从已有 config.json 加载了 {len(tool_hash_to_config)} 个工具，映射了 {len(tool_name_to_hash)} 个工具名称")
        except Exception as e:
            print(f"⚠️ 加载已有 config.json 失败: {e}")
            skip_config_build = False  # 如果加载失败，回退到正常流程
    
    # 先构建一个包含所有工具名称的临时config（用于提取工具调用信息）
    # 我们需要先找出所有任务中使用的工具
    temp_config = {}
    
    # 按category分组task_ids，避免重复加载同一数据集文件
    category_to_task_ids = defaultdict(set)
    for task_id in task_ids:
        category = extract_category_from_task_id(task_id)
        category_to_task_ids[category].add(task_id)
    
    # 按category批量处理，每个category只加载一次数据集文件
    for category, category_task_ids in category_to_task_ids.items():
        data_file = PROJECT_ROOT / "bfcl_eval" / "data" / f"BFCL_v4_{category}.json"
        if not data_file.exists():
            continue
        
        try:
            # 只加载一次数据集文件
            dataset = load_file(data_file)
            
            # 建立task_id到item的索引，O(1)查找
            task_id_to_item = {item.get("id"): item for item in dataset if item.get("id")}
            
            # 对于多轮任务类别，需要先填充 function 字段
            multi_turn_task_ids = {tid for tid in category_task_ids if is_multi_turn(tid)}
            if multi_turn_task_ids:
                # 获取多轮任务的items
                multi_turn_items = [task_id_to_item[tid] for tid in multi_turn_task_ids if tid in task_id_to_item]
                # 使用 populate_test_cases_with_predefined_functions 填充多轮任务的函数文档
                multi_turn_items_filled = populate_test_cases_with_predefined_functions(multi_turn_items)
                # 更新 task_id_to_item 映射
                for filled_item in multi_turn_items_filled:
                    task_id_to_item[filled_item.get("id")] = filled_item
            
            # 批量处理该category下的所有task_ids
            for task_id in category_task_ids:
                item = task_id_to_item.get(task_id)
                if not item:
                    continue
                
                # 统一从function字段提取工具信息（不再区分多轮任务和其他任务）
                function_data = item.get("function", [])
                if isinstance(function_data, list):
                    for func in function_data:
                        if isinstance(func, dict) and "name" in func:
                            tool_name = func["name"]
                            schema = {k: v for k, v in func.items() if k != "name"}
                            if schema:
                                # 如果工具名称已存在，保留第一个（避免覆盖）
                                if tool_name not in temp_config:
                                    temp_config[tool_name] = schema
        except Exception as e:
            print(f"⚠️ 读取数据集文件 {data_file} 时出错: {e}")
    
    # 创建临时config文件用于提取工具调用信息
    temp_config_path = output_config_path.parent / "temp_extract_config.json"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        json.dump(temp_config, f, indent=4, ensure_ascii=False)
    
    # 从评估结果中提取所有工具调用信息（此时工具名称还是原始名称）
    # 只从采样1的文件中提取（如果指定了n_sample）
    tool_call_results = extract_tool_calls_from_results(model, task_ids, temp_config_path, result_dir=result_dir, n_sample=n_sample)
        # 保存tool_call_results_with_hash到JSONL文件
    tool_call_results_path = output_config_path.parent / "tool_call_results.jsonl"
    with open(tool_call_results_path, 'w', encoding='utf-8') as f:
        for result in tool_call_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"✅ 已保存工具调用结果到: {tool_call_results_path}")
    # return tool_call_results, {}
    # 清理临时config文件
    if temp_config_path.exists():
        temp_config_path.unlink()
    
    # 如果output_config_path已存在，先加载已存在的config，避免重复计算
    if not skip_config_build and output_config_path.exists():
        try:
            with open(output_config_path, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
                tool_hash_to_config.update(existing_config)
                print(f"📖 加载已存在的config.json，包含 {len(existing_config)} 个工具")
        except Exception as e:
            print(f"⚠️ 加载已存在的config.json失败: {e}")
    
    # 为每个工具调用结果计算hash并替换工具名称
    tool_call_results_with_hash = []
    
    # 多轮任务缓存：tool_name -> (tool_config, tool_hash)
    # 因为多轮任务中不会出现重名但不同config的工具，所以可以缓存
    # 如果未找到config，缓存值为 (None, None)
    multi_turn_cache: dict[str, tuple[dict | None, str | None]] = {}
    
    for result in tqdm(tool_call_results, desc="计算hash并替换工具名称"):
        original_tool_name = result.get("tool_name")
        if not original_tool_name:
            continue
        
        if skip_config_build:
            # 如果跳过 config 构建，直接使用已有的 tool_name_to_hash 映射
            if original_tool_name in tool_name_to_hash:
                tool_hash = tool_name_to_hash[original_tool_name]
                # 替换工具名称为hash，并更新config字段为永久路径
                result_copy = result.copy()
                result_copy["tool_name"] = tool_hash
                result_copy["config"] = str(output_config_path)
                tool_call_results_with_hash.append(result_copy)
            else:
                print(f"⚠️ 工具 {original_tool_name} 在已有 config.json 中未找到对应 hash，跳过该工具调用")
        else:
            # 正常流程：获取该工具的config并计算hash
            source_task_id = result.get("source_task_id")
            
            # 检查是否是多轮任务，如果是，先检查缓存
            is_multi_turn_task = source_task_id and is_multi_turn(source_task_id)
            
            tool_config = None
            tool_hash = None
            
            if is_multi_turn_task:
                # 多轮任务：先检查缓存
                if original_tool_name in multi_turn_cache:
                    cached_config, cached_hash = multi_turn_cache[original_tool_name]
                    # 如果缓存中是None，说明之前已经查找过但未找到，直接跳过
                    if cached_config is None and cached_hash is None:
                        continue  # 跳过，不打印警告（避免重复打印）
                    tool_config, tool_hash = cached_config, cached_hash
                else:
                    # 缓存未命中，需要计算
                    if source_task_id:
                        tool_config = get_tool_config_from_dataset(source_task_id, original_tool_name)
                    
                    if not tool_config:
                        # 如果从source_task_id获取失败，尝试从temp_config中查找（fallback）
                        if original_tool_name in temp_config:
                            tool_config = temp_config[original_tool_name]
                        else:
                            # 如果temp_config中也没有，从所有task_ids中查找（最后fallback）
                            for task_id in task_ids:
                                tool_config = get_tool_config_from_dataset(task_id, original_tool_name)
                                if tool_config:
                                    break
                    
                    if tool_config:
                        # 计算hash并存入缓存
                        tool_hash = calculate_tool_hash(original_tool_name, tool_config)
                        multi_turn_cache[original_tool_name] = (tool_config, tool_hash)
                    else:
                        # 未找到config，也存入缓存（值为None），避免重复查找
                        multi_turn_cache[original_tool_name] = (None, None)
                        print(f"⚠️ 未找到工具 {original_tool_name} 的config（source_task_id: {source_task_id}），跳过该工具调用")
            else:
                # 非多轮任务：保持原有逻辑（可能存在同名但不同config的情况）
                if source_task_id:
                    # 优先从source_task_id对应的任务中获取config（确保使用正确的config）
                    tool_config = get_tool_config_from_dataset(source_task_id, original_tool_name)
                
                if not tool_config:
                    # 如果从source_task_id获取失败，尝试从temp_config中查找（fallback）
                    if original_tool_name in temp_config:
                        tool_config = temp_config[original_tool_name]
                    else:
                        # 如果temp_config中也没有，从所有task_ids中查找（最后fallback）
                        for task_id in task_ids:
                            tool_config = get_tool_config_from_dataset(task_id, original_tool_name)
                            if tool_config:
                                break
                
                if tool_config:
                    # 计算hash（使用统一的calculate_tool_hash函数，与__main__.py中的方法一致）
                    tool_hash = calculate_tool_hash(original_tool_name, tool_config)
            
            if tool_config and tool_hash:
                # 如果hash已经存在于config中，直接使用（避免重复存储相同的config）
                if tool_hash in tool_hash_to_config:
                    # 替换工具名称为hash，并更新config字段为永久路径
                    result_copy = result.copy()
                    result_copy["tool_name"] = tool_hash
                    result_copy["config"] = str(output_config_path)
                    tool_call_results_with_hash.append(result_copy)
                else:
                    # hash不存在，需要添加新的config
                    tool_hash_to_config[tool_hash] = tool_config
                    
                    # 替换工具名称为hash，并更新config字段为永久路径
                    result_copy = result.copy()
                    result_copy["tool_name"] = tool_hash
                    result_copy["config"] = str(output_config_path)
                    tool_call_results_with_hash.append(result_copy)
            elif not is_multi_turn_task:
                # 非多轮任务且未找到config时才打印警告（多轮任务的警告已在上面处理）
                print(f"⚠️ 未找到工具 {original_tool_name} 的config（source_task_id: {source_task_id}），跳过该工具调用")
    
    if skip_config_build:
        print(f"📊 共 {len(tool_call_results_with_hash)} 条工具调用记录（已使用已有 config.json 映射为hash）")
    else:
        print(f"📊 从评估结果中提取到 {len(tool_hash_to_config)} 个不同的工具config（使用hash区分同名但不同config的工具）")
        print(f"📊 共 {len(tool_call_results_with_hash)} 条工具调用记录（已替换为hash）")
        
        # 保存config.json（使用hash作为key）
        with open(output_config_path, 'w', encoding='utf-8') as f:
            json.dump(tool_hash_to_config, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 已提取 {len(tool_hash_to_config)} 个工具的config并计算hash")
        print(f"✅ 已保存config.json到: {output_config_path}")
    
    # 保存tool_call_results_with_hash到JSONL文件
    tool_call_results_path = output_config_path.parent / "tool_call_results.jsonl"
    with open(tool_call_results_path, 'w', encoding='utf-8') as f:
        for result in tool_call_results_with_hash:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"✅ 已保存工具调用结果到: {tool_call_results_path}")
    
    return tool_call_results_with_hash, tool_name_to_hash


def send_post_requests_with_source_task(workspace_id: str, tool_call_results: list[dict], config_path: Path, source_task: str):
    """
    发送POST请求，包含source_task参数。
    
    Args:
        workspace_id: 工作空间ID
        tool_call_results: 工具调用结果列表
        config_path: config.json文件路径
        source_task: 来源任务ID
    """
    print(f"\n📤 发送POST请求 (workspace_id: {workspace_id}, source_task: {source_task}):")
    print("=" * 80)
    
    # 读取config.json获取当前任务使用的工具名称（现在是hash）
    config_tools = set()
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            config_tools = set(config_data.keys())
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"⚠️ 无法读取config文件: {config_path}")
        config_tools = set()
    
    print(f"🔧 config.json中定义的工具: {sorted(config_tools)}")
    
    # 过滤工具调用结果，只保留config中定义的工具
    filtered_results = []
    for result in tool_call_results:
        tool_name = result.get("tool_name")  # 这里应该是hash
        if tool_name and tool_name in config_tools:
            filtered_result = {k: v for k, v in result.items() if k not in ["source_task_id"]}
            filtered_results.append(filtered_result)
    
    print(f"📊 过滤后保留 {len(filtered_results)}/{len(tool_call_results)} 条工具调用记录")
    
    # 分批发送POST请求，每批大小为5
    batch_size = 5
    total_tools = len(filtered_results)
    total_batches = (total_tools + batch_size - 1) // batch_size
    
    print(f"共 {total_tools} 条工具调用记录，将分 {total_batches} 批发送，每批 {batch_size} 条")
    
    for batch_idx in tqdm(range(total_batches), desc="发送POST批次"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_tools)
        batch_tools = filtered_results[start_idx:end_idx]
        
        batch_post_data = {
            "workspace_id": workspace_id,
            "tool_call_results": batch_tools,
            "source_task": source_task
        }
        
        try:
            response = requests.post("http://localhost:8002/add_tool_call_result", json=batch_post_data)
            if response.status_code != 200:
                print(f"❌ 批次 {batch_idx + 1}/{total_batches} 请求失败，状态码: {response.status_code}, 响应内容: {response.text}")
                continue
            elif not response.json().get('success', False):
                print(f"❌ 批次 {batch_idx + 1}/{total_batches} 工具存储失败，工具数量: {len(batch_tools)}")
        except Exception as e:
            print(f"❌ 批次 {batch_idx + 1}/{total_batches} 发送时出错: {e}")
    
    print("✅ POST请求发送完成")


def get_mem0_memory_instance():
    """
    获取全局的 mem0 Memory 实例（单例模式，避免 Qdrant 并发访问问题）。
    
    Returns:
        Memory 实例，如果初始化失败则返回 None
    """
    global _global_mem0_memory_instance, _global_mem0_memory_lock, _global_mem0_init_failed
    
    # 如果全局实例已存在，直接返回
    if _global_mem0_memory_instance is not None:
        return _global_mem0_memory_instance
    
    # 如果之前初始化失败，直接返回 None，避免重复尝试
    if _global_mem0_init_failed:
        return None
    
    # 使用锁确保只有一个线程创建实例
    with _global_mem0_memory_lock:
        # 双重检查，防止多个线程同时通过锁
        if _global_mem0_memory_instance is not None:
            return _global_mem0_memory_instance
        
        # 再次检查初始化失败标志（在锁内）
        if _global_mem0_init_failed:
            return None
        
        try:
            from mem0 import Memory
            import shutil
            # 使用与 evaluator.py 相同的配置
            # 为 mem0 使用独立的数据目录，避免与旧的 Qdrant 数据冲突
            mem0_qdrant_path = PROJECT_ROOT / "mem0_qdrant_data"
            
            # 检查目录是否已存在且有数据
            # 如果目录已存在且包含数据文件，则保留数据，不删除目录
            # 只有在目录为空或不存在时才创建新目录
            should_clear_data = False
            if mem0_qdrant_path.exists() and mem0_qdrant_path.is_dir():
                # 检查目录中是否有数据文件（Qdrant会创建collection目录）
                has_data = False
                try:
                    # Qdrant会在目录下创建collection子目录，如果有collection目录说明有数据
                    collection_dir = mem0_qdrant_path / "collection"
                    if collection_dir.exists() and collection_dir.is_dir():
                        # 检查collection目录下是否有文件
                        if any(collection_dir.iterdir()):
                            has_data = True
                except Exception:
                    pass
                
                if not has_data:
                    # 目录存在但没有数据，可以清理（可能是上次失败的运行留下的空目录）
                    try:
                        shutil.rmtree(mem0_qdrant_path)
                        print(f"🗑️  已清理空的 mem0 Qdrant 数据目录: {mem0_qdrant_path}")
                        should_clear_data = True
                    except Exception as e:
                        print(f"⚠️  清理旧目录时出错: {e}")
                else:
                    print(f"📖 检测到 mem0 Qdrant 数据目录已存在且包含数据，保留现有数据: {mem0_qdrant_path}")
            
            if not mem0_qdrant_path.exists() or should_clear_data:
                mem0_qdrant_path.mkdir(parents=True, exist_ok=True)
            
            config = {
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "api_key": os.getenv("OPENAI_API_KEY", "your api key"),
                        "model": "text-embedding-v3",
                        "embedding_dims": 512,
                        "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    }
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "api_key": os.getenv("OPENAI_API_KEY", "your api key"),
                        "model": "qwen-plus",
                        "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    }
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "embedding_model_dims": 512,
                        "path": str(mem0_qdrant_path),
                        "on_disk": True
                    }
                }
            }
            _global_mem0_memory_instance = Memory.from_config(config)
            print("✅ mem0 Memory 实例初始化成功")
            return _global_mem0_memory_instance
        except Exception as e:
            print(f"❌ 初始化 mem0 Memory 失败: {e}")
            import traceback
            traceback.print_exc()
            # 标记初始化失败，避免后续任务重复尝试
            _global_mem0_init_failed = True
            return None


# 模块级别的全局变量，用于 mem0 Memory 单例
_global_mem0_memory_instance = None
_global_mem0_memory_lock = threading.Lock()
_global_mem0_init_failed = False  # 标记初始化是否失败，避免重复尝试
_global_mem0_access_lock = threading.Lock()  # 用于序列化 mem0 实例的访问，避免线程数过多

# 缓存机制：缓存工具的 memory，避免每次都调用 get_all()
# 格式：{tool_name: [memory_items]}
_global_mem0_memory_cache: dict[str, list] = {}
_global_mem0_cache_lock = threading.Lock()  # 保护缓存访问的锁


def close_mem0_memory_instance():
    """
    关闭并清理 mem0 Memory 实例，释放 Qdrant 连接。
    这确保在子进程启动前，父进程已经释放了所有资源。
    """
    global _global_mem0_memory_instance
    
    if _global_mem0_memory_instance is None:
        return
    
    try:
        # 尝试关闭 Memory 实例内部的 Qdrant 客户端
        # mem0 Memory 实例可能包含 vector_store 属性，其中包含 Qdrant 客户端
        if hasattr(_global_mem0_memory_instance, 'vector_store'):
            vector_store = _global_mem0_memory_instance.vector_store
            if hasattr(vector_store, 'client'):
                client = vector_store.client
                # Qdrant 客户端可能有 close() 方法
                if hasattr(client, 'close'):
                    client.close()
                    print("✅ 已关闭 Qdrant 客户端连接")
                # 或者可能是 QdrantClient 实例，尝试关闭
                elif hasattr(client, '__class__') and 'QdrantClient' in str(client.__class__):
                    try:
                        # 尝试调用 close 方法（如果存在）
                        if hasattr(client, 'close'):
                            client.close()
                        print("✅ 已关闭 Qdrant 客户端连接")
                    except Exception as e:
                        print(f"⚠️ 关闭 Qdrant 客户端时出错: {e}")
        
        # 重置全局实例
        _global_mem0_memory_instance = None
        print("✅ 已清理 mem0 Memory 实例")
        
    except Exception as e:
        print(f"⚠️ 清理 mem0 Memory 实例时出错: {e}")
        # 即使出错也重置全局实例
        _global_mem0_memory_instance = None


def get_mem0_memories_for_tools(tool_names: list, current_task_id: str, memory_instance, run_id: str = "bfcl") -> dict:
    """
    从 mem0 获取指定工具的记忆，排除当前 task 的记录。
    使用缓存机制，避免每次都调用 get_all()。
    限制每个工具的 memory 长度为 120000/工具数，超过部分截断。
    
    Args:
        tool_names: 工具名称列表
        current_task_id: 当前任务的 ID（用于排除）
        memory_instance: mem0 Memory 实例
        run_id: run_id 参数，默认为 "bfcl"
    
    Returns:
        Dict[str, str]: 工具名称到拼接后的 memory 文本的映射
    """
    global _global_mem0_memory_cache, _global_mem0_cache_lock
    
    memory_map: dict = {name: "" for name in tool_names}
    
    if memory_instance is None:
        return memory_map
    
    # 计算每个工具的最大 memory 长度
    num_tools = len(tool_names) if tool_names else 1
    max_length_per_tool = 120000 // num_tools
    
    for tool_name in tool_names:
        try:
            # 先从缓存中获取
            cached_memories = None
            with _global_mem0_cache_lock:
                if tool_name in _global_mem0_memory_cache:
                    cached_memories = _global_mem0_memory_cache[tool_name]
            
            # 如果缓存中没有，从 mem0 获取并缓存
            if cached_memories is None:
                # 使用锁序列化 mem0 访问，避免 mem0 内部创建过多线程
                # mem0 库内部可能使用线程，并发访问会导致 "can't start new thread" 错误
                with _global_mem0_access_lock:
                    # 获取该工具的所有记忆
                    all_memories_result = memory_instance.get_all(agent_id=tool_name, run_id=run_id)
                
                if all_memories_result and "results" in all_memories_result:
                    cached_memories = all_memories_result["results"]
                    # 缓存结果
                    with _global_mem0_cache_lock:
                        _global_mem0_memory_cache[tool_name] = cached_memories
                else:
                    # 如果没有结果，缓存空列表
                    with _global_mem0_cache_lock:
                        _global_mem0_memory_cache[tool_name] = []
                    continue
            
            # 从缓存的 memory 中过滤掉当前 task 的记录，并拼接 memory 文本
            memory_texts = []
            for mem_item in cached_memories:
                user_id = mem_item.get("user_id", "")
                # 排除当前 task 的记录
                if str(user_id) != str(current_task_id):
                    mem_text = mem_item.get("memory", "")
                    if mem_text:
                        memory_texts.append(mem_text)
            
            # 将多个 memory 拼接起来，用空格分隔
            if memory_texts:
                # prefix = "Besides, this is optional guidance on how to better use the tool. You may refer to it selectively "
                combined_memory = " ".join(memory_texts)
                
                # 限制长度：如果超过最大长度，截断
                if len(combined_memory) > max_length_per_tool:
                    combined_memory = combined_memory[:max_length_per_tool]
                    # 尝试在单词边界处截断，避免截断单词
                    last_space = combined_memory.rfind(' ')
                    if last_space > max_length_per_tool * 0.9:  # 如果最后一个空格在90%位置之后，使用它
                        combined_memory = combined_memory[:last_space]
                
                memory_map[tool_name] = combined_memory
        except Exception as e:
            print(f"警告: 获取工具 {tool_name} 的 mem0 记忆失败: {e}")
            continue
    
    return memory_map


def extract_user_query_from_context_for_mem0(context: str) -> str:
    """
    从 context 中提取用户查询（用于 mem0）。
    对于 BFCL 数据集，context 是字符串格式，包含 "user: ..." 等格式。
    
    Args:
        context: 上下文字符串
        
    Returns:
        str: 提取出的用户查询内容
    """
    if not context:
        return ""
    
    # 提取所有 "user: ..." 行的内容
    user_messages = []
    for line in context.split('\n'):
        line = line.strip()
        if line.startswith('user:'):
            # 提取 "user:" 之后的内容
            user_content = line[5:].strip()  # 跳过 "user:" 前缀
            if user_content:
                user_messages.append(user_content)
    
    if user_messages:
        # 返回最后一条用户消息，或者拼接所有用户消息
        return user_messages[-1] if len(user_messages) == 1 else " ".join(user_messages)
    
    # 如果没有找到用户消息，返回原始 context
    return context


def sample_tool_call_results(tool_call_results: list[dict], max_per_tool: int = 5) -> list[dict]:
    """
    对工具调用结果进行采样，按工具名称分组，每个工具最多保留指定数量的记录。
    
    Args:
        tool_call_results: 工具调用结果列表
        max_per_tool: 每个工具最多保留的记录数，默认为5
    
    Returns:
        采样后的工具调用结果列表
    """
    if not tool_call_results:
        return []
    
    # 按工具名称分组
    tool_to_results = defaultdict(list)
    for result in tool_call_results:
        tool_name = result.get("tool_name")
        if tool_name:
            tool_to_results[tool_name].append(result)
    
    # 对每个工具随机选择最多max_per_tool条记录
    sampled_results = []
    for tool_name, results in tool_to_results.items():
        if len(results) > max_per_tool:
            sampled = random.sample(results, max_per_tool)
            print(f"🔧 工具 {tool_name}: 从 {len(results)} 条记录中随机选择 {max_per_tool} 条")
        else:
            sampled = results
            print(f"🔧 工具 {tool_name}: 共 {len(results)} 条记录，全部使用")
        sampled_results.extend(sampled)
    
    print(f"📊 总共 {len(tool_call_results)} 条记录，采样后 {len(sampled_results)} 条记录")
    return sampled_results


def send_tool_calls_to_mem0(tool_call_results: list[dict], model_name: str):
    """
    将工具调用结果发送到 mem0。
    注意：调用此函数前应该先对 tool_call_results 进行采样。
    
    Args:
        tool_call_results: 工具调用结果列表（应该已经采样过）
        model_name: 模型名称（用于 run_id）
    """
    print(f"\n📤 发送工具调用结果到 mem0 (model: {model_name}):")
    print("=" * 80)
    
    if not tool_call_results:
        print("⚠️ 没有工具调用结果需要发送")
        return
    
    # 获取 mem0 Memory 实例
    memory_instance = get_mem0_memory_instance()
    if memory_instance is None:
        print("❌ mem0 Memory 实例未初始化，无法发送数据到 mem0")
        return
    
    # 使用固定的 run_id（与 evaluator.py 中的 'toole-ds' 类似，这里使用模型名称）
    run_id = f"bfcl-{model_name}"
    
    success_count = 0
    fail_count = 0
    
    for result in tqdm(tool_call_results, desc="发送到 mem0"):
        try:
            tool_name = result.get("tool_name")
            if not tool_name:
                continue
            
            # 提取上下文
            context = result.get("context", "")
            user_query = extract_user_query_from_context_for_mem0(context)
            if not user_query:
                user_query = str(context)
            
            # 提取工具输入
            tool_input = result.get("input", {})
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except:
                    tool_input = {"value": tool_input}
            
            # 提取工具输出
            tool_output = result.get("output", "")
            success = result.get("success", True)
            
            # 构建消息（参考 evaluator.py 的格式：user, assistant with tool_calls, tool）
            messages = [
                {
                    "role": "user",
                    "content": f"{user_query}."
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_input, ensure_ascii=False) if isinstance(tool_input, dict) else str(tool_input)
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "1",
                    "content": f"success: {success} output: {tool_output}"
                }
            ]
            
            # 获取 source_task_id（用于 user_id）
            source_task_id = result.get("source_task_id", "")
            user_id = source_task_id if source_task_id else "unknown"
            
            # 发送到 mem0（使用 procedural_memory 类型）
            memory_instance.add(
                messages, 
                agent_id=tool_name, 
                user_id=user_id, 
                run_id=run_id, 
                memory_type="procedural_memory"
            )
            success_count += 1
        except Exception as e:
            print(f"⚠️ 发送工具调用结果到 mem0 失败: {e}")
            fail_count += 1
    
    print(f"✅ 成功发送 {success_count} 条，失败 {fail_count} 条")
    
    # 关闭 Memory 实例，释放 Qdrant 连接，避免子进程访问冲突
    print(f"\n🔒 关闭 mem0 Memory 实例，释放 Qdrant 连接...")
    close_mem0_memory_instance()


def process_dataset(
    input_file: Path,
    model: str,
    method: str = None,
    additional_args: list[str] = None,
):
    """
    处理数据集的主要流程。
    
    Args:
        input_file: 输入文件路径
        model: 模型名称
        method: 方法选择（"mem"/"m0"/"draft"），如果为空或不在选项中则跳过相应步骤
        additional_args: 额外的命令行参数列表
    """
    print(f"\n{'='*80}")
    print(f"🚀 开始处理数据集")
    print(f"{'='*80}")
    
    # 步骤1: 划分测试集和训练集
    print(f"\n📊 步骤1: 划分测试集和训练集")
    print("-" * 80)
    test_task_ids, train_task_ids = split_dataset(input_file)
    
    if not test_task_ids or not train_task_ids:
        print("❌ 数据集为空，退出")
        return
    
    # 步骤2: 在测试集和训练集上运行生成和评估
    print(f"\n🔧 步骤2: 在测试集和训练集上运行生成和评估")
    print("-" * 80)
    
    
    # 2.1: 先运行测试集（3次采样）
    print("\n📝 2.1: 运行测试集生成（3次采样）...")
    test_category_to_ids = group_task_ids_by_category(test_task_ids)
    generate_test_case_ids_file(test_category_to_ids, TEST_IDS_TO_GENERATE_PATH)
    print(f"✅ 已生成测试集的test_case_ids_to_generate.json")
    
    # 步骤2使用默认的RESULT_PATH（不使用method）
    test_result_dir = RESULT_PATH
    print(f"📂 测试集结果将保存到: {test_result_dir}")
    
    # 运行测试集生成命令（3次采样，不使用method）
    test_additional_args = (additional_args or []) + ["--n-sample", "1"]
    print("🚀 运行测试集生成命令（n_sample=3，不使用method）...")
    if not run_generation(model, test_additional_args, result_dir=test_result_dir, method=None):
        print("❌ 测试集生成失败，退出")
        return
    
    # # 2.2: 运行训练集（1次采样，直接附加到测试集采样1文件）
    # print("\n📝 2.2: 运行训练集生成（1次采样，将附加到测试集采样1文件）...")
    # train_category_to_ids = group_task_ids_by_category(train_task_ids)
    # generate_test_case_ids_file(train_category_to_ids, TEST_IDS_TO_GENERATE_PATH)
    # print(f"✅ 已生成训练集的test_case_ids_to_generate.json")
    
    # # 运行训练集生成命令（1次采样，生成到test_result_dir，会自动附加到采样1文件，不使用method）
    # print("🚀 运行训练集生成命令（n_sample=1，将附加到测试集采样1文件，不使用method）...")
    # train_additional_args = (additional_args or []) + ["--n-sample", "1"]
    # if not run_generation(model, train_additional_args, result_dir=test_result_dir, method=None):
    #     print("❌ 训练集生成失败，退出")
    #     return
    
    # # 2.3: 评估测试集采样1文件（包含训练集+测试集）
    # print("\n📝 2.3: 评估测试集采样1文件（包含训练集+测试集）...")
    # if not run_evaluation_command(model, result_dir=test_result_dir):
    #     print("❌ 评估失败，退出")
    #     return
    
    # 步骤3: 提取工具调用信息，计算hash，保存config.json
    print(f"\n🔍 步骤3: 从采样1中提取工具调用信息并计算hash（仅从采样1文件）")
    print("-" * 80)
    config_path = PROJECT_ROOT / "bfcl_eval" / "data" / "config.json"
    
    # 从测试集采样1文件中提取（包含训练集+测试集），提取所有任务
    all_task_ids = train_task_ids | test_task_ids
    
    # 如果 config.json 已存在，跳过 config 构建过程，但仍需要从当前模型的评估结果中提取工具调用
    skip_config_build = config_path.exists()
    if skip_config_build:
        print(f"📖 检测到 config.json 已存在，将跳过 config 构建过程，但仍会从当前模型的评估结果中提取工具调用...")
    
    tool_call_results_with_hash, tool_name_to_hash = extract_and_map_tools_to_hash(
        model, all_task_ids, config_path, result_dir=test_result_dir, n_sample=1, skip_config_build=skip_config_build
    )
    # 步骤4: 根据method参数进行处理
    if method and method in ["mem", "m0", "draft"]:
        print(f"\n🔧 步骤4: 使用方法 {method} 进行处理")
        print("-" * 80)
        
        if method == "mem":
            # 对工具调用结果进行采样（每个工具最多保留5条）
            print(f"📝 对工具调用结果进行采样（每个工具最多保留5条）...")
            sampled_tool_call_results = sample_tool_call_results(tool_call_results_with_hash, max_per_tool=5)
            # 使用采样后的结果来构建记忆库
            workspace_id = "bfcl-"+model
            # send_post_requests_with_source_task(workspace_id, sampled_tool_call_results, config_path, source_task="train_set")
        elif method == "m0":
            # 对工具调用结果进行采样（每个工具最多保留5条）
            print(f"📝 对工具调用结果进行采样（每个工具最多保留5条）...")
            sampled_tool_call_results = sample_tool_call_results(tool_call_results_with_hash, max_per_tool=5)
            # 使用 mem0 方法存储工具调用结果
            print(f"📝 使用 mem0 方法存储工具调用结果...")
            # 注意：tool_call_results_with_hash 中的工具名称已经是 hash，但 mem0 需要原始工具名称
            # 由于我们已经有了 tool_name_to_hash 映射，我们需要反向映射
            # 但为了简化，我们直接使用 hash 作为工具名称（mem0 会按 agent_id 存储）
            # send_tool_calls_to_mem0(sampled_tool_call_results[528:], model)
        elif method == "draft":
            # 使用 DRAFT 方法改进工具描述
            from bfcl_eval.scripts.draft_improver import improve_config_with_draft
            
            tool_call_results_path = config_path.parent / "tool_call_results.jsonl"
            draft_config_path = config_path.parent / "config_draft.json"
            
            print(f"📝 使用 DRAFT 方法改进工具描述...")
            print(f"   输入配置: {config_path}")
            print(f"   工具调用结果: {tool_call_results_path}")
            print(f"   输出配置: {draft_config_path}")
            
            try:
                # improved_config = improve_config_with_draft(
                #     config_path=config_path,
                #     tool_call_results_path=tool_call_results_path,
                #     output_config_path=draft_config_path,
                #     episodes=5,
                #     model="qwen-plus"
                # )
                print(f"✅ DRAFT 改进完成，改进后的配置已保存到: {draft_config_path}")
            except Exception as e:
                print(f"❌ DRAFT 改进失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ 方法 {method} 暂未实现")
    else:
        print(f"\n⏭️  步骤4: method参数为空或不在支持列表中，跳过")
    
    # 步骤5: 对测试集进行生成（3次采样）和评估
    if method and method in ["mem", "m0", "draft"]:
        print(f"\n🔧 步骤5: 对测试集进行生成（3次采样）和评估（使用方法: {method}）")
        print("-" * 80)
        
        # 生成测试集的test_case_ids_to_generate.json
        print("📝 生成测试集的test_case_ids_to_generate.json...")
        test_category_to_ids = group_task_ids_by_category(test_task_ids)
        generate_test_case_ids_file(test_category_to_ids, TEST_IDS_TO_GENERATE_PATH)
        
        # 根据method获取result目录（通过--result-dir区分，同时传递--method参数以触发bfcl内部不同流程）
        final_result_dir = get_result_dir_for_method(method)
        print(f"📂 测试集结果将保存到: {final_result_dir}（通过--result-dir区分方法）")
        
        # 运行生成命令（3次采样，通过result_dir区分方法，同时传递method参数以触发bfcl内部不同流程）
        print(f"🚀 运行测试集生成命令（n_sample=3，result_dir={final_result_dir}，method={method}）...")
        test_additional_args = (additional_args or []) + ["--n-sample", "3"]
        if not run_generation(model, test_additional_args, result_dir=final_result_dir, method=method):
            print("❌ 测试集生成失败")
            return
        
        # 运行评估命令（只评估采样1的文件）
        print("🔍 运行测试集评估命令（仅评估采样1文件）...")
        if not run_evaluation_command(model, result_dir=final_result_dir, method=method):
            print("❌ 测试集评估失败")
            return
    else:
        print(f"\n⏭️  步骤5: method参数为空或不在支持列表中，跳过")
    
    print(f"\n{'='*80}")
    print(f"✅ 所有流程完成")
    print(f"{'='*80}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="处理数据集，划分训练集和测试集，并运行评估流程"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default="/Users/xiaocan/Toolmem/experiment/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/all_task.jsonl",
        help="输入的 step4_sampled_tasks.jsonl 文件路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="qwen3-8b",
        help="模型名称（例如: gorilla-openfunctions-v2）"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["mem", "m0", "draft"],
        help="方法选择: mem/m0/draft，如果为空则跳过相应步骤"
    )
    parser.add_argument(
        "--additional-args",
        type=str,
        nargs="*",
        default=None,
        help="额外的命令行参数（例如: --temperature 0.7 --num-gpus 2）"
    )

    args = parser.parse_args()

    process_dataset(
        input_file=args.input_file,
        model=args.model,
        method=args.method,
        additional_args=args.additional_args,
    )


if __name__ == "__main__":
    main()

