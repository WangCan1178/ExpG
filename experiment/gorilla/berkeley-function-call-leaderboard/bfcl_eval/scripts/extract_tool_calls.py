import json
import os
import re
from datetime import datetime
import requests
from tqdm import tqdm

# 加载config.json
CONFIG_PATH = "/Users/xiaocan/Toolmem/experiment/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/config.json"
_config_cache = None

def load_config():
    """加载config.json并缓存"""
    global _config_cache
    if _config_cache is None:
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                _config_cache = json.load(f)
        except Exception as e:
            print(f"加载config.json失败: {e}")
            _config_cache = {}
    return _config_cache

def check_tool_in_config(tool_name):
    """
    检查工具是否在config.json中存在
    
    Args:
        tool_name: 工具名称，可能是 "module.function_name" 或 "function_name" 格式
    
    Returns:
        tuple: (exists: bool, matched_key: str or None)
    """
    config = load_config()
    if not config:
        return False, None
    
    # 将工具名称转换为小写
    tool_name_lower = tool_name.lower() if tool_name else None
    
    # 直接检查完整名称（小写）
    if tool_name_lower in config:
        return True, tool_name_lower
    
    # # 如果工具名称包含点号（如 "module.function_name"），尝试多种匹配方式
    # if '.' in tool_name:
    #     # 尝试只使用函数名部分
    #     function_name = tool_name.split('.')[-1]
    #     if function_name in config:
    #         return True, function_name
        
    #     # 尝试所有可能的点号分割组合
    #     parts = tool_name.split('.')
    #     for i in range(len(parts)):
    #         # 从后往前组合，如 "module.function" -> "function", "module.function"
    #         candidate = '.'.join(parts[i:])
    #         if candidate in config:
    #             return True, candidate
    
    # 如果都不匹配，返回False
    return False, None

def extract_tool_calls_from_json_files(model_name, subdir):
    """
    从指定模型和子目录的JSON文件中提取工具调用信息
    支持 live/non_live 和 multi_turn 两种格式
    """
    base_path = f"/Users/xiaocan/Toolmem/experiment/gorilla/berkeley-function-call-leaderboard/score/{model_name}/{subdir}"
    
    tool_call_results = []
    
    # 判断是否为 multi_turn 格式
    is_multi_turn = subdir == "multi_turn"
    
    # 遍历目录中的所有JSON文件
    for filename in os.listdir(base_path):
        
        file_path = os.path.join(base_path, filename)
        
        try:
            with open(file_path, 'r') as f:
                # 跳过第一行（统计信息）
                first_line = f.readline()
                
                # 处理每一行数据
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            
                            if is_multi_turn:
                                # 多轮对话格式：从 inference_log 中提取所有工具调用
                                tool_infos = extract_multi_turn_tool_info(data)
                                if tool_infos:
                                    tool_call_results.extend(tool_infos)
                            else:
                                # 单轮格式：从 model_result_raw 中提取（支持并行任务）
                                tool_infos = extract_tool_info(data)
                                if tool_infos:
                                    if isinstance(tool_infos, list):
                                        tool_call_results.extend(tool_infos)
                                    else:
                                        tool_call_results.append(tool_infos)
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            continue
    
    return tool_call_results

def split_parallel_tool_calls(model_result_raw):
    """
    拆分并行任务中的多个工具调用
    返回工具调用字符串列表
    """
    if not model_result_raw or not isinstance(model_result_raw, str):
        return []
    
    if not model_result_raw.startswith('[') or not model_result_raw.endswith(']'):
        return [model_result_raw]
    
    # 移除外层方括号
    content = model_result_raw[1:-1].strip()
    if not content:
        return []
    
    # 如果只有一个工具调用（没有逗号分隔），直接返回
    if ',' not in content:
        return [f"[{content}]"]
    
    # 拆分多个工具调用
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
                # 找到一个工具调用的边界
                tool_call = current_call.strip()
                if tool_call:
                    tool_calls.append(f"[{tool_call}]")
                current_call = ""
                continue
        current_call += char
    
    # 添加最后一个工具调用
    if current_call.strip():
        tool_calls.append(f"[{current_call.strip()}]")
    
    return tool_calls if tool_calls else [model_result_raw]

def get_error_for_tool_index(error, tool_index):
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
                    import re
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
            import re
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

def extract_tool_info(data):
    """
    从单条数据中提取工具调用信息（支持并行任务）
    返回单个工具调用结果或列表
    """
    try:
        model_result_raw = data.get('model_result_raw', '')
        if not model_result_raw or model_result_raw.strip() == '[]':
            return None
        
        # 拆分并行工具调用
        tool_call_strings = split_parallel_tool_calls(model_result_raw)
        
        if not tool_call_strings:
            return None
        
        is_parallel = len(tool_call_strings) > 1
        
        # 如果只有一个工具调用，保持原有逻辑
        if not is_parallel:
            return extract_single_tool_info(data, tool_call_strings[0], is_parallel=False)
        
        # 多个工具调用，为每个工具单独提取
        tool_call_results = []
        for index, tool_call_str in enumerate(tool_call_strings):
            tool_info = extract_single_tool_info(data, tool_call_str, tool_index=index, is_parallel=True)
            if tool_info:
                tool_call_results.append(tool_info)
        
        return tool_call_results if tool_call_results else None
        
    except Exception as e:
        print(f"提取工具信息时出错: {e}", data)
        return None

def extract_single_tool_info(data, model_result_raw, tool_index=0, is_parallel=False):
    """
    从单条数据中提取单个工具调用信息
    """
    try:
        # 提取工具名称（已转换为小写）
        tool_name = extract_function_name(model_result_raw)
        
        # 如果提取不出工具名称，返回None
        if not tool_name:
            return None
            
        # 提取上下文（只保留question部分）
        context = extract_context(data)
        
        # 提取工具输入
        tool_input = extract_tool_input(model_result_raw)
        
        # 提取工具输出（错误信息）- 为并行任务提供索引
        error = data.get('error', '')
        if is_parallel:
            output = get_error_for_tool_index(error, tool_index)
        else:
            output = extract_output(data)
        
        # success字段：只要能解码出工具名称就可以为true
        success = True
        
        # 检查工具是否在config.json中存在
        tool_exists, matched_key = check_tool_in_config(tool_name)
        
        # 构造工具调用结果
        tool_call_result = {
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool_name": tool_name,
            "context": context,
            "input": tool_input,
            "output": output,
            "config": "/Users/xiaocan/Toolmem/experiment/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/config.json",
            "token_cost": 10,
            "success": success,
            "time_cost": 0.3,
            # "tool_exists_in_config": tool_exists,
            # "matched_config_key": matched_key
        }
        
        # 如果工具不存在，打印警告
        if not tool_exists:
            print(f"警告: 工具 '{tool_name}' 在config.json中不存在")
        
        return tool_call_result
        
    except Exception as e:
        print(f"提取单个工具信息时出错: {e}", data)
        return None

def extract_function_name(model_result_raw):
    """
    从model_result_raw中提取函数名称
    """
    # 匹配函数名，例如 [function_name(...)
    match = re.match(r'^\[(\w+(?:\.\w+)*)\(', model_result_raw)
    if match:
        # 统一转换为小写
        return match.group(1).lower()
    return None

def extract_context(data):
    """
    从数据中提取上下文（question部分）
    """
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

def extract_tool_input(model_result_raw):
    """
    从model_result_raw中提取工具输入参数
    """
    try:
        # 移除开头的[和结尾的]
        if model_result_raw.startswith('[') and model_result_raw.endswith(']'):
            func_call = model_result_raw[1:-1]
            
            # 提取参数部分
            param_str = func_call[func_call.find('(')+1:func_call.rfind(')')]
            
            # 简单解析参数
            params = {}
            
            # 使用更复杂的正则表达式来匹配各种参数格式
            # 匹配 key=value, key=value, ... 格式的参数
            # 处理字符串、数字、布尔值、列表等
            param_pattern = r'(\w+(?:\.\w+)*)\s*=\s*([^,]+?)(?=,\s*\w+\s*=|,\s*\)|\Z)'
            
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

def extract_output(data):
    """
    提取工具输出（错误信息）
    """
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

def extract_multi_turn_tool_info(data):
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
                                tool_name
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

def parse_multiple_tool_calls(assistant_content):
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

def extract_tool_name_from_content(assistant_content):
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
            return tool_name.lower() if tool_name else None
        elif isinstance(assistant_content, dict):
            # 字典格式：{"function_name": "{\"param\": \"value\"}"} 或 {"function_name": {...}}
            tool_name = list(assistant_content.keys())[0] if assistant_content else None
            return tool_name.lower() if tool_name else None
        elif isinstance(assistant_content, list):
            # 列表格式：可能是 [{"function_name": "{\"param\": \"value\"}"}, ...]
            if len(assistant_content) > 0:
                first_item = assistant_content[0]
                if isinstance(first_item, dict):
                    # 获取第一个工具调用的名称
                    tool_name = list(first_item.keys())[0] if first_item else None
                    return tool_name.lower() if tool_name else None
        return None
    except:
        return None

def extract_tool_from_multi_turn_step(assistant_content, tool_output, conversation_history, tool_name):
    """
    从多轮对话的单个步骤中提取工具调用信息
    
    Args:
        assistant_content: 助手的内容，可能是工具调用（字符串或列表）
        tool_output: 工具执行结果（JSON字符串）
        conversation_history: 该工具调用之前的对话历史（格式：["user: ...", "assistant: ...", "tool_result: ...", ...]）
        tool_name: 工具名称
    
    Returns:
        工具调用结果字典，如果提取失败返回None
    """
    try:
        # 如果没有传入工具名称，从内容中提取（已转换为小写）
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
        
        # 检查工具是否在config.json中存在
        tool_exists, matched_key = check_tool_in_config(tool_name)
        
        # 构造工具调用结果
        tool_call_result = {
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool_name": tool_name,
            "context": context,
            "input": tool_input,
            "output": output,
            "config": "/Users/xiaocan/Toolmem/experiment/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/config.json",
            "token_cost": 10,
            "success": True,
            "time_cost": 0.3,
            # "tool_exists_in_config": tool_exists,
            # "matched_config_key": matched_key
        }
        
        # 如果工具不存在，打印警告
        if not tool_exists:
            print(f"警告: 工具 '{tool_name}' 在config.json中不存在")
        
        return tool_call_result
        
    except Exception as e:
        print(f"从步骤中提取工具信息时出错: {e}")
        return None

def format_tool_call_for_react(tool_call_content):
    """
    将工具调用内容格式化为ReAct格式的字符串
    """
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

def construct_post_request(model_name, tool_call_results, fold):
    """
    构造POST请求
    支持单个工具调用结果或工具调用结果列表
    """
    # 如果传入的是单个字典，转换为列表
    if isinstance(tool_call_results, dict):
        tool_call_results = [tool_call_results]
    
    post_data = {
        "workspace_id": model_name+"_"+fold,
        "tool_call_results": tool_call_results
    }
    
    return post_data

def main():
    # 指定模型名称和子目录
    model_name = "qwen3-8b"
    subdir = "non_live"
    fold = "test"
    
    # 提取工具调用信息
    tool_call_results = extract_tool_calls_from_json_files(model_name, subdir)
    
    # 统计工具配置检查结果
    total_tools = len(tool_call_results)
    # tools_in_config = sum(1 for tool in tool_call_results if tool.get('tool_exists_in_config', False))
    # tools_not_in_config = total_tools - tools_in_config
    
    # 构造POST请求
    print(f"提取到 {total_tools} 条工具调用结果。")
    # print(f"其中 {tools_in_config} 个工具在config.json中存在，{tools_not_in_config} 个工具不存在。")
    
    # if tools_not_in_config > 0:
    #     # 列出所有不存在的工具
    #     missing_tools = set()
    #     for tool in tool_call_results:
    #         if not tool.get('tool_exists_in_config', False):
    #             missing_tools.add(tool.get('tool_name', 'unknown'))
    #     print(f"不存在的工具列表: {sorted(missing_tools)}")

    # 分批次发送，每批大小为10
    batch_size = 10
    total_batches = (total_tools + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="发送批次"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_tools)
        batch_tools = tool_call_results[start_idx:end_idx]
        
        # 构造批量POST请求
        post_data = construct_post_request(model_name, batch_tools, fold)
        
        # # 打印构造好的POST请求（不实际发送）
        # for tool in batch_tools:
        #     if tool.get('tool_name', 'unknown') == "touch":
        #         print()
        #         print(json.dumps(tool, ensure_ascii=False, indent=2))
        #         print()
        
        
        try:
            response = requests.post("http://localhost:8002/add_tool_call_result", json=post_data)
            if response.status_code != 200:
                print(f"批次 {batch_idx + 1}/{total_batches} 请求失败，状态码: {response.status_code}, 响应内容: {response.text}")
                continue
            elif not response.json().get('success', False):
                print(f"批次 {batch_idx + 1}/{total_batches} 工具存储失败，工具数量: {len(batch_tools)}")

        except Exception as e:
            print(f"批次 {batch_idx + 1}/{total_batches} 发送时出错: {e}")

    # 发送总结请求
    for tool in tqdm(tool_call_results):
        if tool.get('tool_name', 'unknown') == "lawsuit_search":
            response = requests.post("http://localhost:8002/summary_tool_memory", json={
                "workspace_id": model_name+"_"+fold,
                "tool_names": tool.get('tool_name', 'unknown')
            })
            if response.status_code != 200:
                print(f"summary 请求失败，状态码: {response.status_code}, 响应内容: {response.text}")
            elif not response.json().get('success', False):
                print(f"summary 工具记忆失败，工具名称: {tool.get('tool_name', 'unknown')}")

if __name__ == "__main__":
    main()
