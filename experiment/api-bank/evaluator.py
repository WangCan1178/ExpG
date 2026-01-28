import json
import re
import os
import logging
import copy
import ast
import glob
import traceback
import time
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from rouge import Rouge
from tqdm import tqdm

from tool_manager import ToolManager
from utils import ChatGPTWrapper, DavinciWrapper, GPT4Wrapper, QwenOpenAIWrapper
from api_call_extraction import parse_api_call
from tool_memory import ToolMemoryFetcher

# 模块级别的全局 mem0 Memory 实例（单例模式，避免 Qdrant 并发访问问题）
_global_mem0_memory_instance = None
_global_mem0_memory_lock = threading.Lock()

# OpenAI 客户端相关导入（用于调用本地 vLLM 服务）
try:
    from openai import OpenAI
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False
    print("警告: OpenAI 客户端未安装，--method toolllama/qwenft/llama 选项将不可用。安装命令: pip install openai")


def calculate_rouge_l_score(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    rouge_l_score = scores[0]['rouge-l']['f']
    return rouge_l_score


def _lower_obj(obj):
    """递归将字符串转小写，用于大小写不敏感比较。"""
    if isinstance(obj, str):
        return obj.lower()
    if isinstance(obj, list):
        return [_lower_obj(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _lower_obj(v) for k, v in obj.items()}
    return obj


def _normalize_container(obj):
    """将字符串形式的 list/dict 尝试解析为对象，并统一 list/tuple -> list，递归处理。"""
    # 尝试把类似 "['a', 'b']" 的字符串转成列表
    if isinstance(obj, str):
        stripped = obj.strip()
        if (stripped.startswith('[') and stripped.endswith(']')) or (stripped.startswith('{') and stripped.endswith('}')):
            try:
                obj = ast.literal_eval(stripped)
            except Exception:
                return obj
    if isinstance(obj, tuple):
        obj = list(obj)
    if isinstance(obj, list):
        return [_normalize_container(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _normalize_container(v) for k, v in obj.items()}
    return obj


def write_record(record, path):
    """追加写入一条评估记录到 JSONL 文件。"""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def get_output_path(output_path_base, sample_idx, method: Optional[str] = None):
    """根据采样索引生成输出文件路径。
    
    参数:
        output_path_base: 基础文件名（不含扩展名）
        sample_idx: 采样索引（0 到 n_sample-1）
        method: 方法名称（可选），如果提供会在文件名中添加后缀
    
    返回:
        输出文件路径
    """
    output_dir = os.path.dirname(output_path_base) if os.path.dirname(output_path_base) else '.'
    output_base_name = os.path.basename(output_path_base)
    
    # 如果提供了 method，在文件名中添加后缀
    if method:
        output_path = os.path.join(output_dir, f"{output_base_name}_{method}_{sample_idx}.jsonl")
    else:
        output_path = os.path.join(output_dir, f"{output_base_name}_{sample_idx}.jsonl")
    return output_path


def parse_model_output_raw(model_output_raw: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    从 model_output_raw 中解析工具名称和输入参数。
    
    格式示例: [Calculator(formula='(5+6)*3')]
    返回: (tool_name, input_dict)
    """
    if not model_output_raw:
        return None, {}
    
    # 匹配格式: [ToolName(param1='value1', param2='value2')]
    pattern = r"\[(\w+)\((.*)\)\]"
    match = re.search(pattern, model_output_raw, re.MULTILINE | re.DOTALL)
    
    if not match:
        return None, {}
    
    api_name = match.group(1)
    params_str = match.group(2).strip()
    
    if not params_str:
        return api_name, {}
    
    # 解析参数字符串，支持多种格式：
    # - key='value'
    # - key="value"
    # - key=value (无引号)
    param_dict = {}
    
    # 更健壮的参数解析：处理值中包含引号、括号等复杂情况
    # 使用状态机方式解析，处理嵌套引号
    i = 0
    while i < len(params_str):
        # 跳过空白和逗号
        while i < len(params_str) and params_str[i] in ' ,':
            i += 1
        if i >= len(params_str):
            break
        
        # 找到 key
        key_start = i
        while i < len(params_str) and params_str[i] not in ' =':
            i += 1
        key = params_str[key_start:i].strip()
        
        if not key:
            break
        
        # 跳过等号和空白
        while i < len(params_str) and params_str[i] in ' =':
            i += 1
        if i >= len(params_str):
            break
        
        # 解析 value
        if params_str[i] == "'":
            # 单引号字符串
            i += 1
            value_start = i
            while i < len(params_str) and params_str[i] != "'":
                if params_str[i] == '\\' and i + 1 < len(params_str):
                    i += 2  # 跳过转义字符
                else:
                    i += 1
            value = params_str[value_start:i]
            i += 1  # 跳过结束引号
        elif params_str[i] == '"':
            # 双引号字符串
            i += 1
            value_start = i
            while i < len(params_str) and params_str[i] != '"':
                if params_str[i] == '\\' and i + 1 < len(params_str):
                    i += 2  # 跳过转义字符
                else:
                    i += 1
            value = params_str[value_start:i]
            i += 1  # 跳过结束引号
        else:
            # 无引号值，取到下一个逗号或结束
            value_start = i
            while i < len(params_str) and params_str[i] != ',':
                i += 1
            value = params_str[value_start:i].strip()

        # 尝试恢复原始容器类型（例如 list / dict），保持参数格式
        # 仅在看起来像列表/字典字面量时才做安全解析
        value_stripped = value.strip()
        if (value_stripped.startswith('[') and value_stripped.endswith(']')) or (
            value_stripped.startswith('{') and value_stripped.endswith('}')
        ):
            try:
                value_evaluated = ast.literal_eval(value_stripped)
                value = value_evaluated
            except Exception:
                # 解析失败则退回原始字符串
                pass

        param_dict[key] = value
    
    return api_name, param_dict


def extract_tool_output(tool_result: Any) -> str:
    """
    从 tool_result 中提取输出。
    如果 tool_result 是字符串，直接返回；否则解析字典中的 output 字段。
    优先返回 output，如果没有 output 则返回 exception。
    """
    if isinstance(tool_result, str):
        return tool_result
    elif isinstance(tool_result, dict):
        output = tool_result.get("output", "")
        exception = tool_result.get("exception", None)
        # 优先返回 output，如果没有 output 则返回 exception
        if output:
            return str(output)
        elif exception is not None:
            return str(exception)
        else:
            return ""
    else:
        return str(tool_result) if tool_result else ""


def extract_final_response(content: str) -> str:
    """
    提取思考标签之后的内容，支持多种标签格式。
    用于处理微调模型的输出，移除思考过程，只保留最终回答。
    
    Args:
        content: 模型返回的完整内容
        
    Returns:
        str: 提取出的最终回答
    """
    if not content:
        return ""
    
    # 支持的标签列表（按常见程度排序）
    tags = ['</think>', '</thinking>']
    
    for tag in tags:
        if tag in content:
            # 找到标签后的内容
            parts = content.split(tag, 1)
            if len(parts) > 1:
                return parts[1].strip()
    
    # 如果没有找到标签，返回原始内容
    return content.strip()


def _convert_toolllama_output_to_apibank_style(text: str) -> Optional[str]:
    """
    将 ToolLLaMA 风格的输出转换为 API-Bank 需要的格式。
    
    支持多种格式：
    1. Action: ApiName / Action Input: { ...json... }
    2. [Finish(...)] 格式（识别为 finish，返回 None）
    3. [ApiName(...)] 格式（已经是 API-Bank 格式，直接返回）
    4. 从完整对话历史中提取最后一部分的 Action/Action Input
    
    如果 Action 为 finish（不区分大小写），返回 None，表示不调用 API。
    解析失败时，返回原始文本。
    """
    if not text:
        return text
    
    import re
    
    # 首先检查是否已经是 API-Bank 格式 [ApiName(...)]
    # 使用括号匹配来正确提取包含嵌套括号的情况
    def find_api_calls(text: str):
        """查找所有 [ApiName(...)] 格式的调用，使用括号匹配"""
        results = []
        i = 0
        while i < len(text):
            # 查找 [
            if text[i] == '[':
                start = i
                i += 1
                # 提取 API 名称
                api_name_start = i
                while i < len(text) and (text[i].isalnum() or text[i] == '_'):
                    i += 1
                if i == api_name_start:
                    i += 1
                    continue
                api_name = text[api_name_start:i]
                
                # 查找 (
                if i >= len(text) or text[i] != '(':
                    i += 1
                    continue
                i += 1
                
                # 括号匹配提取参数部分
                paren_level = 1
                param_start = i
                while i < len(text) and paren_level > 0:
                    if text[i] == '(':
                        paren_level += 1
                    elif text[i] == ')':
                        paren_level -= 1
                    i += 1
                
                if paren_level == 0:
                    # 找到完整的调用
                    param_end = i - 1
                    # 查找 ]
                    if i < len(text) and text[i] == ']':
                        full_match = text[start:i+1]
                        results.append((start, full_match, api_name))
                        i += 1
                        continue
            
            i += 1
        return results
    
    api_calls = find_api_calls(text)
    if api_calls:
        # 优先选择第一个非 Finish 的调用；如果全是 Finish，则返回 None
        for start, full_match, api_name in api_calls:
            if api_name.lower() != "finish":
                return full_match
        # 如果只有 Finish 调用，则视为不调用 API
        return None
    
    # 也检查是否有 "Finish" 作为 Action（不区分大小写）
    finish_patterns = [
        r"Action:\s*[Ff]inish",
        r"\[[Ff]inish\s*\(",
        r"\[[Ff]inish\s*\]"
    ]
    for pattern in finish_patterns:
        if re.search(pattern, text):
            return None
    
    # 尝试查找 Action: ... Action Input: ... 格式
    # 找到最后一次出现的 Action（防止前面有多轮）
    action_idx = text.rfind("Action:")
    if action_idx == -1:
        # 如果找不到 Action，尝试从完整对话历史中提取最后一部分
        # 可能模型返回的是完整的对话历史，需要找到最后一部分
        # 尝试查找最后一个角色标记（AI: / User: / System:）之后的内容
        last_ai_idx = text.rfind("\nAI:")
        last_user_idx = text.rfind("\nUser:")
        last_system_idx = text.rfind("\nSystem:")
        
        # 也检查开头就是角色标记的情况
        if text.startswith("AI:"):
            last_ai_idx = max(last_ai_idx, 0)
        if text.startswith("User:"):
            last_user_idx = max(last_user_idx, 0)
        if text.startswith("System:"):
            last_system_idx = max(last_system_idx, 0)
        
        # 找到最后一个角色标记的位置
        last_role_idx = max(last_ai_idx, last_user_idx, last_system_idx)
        if last_role_idx != -1:
            # 从最后一个角色标记之后的内容中查找 Action
            remaining_text = text[last_role_idx:]
            action_idx = remaining_text.find("Action:")
            if action_idx != -1:
                action_idx = last_role_idx + action_idx
            else:
                # 如果最后一部分没有 Action，尝试查找是否有 [ApiName(...)] 格式
                # 这可能是模型直接返回的 API 调用格式
                remaining_api_calls = find_api_calls(remaining_text)
                if remaining_api_calls:
                    # 同样优先选择第一个非 Finish 的调用
                    for start, full_match, api_name in remaining_api_calls:
                        if api_name.lower() != "finish":
                            return full_match
                    return None
                return text
        else:
            return text
    
    action_part = text[action_idx:]
    
    # 提取 Action 名称
    # 形如: Action: OpenBankAccount
    m = re.search(r"Action:\s*([A-Za-z0-9_]+)", action_part)
    if not m:
        return text
    
    action_name = m.group(1).strip()
    if action_name.lower() == "finish":
        # 明确表示不调用 API
        return None
    
    # 提取 Action Input JSON
    # 形如:
    # Action Input: {
    #   "account": "john",
    #   ...
    # }
    input_idx = action_part.find("Action Input")
    if input_idx == -1:
        # 没有输入就当做无效，返回原文
        return text
    
    input_part = action_part[input_idx:]
    brace_start = input_part.find("{")
    if brace_start == -1:
        return text
    
    # 从第一个 { 开始做括号匹配，抓完整 JSON
    brace_level = 0
    json_start = brace_start
    json_end = None
    for i, ch in enumerate(input_part[brace_start:], start=brace_start):
        if ch == "{":
            brace_level += 1
        elif ch == "}":
            brace_level -= 1
            if brace_level == 0:
                json_end = i
                break
    
    if json_end is None:
        return text
    
    json_str = input_part[json_start:json_end + 1]
    
    try:
        params = json.loads(json_str)
        if not isinstance(params, dict):
            return text
    except Exception:
        # JSON 解析失败则返回原文，避免影响其它逻辑
        return text
    
    # 构造 API-Bank 风格的调用字符串
    # [ApiName(key1='value1', key2='value2', ...)]
    kv_parts = []
    for k, v in params.items():
        v_str = str(v)
        # 简单转义单引号
        v_str = v_str.replace("'", "\\'")
        kv_parts.append(f"{k}='{v_str}'")
    inner = ", ".join(kv_parts)
    return f"[{action_name}({inner})]"


def _build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    将 OpenAI-style messages 扁平化为一个 prompt 字符串，形如：
    System:
    ...
    
    User:
    ...
    
    AI:
    ...
    """
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "system":
            prefix = "System"
        elif role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "AI"
        else:
            # 兜底：直接使用原始 role
            prefix = role or "System"
        parts.append(f"{prefix}:\n{content}\n")
    return "\n".join(parts).strip()


def get_res_vllm_messages(
    messages: List[Dict[str, str]],
    model_name: str,
    base_url: str = "http://127.0.0.1:57723/v1",
    temperature: float = 0.5,
    max_retries: int = 3,
    request_index: Optional[int] = None
) -> str:
    """
    使用本地 vLLM 服务进行推理（通过 OpenAI 兼容接口）。
    用于调用经过微调的模型（toolllama、qwenft、llama）。
    
    Args:
        messages: 消息列表，包含 system, user, assistant 等角色
        model_name: 模型名称（例如 "toolqwen3"）
        base_url: vLLM 服务的 base URL，默认为 "http://127.0.0.1:8000/v1"
        temperature: 生成温度，默认为 0.5
        max_retries: 最大重试次数
        request_index: 请求索引（用于批量推理时的标识），可选
        
    Returns:
        str: 模型的最终回答（已提取思考标签后的内容）
    """
    if not OPENAI_CLIENT_AVAILABLE:
        return "[ERROR] OpenAI 客户端未安装，无法调用 vLLM 服务。安装命令: pip install openai"
    
    # 打印最终发送给 vLLM 的 messages（用于调试）
    index_prefix = f"[请求 {request_index + 1}] " if request_index is not None else ""
    print(f"{index_prefix}===== MODEL PROMPT (vLLM) START =====")
    for msg in messages:
        print(f"{index_prefix}{msg.get('role', 'unknown')}: {msg.get('content', '')}")
    print(f"{index_prefix}===== MODEL PROMPT (vLLM) END =====")
    
    # 创建 OpenAI 客户端
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY"  # vLLM 默认不会校验，用随便字符串即可
    )
    # 将 messages 扁平化为单个 prompt（适配不支持 chat 接口的模型，如 toollama）
    prompt = _build_prompt_from_messages(messages)
    
    # 重试逻辑
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=256,
                temperature=temperature,
                extra_body={
                    "enable_thinking": False
                }
            )
            
            # 获取完整内容（completion 接口返回 text）
            full_content = resp.choices[0].text or ""
            
            # 提取最终回答（移除思考标签）
            final_response = extract_final_response(full_content)
            
            # 对 ToolLLaMA 风格的输出进行二次解析，转成 API-Bank 风格
            # 这里依赖 model_name == "toollama"（在主流程里由 method="toolllama" 映射而来）
            if model_name.lower() in ("toollama", "toolllama"):
                converted = _convert_toolllama_output_to_apibank_style(final_response)
                # 如果是 None，表示 Action 为 finish，不调用 API
                # 下游 get_api_call 会返回 None
                if converted is None:
                    final_response = ""
                else:
                    final_response = converted
            
            print(f"完整返回内容: {full_content}...")
            print(f"提取的最终回答: {final_response}...")
            
            return final_response
            
        except Exception as e:
            error_msg = f"[ERROR] vLLM 请求异常: {str(e)}"
            print(f"尝试 {attempt + 1}/{max_retries}: {error_msg}")
            last_error = error_msg
            if attempt < max_retries - 1:
                time.sleep(2)  # 等待后重试
                continue
            else:
                return error_msg
    
    # 如果所有重试都失败，返回最后的错误信息
    return last_error if last_error else "[ERROR] 未知错误"


def get_res_vllm_messages_batch(
    messages_list: List[List[Dict[str, str]]],
    model_name: str,
    base_url: str = "http://127.0.0.1:57723/v1",
    temperature: float = 0.5,
    max_retries: int = 3,
    max_workers: int = 8
) -> List[str]:
    """
    使用本地 vLLM 服务进行批量推理（通过 OpenAI 兼容接口）。
    用于批量调用经过微调的模型（toolllama、qwenft、llama）。
    
    Args:
        messages_list: 消息列表的列表，每个元素是一个 messages 列表
        model_name: 模型名称（例如 "toolqwen3"）
        base_url: vLLM 服务的 base URL，默认为 "http://127.0.0.1:8000/v1"
        temperature: 生成温度，默认为 0.5
        max_retries: 最大重试次数
        max_workers: 最大并发线程数，默认为 8
        
    Returns:
        List[str]: 每个请求的模型回答列表（已提取思考标签后的内容）
    """
    if not OPENAI_CLIENT_AVAILABLE:
        return ["[ERROR] OpenAI 客户端未安装，无法调用 vLLM 服务。安装命令: pip install openai"] * len(messages_list)
    
    if not messages_list:
        return []
    
    print(f"===== 批量推理开始，共 {len(messages_list)} 个请求 =====")
    
    # 使用线程池进行并发请求
    results = [None] * len(messages_list)
    
    def process_single(index: int, messages: List[Dict[str, str]]) -> Tuple[int, str]:
        """处理单个请求的辅助函数"""
        try:
            result = get_res_vllm_messages(
                messages=messages,
                model_name=model_name,
                base_url=base_url,
                temperature=temperature,
                max_retries=max_retries,
                request_index=index
            )
            return index, result
        except Exception as e:
            error_msg = f"[ERROR] 批量推理中单个请求失败: {str(e)}"
            print(f"  请求 {index + 1} 失败: {error_msg}")
            return index, error_msg
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, idx, msgs): idx
            for idx, msgs in enumerate(messages_list)
        }
        
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            print(f"  完成请求 {idx + 1}/{len(messages_list)}")
    
    print(f"===== 批量推理完成 =====")
    return results


def send_tool_call_result(
    record: Dict[str, Any],
    workspace_id: str,
    endpoint: str = "http://localhost:8002/add_tool_call_result",
    timeout: float = 30.0,
) -> bool:
    """
    从一条评估记录中提取工具调用信息并发送到工具记忆服务。
    
    参数:
        record: 评估记录，包含 model_output_raw, tool_result, chat_history, file 等字段
        workspace_id: 工作空间 ID
        endpoint: 接收工具调用结果的 HTTP endpoint
        timeout: HTTP 请求超时时间（秒）
    
    返回:
        bool: 发送是否成功
    """
    # 只处理 API 类型的记录
    if record.get("type") != "API":
        return False
    
    # 从 model_output_raw 中提取工具名称和输入（模型实际调用）
    model_output_raw = record.get("model_output_raw", "")
    api_name, tool_input = parse_model_output_raw(model_output_raw)
    
    if not api_name:
        # 如果无法从 model_output_raw 解析，跳过
        return False
    
    # 从 tool_result 中提取输出
    tool_result = record.get("tool_result")
    
    # 判断 success 和提取 output
    # tool_result 可能是：
    # 1. 字典：包含 output 和 exception 字段（来自 extract_tool_posts_from_jsonl.py 的格式）
    # 2. 字典：直接是工具调用的结果（来自 evaluator.evaluate 的成功返回）
    # 3. 字符串：错误信息（来自 evaluator.evaluate 的失败返回）
    if isinstance(tool_result, dict):
        # 检查是否有 exception 字段（extract_tool_posts_from_jsonl.py 的格式）
        if "exception" in tool_result:
            exception = tool_result.get("exception", None)
            success = exception is None
            final_output = extract_tool_output(tool_result)
        else:
            # 直接是工具调用的结果，视为成功
            success = True
            final_output = str(tool_result)
    else:
        # 如果 tool_result 是字符串（通常是错误信息），视为失败
        success = False
        final_output = extract_tool_output(tool_result)
    
    # 获取上下文和文件名
    context = record.get("chat_history", [])
    file_name = record.get("file", "")
    
    # 构造工具调用结果
    tool_call_result = {
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool_name": api_name,
        "context": str(context),
        "config": "/Users/xiaocan/Toolmem/experiment/api-bank/config.json",
        "input": tool_input if tool_input else {},
        "output": final_output,
        "token_cost": 0,
        "success": success,
        "time_cost": 0.0,
        "source_task": file_name,
    }
    
    # 构造 POST payload
    payload = {
        "workspace_id": workspace_id,
        "tool_call_results": [tool_call_result],
    }
    
    # 发送请求
    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        if resp.status_code != 200:
            logging.warning(
                f"发送工具调用结果失败，状态码: {resp.status_code}, "
                f"响应: {resp.text}, 工具: {api_name}, 文件: {file_name}"
            )
            return False
        
        body = {}
        try:
            body = resp.json()
        except Exception:
            pass
        
        if not body or not body.get("success", False):
            logging.warning(
                f"接口返回失败，工具: {api_name}, 文件: {file_name}, "
                f"响应: {body or resp.text}"
            )
            return False
        
        logging.debug(f"成功发送工具调用结果: {api_name}, 文件: {file_name}")
        return True
    except Exception as e:
        logging.warning(f"发送工具调用结果异常: {e}, 工具: {api_name}, 文件: {file_name}")
        return False


def extract_user_query_from_context_for_mem0(context: Any) -> str:
    """
    从 context 中提取用户查询（用于 mem0）。
    对于 API-Bank 数据集，context 可能是 chat_history（列表）或字符串形式。
    
    Args:
        context: chat_history 列表或字符串
        
    Returns:
        str: 提取出的用户查询内容
    """
    if not context:
        return ""
    
    # 如果 context 是字符串，尝试解析为 JSON（可能是列表的字符串表示）
    if isinstance(context, str):
        try:
            # 尝试解析为 JSON
            parsed = json.loads(context)
            context = parsed
        except (json.JSONDecodeError, ValueError):
            # 如果解析失败，可能是 Python 列表的字符串表示（如 "[{...}]"）
            # 尝试使用 ast.literal_eval
            try:
                parsed = ast.literal_eval(context)
                context = parsed
            except (ValueError, SyntaxError):
                # 如果都解析失败，直接返回字符串
                return context
    
    # 如果 context 是列表（chat_history），提取用户消息
    if isinstance(context, list):
        user_messages = []
        for item in context:
            if isinstance(item, dict) and item.get("role") == "User":
                text = item.get("text", "")
                if text:
                    user_messages.append(text)
        if user_messages:
            # 返回最后一条用户消息，或者拼接所有用户消息
            return user_messages[-1] if len(user_messages) == 1 else " ".join(user_messages)
        # 如果没有找到用户消息，返回整个 context 的字符串表示
        return str(context)
    
    # 其他情况，直接转换为字符串
    return str(context)


def send_tool_call_to_mem0(
    record: Dict[str, Any],
    memory_instance,
) -> bool:
    """
    从一条评估记录中提取工具调用信息并发送到 mem0。
    
    参数:
        record: 评估记录，包含 model_output_raw, tool_result, chat_history, file 等字段
        memory_instance: mem0 Memory 实例
    
    返回:
        bool: 发送是否成功
    """
    if memory_instance is None:
        logging.warning("mem0 Memory 实例未初始化，无法发送数据到 mem0")
        return False
    
    # 只处理 API 类型的记录
    if record.get("type") != "API":
        return False
    
    # 从 model_output_raw 中提取工具名称和输入（模型实际调用）
    model_output_raw = record.get("model_output_raw", "")
    api_name, tool_input = parse_model_output_raw(model_output_raw)
    
    if not api_name:
        # 如果无法从 model_output_raw 解析，跳过
        return False
    
    # 从 tool_result 中提取输出
    tool_result = record.get("tool_result")
    
    # 判断 success 和提取 output
    if isinstance(tool_result, dict):
        # 检查是否有 exception 字段
        if "exception" in tool_result:
            exception = tool_result.get("exception", None)
            success = exception is None
            final_output = extract_tool_output(tool_result)
        else:
            # 直接是工具调用的结果，视为成功
            success = True
            final_output = str(tool_result)
    else:
        # 如果 tool_result 是字符串（通常是错误信息），视为失败
        success = False
        final_output = extract_tool_output(tool_result)
    
    # 获取上下文和文件名
    context = record.get("chat_history", [])
    file_name = record.get("file", "")
    
    try:
        # 从 context 中提取用户查询
        user_query = extract_user_query_from_context_for_mem0(context)
        if not user_query:
            # 如果提取失败，使用完整的 context
            user_query = str(context)
        
        # 构建消息（参考 extract.py 的格式：user, assistant with tool_calls, tool）
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
                            "name": api_name,
                            "arguments": str(tool_input if tool_input else {})
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": "success: "+str(success)+" output: "+str(final_output)
            }
        ]
        
        # 发送到 mem0（使用 procedural_memory 类型，保持 run_id 参数）
        memory_instance.add(messages, agent_id=api_name, user_id=file_name, run_id='toole-ds', memory_type="procedural_memory")
        logging.debug(f"成功发送工具调用结果到 mem0: {api_name}, 文件: {file_name}")
        return True
    except Exception as e:
        logging.warning(f"发送工具调用结果到 mem0 异常: {e}, 工具: {api_name}, 文件: {file_name}")
        return False


class Sample:
    def __init__(self, chat_history, apis, ground_truth):
        self.chat_history = chat_history
        self.apis = apis
        self.ground_truth = ground_truth

    def __repr__(self):
        return 'Sample(chat_history={}, apis={}, ground_truth={})'.format(self.chat_history, self.apis, self.ground_truth)
        # return 'Chat history: {}, apis: {}, ground truth: {}'.format(self.chat_history, self.apis, self.ground_truth)

    @classmethod
    def from_chat_history(cls, chat_history):
        apis = set()
        api_positions = []
        for i, item in enumerate(chat_history):
            if item['role'] == 'API':
                apis.add(item['api_name']) 
                api_positions.append(i)

        samples = []
        for i in api_positions:
            sample = cls(chat_history[:i], apis, chat_history[i])
            samples.append(sample)
            sample = cls(chat_history[:i + 1], apis, chat_history[i + 1])
            samples.append(sample)

        return samples


class Evaluator:

    def __init__(
        self,
        samples,
        source_task=None,
        model_name: str = "unknown",
        workspace_id: str = "api_bank",
        method: Optional[str] = None,
        draft_descriptions_file: Optional[str] = None,
    ):
        """评估器

        参数:
            samples: 数据集样本
            source_task: 当前数据对应的任务名（用于记忆检索）
            model_name: 模型名称（用于记忆检索打标）
            workspace_id: 向量库 workspace
            method: 记忆方法，可选值：None, "mem", "m0", "draft", "toolllama", "qwenft", "llama"
                - None: 不使用记忆，使用常规模型调用
                - "mem": 使用 ToolMemoryFetcher 服务
                - "m0": 使用 mem0 进行记忆管理
                - "draft": 使用 DRAFT 改进后的工具描述（需要先运行 draft_improver.py 生成改进后的描述文件）
                - "toolllama"/"qwenft"/"llama": 使用本地 vLLM 服务调用微调模型（需要安装 openai: pip install openai，且需要本地 vLLM 服务运行在 http://127.0.0.1:8000/v1）
            draft_descriptions_file: DRAFT 改进后的描述文件路径，默认值为 "draft_improved_descriptions.json"
        """
        self.dataset = samples
        self.sample_ids = list(range(len(self.dataset)))
        self.method = method
        self.workspace_id = workspace_id
        self.source_task = source_task or ""
        self.model_name = model_name

        # 根据 method 初始化相应的记忆服务
        self.tool_memory_fetcher = None
        if self.method == "mem":
            self.tool_memory_fetcher = ToolMemoryFetcher(
                workspace_id=workspace_id,
                source_task=self.source_task,
                model_name=model_name,
            )
        elif self.method == "m0":
            # mem0 相关导入（可选，如果未安装会优雅降级）
            try:
                from mem0 import Memory
                self.mem0_available = True
            except ImportError:
                self.mem0_available = False
                print("警告: mem0 未安装，--method m0 选项将不可用。安装命令: pip install mem0ai")

        self._memory_cache = {}
        self.tools_with_memory = set()
        self.tools_without_memory = set()
        # mem0 实例使用全局单例（模块级别），避免 Qdrant 并发访问问题
        # self._mem0_memory_instance 不再使用，保留仅为兼容性
        self._mem0_memory_instance = None
        
        # DRAFT 改进描述加载
        self._draft_descriptions = {}
        if self.method == "draft":
            draft_file = draft_descriptions_file or "draft_improved_descriptions.json"
            if os.path.exists(draft_file):
                try:
                    with open(draft_file, 'r', encoding='utf-8') as f:
                        self._draft_descriptions = json.load(f)
                    print(f"已加载 DRAFT 改进描述文件: {draft_file}，包含 {len(self._draft_descriptions)} 个 API")
                except Exception as e:
                    print(f"警告: 加载 DRAFT 描述文件失败: {e}，将使用原始描述")
            else:
                print(f"警告: DRAFT 描述文件不存在: {draft_file}，将使用原始描述。请先运行 draft_improver.py 生成改进后的描述")

    def _get_mem0_memory(self):
        """获取 mem0 Memory 实例（延迟初始化，使用全局单例以避免 Qdrant 并发访问问题）。"""
        if self.method != "m0":
            return None
        
        if not hasattr(self, 'mem0_available') or not self.mem0_available:
            return None
        
        # 使用全局单例，避免多个 Evaluator 实例创建多个 Memory 实例导致 Qdrant 并发访问问题
        global _global_mem0_memory_instance, _global_mem0_memory_lock
        
        # 如果全局实例已存在，直接返回
        if _global_mem0_memory_instance is not None:
            return _global_mem0_memory_instance
        
        # 使用锁确保只有一个线程创建实例
        with _global_mem0_memory_lock:
            # 双重检查，防止多个线程同时通过锁
            if _global_mem0_memory_instance is not None:
                return _global_mem0_memory_instance
            
            try:
                from mem0 import Memory
                # 使用与 run.py 相同的配置
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
                            "model": "qwen3-max",
                            "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
                        }
                    },
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "embedding_model_dims": 512,
                            "on_disk": True
                        }
                    }
                }
                _global_mem0_memory_instance = Memory.from_config(config)
                return _global_mem0_memory_instance
            except Exception as e:
                print(f"错误: 初始化 mem0 Memory 失败: {e}")
                return None

    
    def _record_memory_stat(self, api_name: str, memory: str) -> None:
        """Track which tools have memory available."""
        if memory:
            self.tools_with_memory.add(api_name)
            self.tools_without_memory.discard(api_name)
        else:
            if api_name not in self.tools_with_memory:
                self.tools_without_memory.add(api_name)

    def get_all_sample_ids(self):
        return self.sample_ids
    
    def get_api_description(self, api_name):
        tool_manager = ToolManager()
        
        # 如果使用 DRAFT 方法，优先使用改进后的描述
        if self.method == "draft" and api_name in self._draft_descriptions:
            improved_desc = self._draft_descriptions[api_name]
            description = json.dumps(improved_desc, ensure_ascii=False)
        else:
            description = tool_manager.get_api_description(api_name)

        # 未开启 memory 时，直接返回描述（可能是原始或改进后的）
        if not self.method or self.method not in ("mem", "m0"):
            return description

        # 获取 memory（根据 method 选择不同的方法）
        if api_name not in self._memory_cache:
            if self.method == "mem":
                # 使用 ToolMemoryFetcher 服务
                if self.tool_memory_fetcher:
                    memory_dict = self.tool_memory_fetcher.collect_memory([api_name]).get(api_name, {})
                    memory = memory_dict.get("content", "") if isinstance(memory_dict, dict) else ""
                else:
                    memory = ""
            elif self.method == "m0":
                # 从 mem0 获取记忆
                memory_instance = self._get_mem0_memory()
                if memory_instance:
                    memory_map = get_mem0_memories_for_tools([api_name], self.source_task, memory_instance)
                    memory = memory_map.get(api_name, "") or ""
                else:
                    memory = ""
            else:
                memory = ""
            self._memory_cache[api_name] = memory
        else:
            memory = self._memory_cache.get(api_name) or ""
        
        self._record_memory_stat(api_name, memory)
        if memory:
            description = (
                f"{description} \n Besides, this is optional guidance on how to better use the tool. You may refer to it selectively:\n{memory}"
            )
        return description

    
    def get_model_input(self, sample_id):
        sample = self.dataset[sample_id]
        apis = sample.apis
        chat_history = sample.chat_history
        tool_manager = ToolManager()
        api_descriptions = []
        for api_name in apis:
            # 如果使用 DRAFT 方法，优先使用改进后的描述
            if self.method == "draft" and api_name in self._draft_descriptions:
                improved_desc = self._draft_descriptions[api_name]
                tool_schema = json.dumps(improved_desc, ensure_ascii=False)
            else:
                tool_schema = tool_manager.get_api_description(api_name)
            # 获取 memory（根据 method 选择不同的方法）
            if self.method in ("mem", "m0"):
                if api_name not in self._memory_cache:
                    if self.method == "mem":
                        # 使用 ToolMemoryFetcher 服务
                        if self.tool_memory_fetcher:
                            memory_dict = self.tool_memory_fetcher.collect_memory([api_name]).get(api_name, {})
                            memory = memory_dict.get("content", "") if isinstance(memory_dict, dict) else ""
                        else:
                            memory = ""
                    elif self.method == "m0":
                        # 从 mem0 获取记忆
                        memory_instance = self._get_mem0_memory()
                        if memory_instance:
                            memory_map = get_mem0_memories_for_tools([api_name], self.source_task, memory_instance)
                            memory = memory_map.get(api_name, "") or ""
                        else:
                            memory = ""
                    else:
                        memory = ""
                    self._memory_cache[api_name] = memory
                else:
                    memory = self._memory_cache.get(api_name) or ""
                
                self._record_memory_stat(api_name, memory)
                if memory:
                    # 尝试把字符串 JSON 解析后，在 description 后拼接 memory；失败则直接字符串拼接
                    try:
                        schema_obj = json.loads(tool_schema)
                        if isinstance(schema_obj, dict) and "description" in schema_obj:
                            schema_obj["description"] = (
                                f"{schema_obj['description']} \n Besides, this is optional guidance on how to better use the tool. You may refer to it selectively: \n{memory}"
                            )
                            tool_schema = json.dumps(schema_obj, ensure_ascii=False)
                        else:
                            tool_schema = (
                                f"{tool_schema}\n Besides, this is optional guidance on how to better use the tool. You may refer to it selectively: \n{memory}"
                            )
                    except Exception as e:
                        print("⚠️⚠️⚠️⚠️⚠️exception happen:", e)
            api_descriptions.append(tool_schema)
        api_descriptions = '\n'.join(api_descriptions)
        return api_descriptions, chat_history

    
    def evaluate(self, sample_id, model_output):
        # model_output [ApiName(param1=value1, param2=value2), ...)]
        tool_manager = ToolManager()

        sample = self.dataset[sample_id]
        ground_truth = sample.ground_truth
        if ground_truth['role'] == 'API':
            api_name, param_dict = parse_api_call(model_output)
            if api_name != ground_truth['api_name']:
                return False, 'API Name Mismatch: {} vs {}'.format(api_name, ground_truth['api_name'])
            try:
                result = tool_manager.api_call(api_name, **param_dict)
            except Exception as e:
                return False, str(e)
            api = tool_manager.init_tool(api_name)
            try:
                # 深拷贝避免被下游比对逻辑修改
                res_copy = copy.deepcopy(result)
                gt_copy = copy.deepcopy(ground_truth['result'])
                correct = api.check_api_call_correctness(res_copy, gt_copy)
                # 兜底1：大小写不敏感比较（处理大小写不一致的情况）
                if not correct and _lower_obj(res_copy) == _lower_obj(gt_copy):
                    correct = True
                # 兜底2：字符串形式的 list/dict 解析后再比（处理格式不一致的情况）
                if not correct and _lower_obj(_normalize_container(res_copy)) == _lower_obj(_normalize_container(gt_copy)):
                    correct = True
            except KeyError:
                correct = False
                result = 'KeyError' + str(result)
            return correct, result
        elif ground_truth['role'] == 'AI':
            score = calculate_rouge_l_score(ground_truth['text'].lower(), model_output.lower())
            return round(score, 4)


def get_api_call(model_output):
    api_call_pattern = r"\[(\w+)\((.*)\)\]"
    api_call_pattern = re.compile(api_call_pattern)
    match = api_call_pattern.search(model_output)
    if match:
        return match.group(0)
    else:
        return None


def get_mem0_memory():
    """
    获取全局的 mem0 Memory 实例（单例模式）。
    如果尚未初始化，则进行初始化。
    
    Returns:
        Memory 实例，如果初始化失败则返回 None
    """
    # 注意：这里使用模块级别的全局变量，但实际应该在 Evaluator 类中管理
    # 为了简化，我们暂时返回 None，由 Evaluator 类自己管理
    return None


def get_mem0_memories_for_tools(tool_names: list, current_task_id: str, memory_instance) -> Dict[str, str]:
    """
    从 mem0 获取指定工具的记忆，排除当前 task 的记录。
    限制每个工具的 memory 长度为 120000/工具数，超过部分截断。
    
    Args:
        tool_names: 工具名称列表
        current_task_id: 当前任务的 ID（用于排除）
        memory_instance: mem0 Memory 实例
    
    Returns:
        Dict[str, str]: 工具名称到拼接后的 memory 文本的映射
    """
    memory_map: Dict[str, str] = {name: "" for name in tool_names}
    
    if memory_instance is None:
        return memory_map
    
    # 计算每个工具的最大 memory 长度
    num_tools = len(tool_names) if tool_names else 1
    max_length_per_tool = 120000 // num_tools
    
    for tool_name in tool_names:
        try:
            # 获取该工具的所有记忆
            all_memories_result = memory_instance.get_all(agent_id=tool_name, run_id='toole-ds')
            if not all_memories_result or "results" not in all_memories_result:
                continue
            
            # 过滤掉当前 task 的记录，并拼接 memory 文本
            memory_texts = []
            for mem_item in all_memories_result["results"]:
                user_id = mem_item.get("user_id", "")
                # 排除当前 task 的记录
                if str(user_id) != str(current_task_id):
                    mem_text = mem_item.get("memory", "")
                    if mem_text:
                        memory_texts.append(mem_text)
            
            # 将多个 memory 拼接起来，用空格分隔
            if memory_texts:
                prefix = "Besides, this is optional guidance on how to better use the tool. You may refer to it selectively "
                combined_memory = " ".join(memory_texts)
                
                # 限制长度：如果超过最大长度，截断
                if len(combined_memory) > max_length_per_tool:
                    combined_memory = combined_memory[:max_length_per_tool]
                    # 尝试在单词边界处截断，避免截断单词
                    last_space = combined_memory.rfind(' ')
                    if last_space > max_length_per_tool * 0.9:  # 如果最后一个空格在90%位置之后，使用它
                        combined_memory = combined_memory[:last_space]
                
                memory_map[tool_name] = prefix + combined_memory
        except Exception as e:
            print(f"警告: 获取工具 {tool_name} 的 mem0 记忆失败: {e}")
            continue
    
    return memory_map

if __name__ == '__main__':
    datas = ['lv1-lv2-samples/level-1-given-desc']
    # datas = ['lv1-lv2-samples/level-2-toolsearcher']
    # 每次模型推理的采样数目
    n_sample = 3
    for data_dir in datas:
        api_test_enabled = True
        dialog_test_enabled = False
        # 记忆方法，可选值：None, "mem", "m0", "draft", "toolllama", "qwenft", "llama"
        # None: 不使用记忆，使用常规模型调用
        # "mem": 使用 ToolMemoryFetcher 服务
        # "m0": 使用 mem0 进行记忆管理
        # "draft": 使用 DRAFT 改进后的工具描述（需要先运行 draft_improver.py 生成改进后的描述文件）
        # "toolllama"/"qwenft"/"llama": 使用本地 vLLM 服务调用微调模型（需要安装 openai: pip install openai，且需要本地 vLLM 服务运行在 http://127.0.0.1:8000/v1）
        method = 'mem'
        output_path_base = "deepseek-v3"  # 汇总每条样本的推理与评估的基础文件名
        workspace_id = "deepseek-v3"
        # output_path_base = "qwen3-8b"  # 汇总每条样本的推理与评估的基础文件名

        api_str = 'api' if api_test_enabled else 'response'


        if os.path.basename(data_dir).endswith('given-desc'):
            tool_search_enabled = False
        else:
            tool_search_enabled = True
        # 工具调用与回复生成统一使用 qwen3-8b；需事先 export DASHSCOPE_API_KEY
        api_chatgpt = QwenOpenAIWrapper(model=output_path_base)
        dialog_chatgpt = api_chatgpt

        # 根据不同 method 设计不同的提示格式
        if method == "toolllama":
            # ToolLLaMA 风格提示：整体结构与其他方法一致，只是输出格式改为 Action / Action Input
            api_call_prompt = '''
    Based on the given API description and the existing conversation history 1..t, please decide which API the AI should call in step t+1 and what arguments to use.
    Your answer MUST follow the format:
    
    Action: <ApiName>
    Action Input: <JSON object with arguments>
    
    - <ApiName> MUST be one of the provided APIs.
    - The JSON object MUST be valid JSON, containing all required arguments for this API.
    - Do NOT include any explanations, natural language, or additional text outside of the two lines above.
    
    This year is 2023.
    Input: 
    User: [User's utterence]
    AI: [AI's utterence]
    
    Expected output:
    Action: <ApiName>
    Action Input: <JSON object with arguments>
    
    API descriptions:
    '''
        else:
            # 默认提示：要求输出 [ApiName(key1='value1', ...)] 格式
            api_call_prompt = '''
    Based on the given API description and the existing conversation history 1..t, please generate the API request that the AI should call in step t+1 and output it in the format of [ApiName(key1='value1', key2='value2', ...)], replace the ApiName with the actual API name, and replace the key and value with the actual parameters. 
    Your output should start with a square bracket "[" and end with a square bracket "]". Do not output any other explanation or prompt or the result of the API call in your output. 
    This year is 2023.
    Input: 
    User: [User's utterence]
    AI: [AI's utterence]
    
    Expected output:
    [ApiName(key1='value1', key2='value2', ...)]
    
    API descriptions:
    '''

        response_prompt = '''
    Based on the given API description and the existing conversation history 1..t, please generate the next dialog that the AI should response after the API call t.
    This year is 2023.
    Input: 
    User: [User's utterence]
    AI: [AI's utterence]
    [ApiName(key1='value1', key2='value2', …)]

    Expected output:
    AI: [AI's utterence]

    API descriptions:
    '''

        total_api_calls = 0
        correct_api_calls = 0
        api_calls_with_memory = 0
        api_calls_without_memory = 0
        total_after_api_response = 0
        total_score_after_api_response = 0

        rougel_scores = []

        jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]

        # 读取已处理的样本，避免重复处理
        # 根据 n_sample 检查所有采样文件（0 到 n_sample-1）
        processed_samples = {}
        # processed_samples 的结构: {(file, sample_id): [sample_idx1, sample_idx2, ...]}
        # 表示该样本在哪些采样文件中已经存在
        
        output_dir = os.path.dirname(output_path_base) if os.path.dirname(output_path_base) else '.'
        output_base_name = os.path.basename(output_path_base)
        
        # 检查所有采样文件（0 到 n_sample-1）
        for sample_idx in range(n_sample):
            result_file = get_output_path(output_path_base, sample_idx, method=method)
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                record = json.loads(line)
                                key = (record['file'], record['sample_id'])
                                if key not in processed_samples:
                                    processed_samples[key] = []
                                processed_samples[key].append(sample_idx)
                            except (json.JSONDecodeError, KeyError):
                                continue
        
        if processed_samples:
            total_processed = len(processed_samples)
            print(f"已加载 {total_processed} 个已处理的样本，将跳过已完成的采样")

        overall_tools_with_memory = set()
        overall_tools_without_memory = set()

        # 判断是否使用批量推理（仅 vLLM 方法支持）
        # 注意：m0 方法使用 Qdrant 本地存储，不支持并发访问，因此禁用批量推理
        use_batch_inference = method in ("toolllama", "qwenft", "llama") and method != "m0"
        
        for file in tqdm(jsonl_files, desc='Processing files', ncols=100):
            history = []
            with open(os.path.join(data_dir, file), 'r') as f:
                for line in f:
                    history.append(json.loads(line))
            samples = Sample.from_chat_history(history)
            evaluator = Evaluator(
                samples,
                source_task=os.path.basename(file),
                model_name=output_path_base,
                workspace_id=workspace_id,
                method=method,
            )

            for sample_id in evaluator.get_all_sample_ids():
                sample_key = (file, sample_id)
                
                # 检查哪些采样已完成
                completed_samples = processed_samples.get(sample_key, [])
                need_process = len(completed_samples) < n_sample
                
                # 如果所有采样都已完成，跳过该样本
                if not need_process:
                    continue
                
                sample = evaluator.dataset[sample_id]
                
                # 打印当前处理的样本信息
                gt_role = sample.ground_truth.get('role', 'Unknown')
                if gt_role == 'API':
                    gt_api_name = sample.ground_truth.get('api_name', 'Unknown')
                    print(f"\n处理样本: 文件={file}, sample_id={sample_id}, API={gt_api_name}, 已完成采样={len(completed_samples)}/{n_sample}")
                else:
                    print(f"\n处理样本: 文件={file}, sample_id={sample_id}, 类型={gt_role}, 已完成采样={len(completed_samples)}/{n_sample}")
                
                if sample.ground_truth['role'] == 'API' and api_test_enabled:
                    if tool_search_enabled:
                        _, chat_history = evaluator.get_model_input(sample_id)
                        api_descriptions = evaluator.get_api_description('ToolSearcher')
                    else:
                        api_descriptions, chat_history = evaluator.get_model_input(sample_id)
                    # 判断此次 API 调用是否有可用的 memory（基于 ground_truth 的 api_name）
                    gt_api_name = sample.ground_truth.get('api_name')
                    memory_available = bool(evaluator._memory_cache.get(gt_api_name))
                    if memory_available:
                        api_calls_with_memory += 1
                    else:
                        api_calls_without_memory += 1
                    prompt = api_call_prompt + api_descriptions
                    messages = [
                        {'role': 'system', 'content': prompt},
                    ]
                    for item in chat_history:
                        if item['role'] == 'User':
                            chat_role = 'user'
                            chat_content = item['text']
                        elif item['role'] == 'AI':
                            chat_role = 'assistant'
                            chat_content = item['text']
                        elif item['role'] == 'API':
                            # 保持使用 system 角色，与其他方法一致
                            chat_role = 'system'
                            chat_content = '[{}({})] Response: {}'.format(item['api_name'], ', '.join(['{}=\'{}\''.format(k, v) for k, v in item['param_dict'].items()]), str(item['result']['output']))
                        else:
                            raise ValueError('Invalid chat role: {}'.format(item['role']))
                        messages.append({'role': chat_role, 'content': chat_content})

                    # 收集批量推理请求（vLLM 方法支持）
                    # 对于每个样本，收集所有未完成的采样，使用相同的 messages 批量处理
                    batch_requests = []  # 存储该样本所有需要批量处理的请求（每个采样使用相同的 messages）
                    batch_api_descriptions = []  # 存储对应的 API 描述
                    batch_indices = []  # 存储对应的索引信息
                    
                    # 先收集所有需要处理的采样索引
                    if use_batch_inference:
                        for sample_idx in range(n_sample):
                            if sample_idx not in completed_samples:
                                batch_requests.append(messages)  # 相同的 messages，但会被处理多次以支持采样
                                batch_api_descriptions.append(api_descriptions)  # 相同的 API 描述
                                batch_indices.append({
                                    'sample_id': sample_id,
                                    'sample_idx': sample_idx,
                                    'chat_history': chat_history,
                                })
                        
                        # 如果有需要批量处理的请求，执行批量推理
                        if batch_requests:
                            print(f"\n  开始批量推理该样本的 {len(batch_requests)} 个采样...")
                            
                            if method in ("toolllama", "qwenft", "llama"):
                                # 根据 method 确定模型名称
                                model_name_map = {
                                    "toolllama": "toollama",
                                    "qwenft": "toolqwen3",
                                    "llama": "llama"
                                }
                                vllm_model_name = model_name_map.get(method, "toolqwen3")
                                
                                # 使用 vLLM 批量推理
                                model_outputs = get_res_vllm_messages_batch(
                                    messages_list=batch_requests,
                                    model_name=vllm_model_name,
                                    temperature=0.5
                                )
                            else:
                                # 其他方法不支持批量推理，应该不会到这里
                                model_outputs = []
                            
                            # 处理每个结果
                            for idx, (model_output, batch_info) in enumerate(zip(model_outputs, batch_indices)):
                                sample_idx = batch_info['sample_idx']
                                sample_id_batch = batch_info['sample_id']
                                chat_history_batch = batch_info['chat_history']
                                
                                try:
                                    api_call = get_api_call(model_output)
                                    record = {
                                        'file': file,
                                        'sample_id': sample_id_batch,
                                        'type': 'API',
                                        'chat_history': chat_history_batch,
                                        'ground_truth': sample.ground_truth,
                                        'model_output_raw': model_output,
                                        'parsed_api_call': api_call,
                                    }
                                    if api_call:
                                        try:
                                            correct, model_output_result = evaluator.evaluate(sample_id_batch, api_call)
                                        except AssertionError as e:
                                            if not 'The API name is not correct.' in str(e):
                                                raise e
                                            logging.info('AssertionError: {}'.format(e))
                                            correct = False
                                            model_output_result = str(e)
                                    else:
                                        model_output_result = 'No API call found'
                                        correct = False
                                    if correct:
                                        correct_api_calls += 1
                                        logging.info('Correct API call: {} Ground truth: {}'.format(api_call, sample.ground_truth))
                                    else:                    
                                        logging.info('Incorrect model output: {} Result: {} Ground truth: {} File: {} Sample ID: {}'.format(model_output.replace('\n', ' '), model_output_result, sample.ground_truth, file, sample_id_batch))
                                    total_api_calls += 1
                                    record.update({
                                        'tool_result': model_output_result,
                                        'correct': correct,
                                        'messages': batch_requests[idx],
                                    })
                                    current_output_path = get_output_path(output_path_base, sample_idx, method=method)
                                    write_record(record, current_output_path)
                                    
                                    # 如果启用 memory（method == "mem"），只对第一个采样发送工具调用结果到工具记忆服务
                                    if method == "mem" and sample_idx == 0:
                                        send_tool_call_result(
                                            record=record,
                                            workspace_id=evaluator.workspace_id,
                                            endpoint="http://localhost:8002/add_tool_call_result",
                                            timeout=30.0,
                                        )
                                    # 如果启用 m0（method == "m0"），只对第一个采样发送工具调用结果到 mem0
                                    elif method == "m0" and sample_idx == 0:
                                        memory_instance = evaluator._get_mem0_memory()
                                        send_tool_call_to_mem0(
                                            record=record,
                                            memory_instance=memory_instance,
                                        )
                                except Exception as e:
                                    error_msg = f"[ERROR] 批量推理结果处理失败: {str(e)}"
                                    print(f"  错误 {sample_idx + 1}: {error_msg}")
                                    logging.error(f"采样 {sample_idx + 1} 结果处理失败: {error_msg}")
                                    record = {
                                        'file': file,
                                        'sample_id': sample_id_batch,
                                        'type': 'API',
                                        'chat_history': chat_history_batch,
                                        'ground_truth': sample.ground_truth,
                                        'model_output_raw': error_msg,
                                        'parsed_api_call': None,
                                        'tool_result': error_msg,
                                        'correct': False,
                                        'messages': batch_requests[idx],
                                    }
                                    current_output_path = get_output_path(output_path_base, sample_idx, method=method)
                                    write_record(record, current_output_path)
                    else:
                        # 非批量推理方法，保持原有逻辑
                        # 对每个未完成的采样进行推理
                        for sample_idx in range(n_sample):
                            if sample_idx in completed_samples:
                                print(f"  采样 {sample_idx + 1}/{n_sample} 已存在，跳过")
                                continue
                            
                            try:
                                print(f"  采样 {sample_idx + 1}/{n_sample}...")
                                
                                # 使用常规模型调用
                                # GPT 系列模型不需要 enable_thinking 参数，其他模型需要设置为 False
                                call_kwargs = {
                                    "temperature": 0.5
                                }
                                # 只有非 GPT 系列模型才添加 enable_thinking 参数
                                if "gpt" not in output_path_base.lower():
                                    call_kwargs["extra_body"] = {"enable_thinking": False}
                                response = api_chatgpt.call(messages, **call_kwargs)
                                # 兼容 OpenAI/Completion 返回结构
                                if 'message' in response['choices'][0]:
                                    model_output = response['choices'][0]['message']['content']
                                else:
                                    model_output = response['choices'][0]['text'].strip()
                                api_call = get_api_call(model_output)
                                
                                # 处理结果（与批量推理共享的代码）
                                record = {
                                    'file': file,
                                    'sample_id': sample_id,
                                    'type': 'API',
                                    'chat_history': chat_history,
                                    'ground_truth': sample.ground_truth,
                                    'model_output_raw': model_output,
                                    'parsed_api_call': api_call,
                                }
                                if api_call:
                                    try:
                                        correct, model_output_result = evaluator.evaluate(sample_id, api_call)
                                    except AssertionError as e:
                                        if not 'The API name is not correct.' in str(e):
                                            raise e
                                        logging.info('AssertionError: {}'.format(e))
                                        correct = False
                                        model_output_result = str(e)
                                else:
                                    model_output_result = 'No API call found'
                                    correct = False
                                if correct:
                                    correct_api_calls += 1
                                    logging.info('Correct API call: {} Ground truth: {}'.format(api_call, sample.ground_truth))
                                else:                    
                                    logging.info('Incorrect model output: {} Result: {} Ground truth: {} File: {} Sample ID: {} Messages: {}'.format(model_output.replace('\n', ' '), model_output_result, sample.ground_truth, file, sample_id, messages[1:]))
                                total_api_calls += 1
                                record.update({
                                    'tool_result': model_output_result,
                                    'correct': correct,
                                    'messages': messages,
                                })
                                current_output_path = get_output_path(output_path_base, sample_idx, method=method)
                                write_record(record, current_output_path)
                                
                                # 如果启用 memory（method == "mem"），只对第一个采样发送工具调用结果到工具记忆服务
                                if method == "mem" and sample_idx == 0:
                                    send_tool_call_result(
                                        record=record,
                                        workspace_id=evaluator.workspace_id,
                                        endpoint="http://localhost:8002/add_tool_call_result",
                                        timeout=30.0,
                                    )
                                # 如果启用 m0（method == "m0"），只对第一个采样发送工具调用结果到 mem0
                                elif method == "m0" and sample_idx == 0:
                                    memory_instance = evaluator._get_mem0_memory()
                                    send_tool_call_to_mem0(
                                        record=record,
                                        memory_instance=memory_instance,
                                    )
                            except Exception as e:
                                error_msg = f"[ERROR] 推理失败: {str(e)}"
                                print(f"  错误 {sample_idx + 1}: {error_msg}")
                                logging.error(f"采样 {sample_idx + 1} 失败: {error_msg}")
                                # 即使失败也记录，避免重复尝试
                                record = {
                                    'file': file,
                                    'sample_id': sample_id,
                                    'type': 'API',
                                    'chat_history': chat_history,
                                    'ground_truth': sample.ground_truth,
                                    'model_output_raw': error_msg,
                                    'parsed_api_call': None,
                                    'tool_result': error_msg,
                                    'correct': False,
                                    'messages': messages,
                                }
                                current_output_path = get_output_path(output_path_base, sample_idx, method=method)
                                write_record(record, current_output_path)
                elif sample.ground_truth['role'] == 'AI' and dialog_test_enabled:
                    api_descriptions, chat_history = evaluator.get_model_input(sample_id)
                    prompt = response_prompt + api_descriptions
                    messages = [
                        {'role': 'system', 'content': prompt},
                    ]
                    for item in chat_history:
                        if item['role'] == 'User':
                            chat_role = 'user'
                            chat_content = item['text']
                        elif item['role'] == 'AI':
                            chat_role = 'assistant'
                            chat_content = item['text']
                        elif item['role'] == 'API':
                            chat_role = 'system'
                            chat_content = '[{}({})] Response: {}'.format(item['api_name'], ', '.join(['{}=\'{}\''.format(k, v) for k, v in item['param_dict'].items()]), str(item['result']['output']))
                        else:
                            raise ValueError('Invalid chat role: {}'.format(item['role']))
                        messages.append({'role': chat_role, 'content': chat_content})
                    
                    # 对每个未完成的采样进行推理
                    for sample_idx in range(n_sample):
                        if sample_idx in completed_samples:
                            print(f"  采样 {sample_idx + 1}/{n_sample} 已存在，跳过")
                            continue
                        
                        try:
                            print(f"  采样 {sample_idx + 1}/{n_sample}...")
                            
                            # 根据 method 选择推理方式
                            if method in ("toolllama", "qwenft", "llama"):
                                # 使用本地 vLLM 服务调用微调模型
                                # 根据 method 确定模型名称
                                model_name_map = {
                                    "toolllama": "toolllama",
                                    "qwenft": "toolqwen3",
                                    "llama": "toolllama"
                                }
                                vllm_model_name = model_name_map.get(method, "toolqwen3")
                                model_output = get_res_vllm_messages(
                                    messages=messages,
                                    model_name=vllm_model_name,
                                    temperature=0.5
                                )
                            else:
                                # 使用常规模型调用
                                # GPT 系列模型不需要 enable_thinking 参数，其他模型需要设置为 False
                                call_kwargs = {
                                    "temperature": 0.5
                                }
                                # 只有非 GPT 系列模型才添加 enable_thinking 参数
                                if "gpt" not in output_path_base.lower():
                                    call_kwargs["extra_body"] = {"enable_thinking": False}
                                response = dialog_chatgpt.call(messages, **call_kwargs)
                                if 'message' in response['choices'][0]:
                                    model_output = response['choices'][0]['message']['content']
                                else:
                                    model_output = response['choices'][0]['text'].strip()

                            if model_output:
                                score = evaluator.evaluate(sample_id, model_output)
                            else:
                                score = 0    
                            rougel_scores.append(score)
                            if score < 0.2:
                                logging.info('Low score: {} Score: {} Ground truth: {} File: {} Sample ID: {} Messages: {}'.format(model_output.replace('\n', ' '), score, sample.ground_truth, file, sample_id, messages[1:]))
                            record = {
                                'file': file,
                                'sample_id': sample_id,
                                'type': 'Dialog',
                                'chat_history': chat_history,
                                'ground_truth': sample.ground_truth,
                                'model_output_raw': model_output,
                                'rouge_l': score,
                                'messages': messages,
                            }
                            current_output_path = get_output_path(output_path_base, sample_idx, method=method)
                            write_record(record, current_output_path)
                        except Exception as e:
                            error_msg = f"[ERROR] 推理失败: {str(e)}"
                            print(f"  错误 {sample_idx + 1}: {error_msg}")
                            logging.error(f"采样 {sample_idx + 1} 失败: {error_msg}")
                            # 即使失败也记录，避免重复尝试
                            record = {
                                'file': file,
                                'sample_id': sample_id,
                                'type': 'Dialog',
                                'chat_history': chat_history,
                                'ground_truth': sample.ground_truth,
                                'model_output_raw': error_msg,
                                'rouge_l': 0,
                                'messages': messages,
                            }
                            current_output_path = get_output_path(output_path_base, sample_idx, method=method)
                            write_record(record, current_output_path)

            overall_tools_with_memory.update(evaluator.tools_with_memory)
            overall_tools_without_memory.update(evaluator.tools_without_memory)


        if dialog_test_enabled:
            print('Dialog score: {:.4f}'.format(np.mean(rougel_scores)))

        if api_test_enabled:
            print('Total API calls: {}'.format(total_api_calls))
            print('Correct API calls: {}'.format(correct_api_calls))
            print('Accuracy: {:.4f}'.format(correct_api_calls / total_api_calls))
            print('API calls with memory: {}'.format(api_calls_with_memory))
            print('API calls without memory: {}'.format(api_calls_without_memory))
            logging.info('Total API calls: {}'.format(total_api_calls))
            logging.info('Correct API calls: {}'.format(correct_api_calls))
            logging.info('Accuracy: {:.4f}'.format(correct_api_calls / total_api_calls))
            logging.info('API calls with memory: {}'.format(api_calls_with_memory))
            logging.info('API calls without memory: {}'.format(api_calls_without_memory))

        print(f"Tools with memory: {len(overall_tools_with_memory)}")
        print(f"Tools without memory: {len(overall_tools_without_memory)}")
