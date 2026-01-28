#!/usr/bin/env python3
"""
从 API-Bank 的本地 jsonl 结果文件（如 run_results_v2.jsonl）中直接提取工具调用结果，
并发送到工具记忆服务（mem 或 mem0）。

一行 json 的典型结构（简化）：
{
  "file": "QueryHistoryToday-level-3-3.jsonl",
  "sample_id": 0,
  "type": "API",
  "chat_history": [...],
  "ground_truth": {...},
  "model_output_raw": "...",
  "parsed_api_call": "...",
  "tool_result": {
      "api_name": "ToolSearcher",
      "input": {...},
      "output": {...},
      "exception": null
  },
  "correct": false,
  "messages": [...]
}

我们从 model_output_raw 和 tool_result 中提取：
    - tool_name: 从 model_output_raw 中解析（格式: [ToolName(...)]）
    - input:     从 model_output_raw 中解析参数
    - output:    从 tool_result 中提取（如果是字符串直接使用，否则解析字典中的 output 字段）
    - success:   根据 tool_result 判断（如果是字典且有 exception 则为失败，如果是字符串也视为失败）
    - 其它（时间、token_cost 等）暂时用默认值补齐，字段名保持与原先一致

支持两种发送方式：
    - --method mem: 发送到 HTTP 服务（默认）
    - --method m0: 发送到 mem0（需要安装 mem0ai）
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import os
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
import requests
import tqdm

# mem0 相关导入（可选，如果未安装会优雅降级）
try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    print("警告: mem0 未安装，--method m0 选项将不可用。安装命令: pip install mem0ai")


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


def extract_user_query_from_context(context: Any) -> str:
    """
    从 context 中提取用户查询。
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
                import ast
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


def get_mem0_memory():
    """
    获取全局的 mem0 Memory 实例（单例模式）。
    如果尚未初始化，则进行初始化。
    
    Returns:
        Memory 实例，如果初始化失败则返回 None
    """
    global _mem0_memory_instance
    
    if '_mem0_memory_instance' not in globals():
        globals()['_mem0_memory_instance'] = None
    
    if globals()['_mem0_memory_instance'] is not None:
        return globals()['_mem0_memory_instance']
    
    if not MEM0_AVAILABLE:
        return None
    
    try:
        config = get_mem0_config()
        globals()['_mem0_memory_instance'] = Memory.from_config(config)
        return globals()['_mem0_memory_instance']
    except Exception as e:
        print(f"错误: 初始化 mem0 Memory 失败: {e}")
        return None


def get_mem0_config():
    """
    获取 mem0 的默认配置。
    """
    return {
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


def send_tool_call_to_mem0(
    tool_call_record: Dict[str, Any],
) -> bool:
    """
    将单个工具调用记录发送到 mem0。
    
    Args:
        tool_call_record: 工具调用记录，包含 tool_name, context, input, output, success, token_cost, time_cost, source_task
        
    Returns:
        bool: 发送是否成功
    """
    memory = get_mem0_memory()
    if memory is None:
        print("错误: mem0 未安装或初始化失败，无法发送数据到 mem0")
        return False
    
    try:
        tool_name = tool_call_record.get("tool_name", "")
        context_full = tool_call_record.get("context", "")
        input_data = tool_call_record.get("input", {})
        output = tool_call_record.get("output", "")
        success = tool_call_record.get("success", False)
        token_cost = tool_call_record.get("token_cost", 0)
        time_cost = tool_call_record.get("time_cost", 0.0)
        source_task = tool_call_record.get("source_task", "")
        
        # 从 context 中提取用户查询
        user_query = extract_user_query_from_context(context_full)
        if not user_query:
            # 如果提取失败，使用完整的 context
            user_query = str(context_full)
        
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
                            "name": tool_name,
                            "arguments": str(input_data)
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": "success: "+str(success)+" output: "+str(output)
            }
        ]
        
        # 发送到 mem0（使用 procedural_memory 类型，保持 run_id 参数）
        memory.add(messages, agent_id=tool_name, user_id=source_task, run_id='apibank-ds', memory_type="procedural_memory")
        return True
    except Exception as e:
        print(f"❌ 记录发送到 mem0 失败 (tool: {tool_call_record.get('tool_name', 'unknown')}, task: {tool_call_record.get('source_task', 'unknown')}): {e}")
        traceback.print_exc()
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 API-Bank 的 jsonl 结果文件中提取工具调用结果，并打印 POST payload（不发送）。"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入 API-Bank 结果 jsonl 文件路径（每行一个 JSON，对应一个样本）",
    )
    parser.add_argument(
        "--workspace-id",
        default="deepseek-v3",
        help="POST body 中使用的 workspace_id（与向量库中的 workspace_id 无需一致）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="每个 POST body 中包含的工具调用数量",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8002/add_tool_call_result",
        help="接收工具调用结果的 HTTP endpoint",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP 请求超时时间（秒）",
    )
    parser.add_argument(
        "--gt",
        action="store_true",
        help="如果启用，将 ground_truth 拼接到 output 后面",
    )
    parser.add_argument(
        "--right",
        action="store_true",
        help="如果启用，并且该样本的 ground_truth 为 API，会将 ground_truth 也当作一次工具调用单独提取并发送",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mem",
        choices=["mem", "m0"],
        help="选择发送方式：'mem' 表示发送到 HTTP 服务（默认），'m0' 表示发送到 mem0",
    )
    return parser.parse_args()


def extract_tool_calls_from_line(
    record: Dict[str, Any],
    include_gt: bool = False,
    include_right: bool = False,
) -> List[Dict[str, Any]]:
    """
    从一条 API-Bank 的 jsonl 记录中抽取「一次工具调用」的信息。

    工具名称和输入从 model_output_raw 中提取，输出从 tool_result 中提取。

    主要使用的字段：
        - file:       场景文件名，用于构造 source_task
        - sample_id:  样本编号
        - chat_history / messages: 用来构造上下文（context）
        - model_output_raw: 模型原始输出，包含工具名称和输入参数
        - tool_result: 工具调用的结果，包含 output 或 exception
        - correct:    API-Bank 评测的正确性，我们可以选择性地记录在 context 中
    """
    # 构造 source_task：用 file + sample_id，方便追溯到原始样本
    file_name = record.get("file", "")
    chat_history = record.get("chat_history") or []
    type = record.get("type", "")
    context = chat_history
    
    results: List[Dict[str, Any]] = []

    # 目前只处理 type == "API" 的记录
    if type != "API":
        return results


    # ===== 1. 如果启用 right，加一条 ground_truth 调用（如果存在且为 API）=====
    if include_right:
        gt = record.get("ground_truth") or {}
        if isinstance(gt, dict) and gt.get("role") == "API":
            gt_api_name = gt.get("api_name")
            gt_param_dict = gt.get("param_dict") or {}
            gt_result = gt.get("result")
            gt_output = extract_tool_output(gt_result)

            gt_success = True
            if isinstance(gt_result, dict):
                gt_exception = gt_result.get("exception", None)
                gt_success = gt_exception is None

            results.append(
                {
                    "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tool_name": gt_api_name,
                    "context": str(context),
                    "config": "/Users/xiaocan/Toolmem/experiment/api-bank/config.json",
                    "input": gt_param_dict,
                    "output": gt_output,
                    "token_cost": 0,
                    "success": gt_success,
                    "time_cost": 0.0,
                    "source_task": file_name,
                }
            )

    # ===== 2. 基于模型实际调用提取（始终保留这一条）=====
    # 从 model_output_raw 中提取工具名称和输入（模型实际调用）
    model_output_raw = record.get("model_output_raw", "")
    api_name, tool_input = parse_model_output_raw(model_output_raw)


    if not api_name:
        # 如果无法从 model_output_raw 解析，模型这一条调用直接跳过
        return results
    
    # 从 tool_result 中提取输出
    tool_result = record.get("tool_result")
    final_output = extract_tool_output(tool_result)
    
    # 判断 success：如果 tool_result 是字典且有 exception，则为失败
    if isinstance(tool_result, dict):
        exception = tool_result.get("exception", None)
        success = exception is None
    else:
        # 如果 tool_result 是字符串（通常是错误信息），视为失败
        success = False
    
    # 如果启用 gt 参数，将 ground_truth 拼接到 output 后面
    if include_gt:
        ground_truth = record.get("ground_truth")
        correct = record.get("correct")
        if not correct and ground_truth:
            gt_str = json.dumps(ground_truth, ensure_ascii=False)
            final_output = f"{final_output}\n Compare this tool callwith ground truth, is judged as {correct}, and the groundtruth is as below for reference: {gt_str}"

    # 加入“模型实际调用”的这一次
    results.append(
        {
            # 时间：用当前时间
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool_name": api_name,
            "context": str(context),
            "config": "/Users/xiaocan/Toolmem/experiment/api-bank/config.json",  # API-Bank 这里没有单独的 config，先留空
            "input": tool_input if tool_input else {},
            "output": final_output,
            "token_cost": 0,
            "success": success,
            "time_cost": 0.0,
            "source_task": file_name,
        }
    )


    return results


def extract_from_jsonl_file(
    input_path: Path,
    workspace_id: str,
    batch_size: int,
    include_gt: bool = False,
    include_right: bool = False,
) -> List[Dict[str, Any]]:
    """从 jsonl 文件中读取所有工具调用结果，并按 batch_size 分批组装 POST payload。"""
    all_results: List[Dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"跳过第 {line_no} 行，JSON 解析失败: {e}")
                continue
            tool_calls = extract_tool_calls_from_line(
                record,
                include_gt=include_gt,
                include_right=include_right,
            )
            if tool_calls:
                all_results.extend(tool_calls)

    if not all_results:
        print("没有在 jsonl 中发现任何 tool_call_results。")
        return []

    batches: List[Dict[str, Any]] = []
    total_tools = len(all_results)
    total_batches = (total_tools + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_tools)
        batch = all_results[start_idx:end_idx]

        batches.append(
            {
                "workspace_id": workspace_id,
                "tool_call_results": batch,
            }
        )

    return batches


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 提取所有工具调用记录
    all_results: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"跳过第 {line_no} 行，JSON 解析失败: {e}")
                continue
            tool_calls = extract_tool_calls_from_line(
                record,
                include_gt=args.gt,
                include_right=args.right,
            )
            if tool_calls:
                all_results.extend(tool_calls)

    if not all_results:
        print("没有在 jsonl 中发现任何 tool_call_results。")
        return

    # 根据 method 选择发送方式
    if args.method == "m0":
        # 发送到 mem0
        if not MEM0_AVAILABLE:
            print("错误: mem0 未安装，无法使用 --method m0。请安装: pip install mem0ai")
            return
        
        print(f"使用 mem0 发送 {len(all_results)} 条工具调用记录...")
        success_count = 0
        fail_count = 0
        
        for idx, tool_call_record in tqdm.tqdm(enumerate(all_results, start=1)):
            if send_tool_call_to_mem0(tool_call_record):
                success_count += 1
                if idx % 10 == 0:
                    print(f"  已处理 {idx}/{len(all_results)} 条...")
            else:
                fail_count += 1
        
        print(f"✅ mem0 发送完成: 成功 {success_count} 条, 失败 {fail_count} 条")
    
    else:
        # 发送到 HTTP 服务（mem）
        batches = []
        total_tools = len(all_results)
        total_batches = (total_tools + args.batch_size - 1) // args.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, total_tools)
            batch = all_results[start_idx:end_idx]

            batches.append(
                {
                    "workspace_id": args.workspace_id,
                    "tool_call_results": batch,
                }
            )
        
        print(f"使用 HTTP 服务发送 {len(all_results)} 条工具调用记录，分 {len(batches)} 个批次...")
        
        for idx, payload in enumerate(batches, start=1):
            try:
                resp = requests.post(args.endpoint, json=payload, timeout=args.timeout)
                if resp.status_code != 200:
                    print(f"❌ 批次 {idx} 请求失败，状态码: {resp.status_code}, 响应: {resp.text}")
                    continue
                body = {}
                try:
                    body = resp.json()
                except Exception:
                    pass
                if not body or not body.get("success", False):
                    print(f"❌ 批次 {idx} 接口返回失败，工具数量: {len(payload.get('tool_call_results', []))}, 响应: {body or resp.text}")
                    continue
                print(f"✅ 批次 {idx} 已发送，工具数量: {len(payload.get('tool_call_results', []))}")
            except Exception as e:
                print(f"❌ 批次 {idx} 发送异常: {e}")


if __name__ == "__main__":
    main()


