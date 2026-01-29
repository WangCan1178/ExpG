"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import os
import sys
import time
import re

# 添加当前目录到路径，以便可以直接运行此脚本
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import requests
from fastchat.model import load_model, get_conversation_template, add_model_args
import logging
import argparse
import json
from tqdm import tqdm
from utils import *
from dotenv import load_dotenv
import openai
import traceback
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenAI 客户端相关导入（用于调用本地 vLLM 服务）
try:
    from openai import OpenAI
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False
    print("警告: OpenAI 客户端未安装，--method toolllama/qwenft/llama 选项将不可用。安装命令: pip install openai")

from tool_memory import ToolMemoryFetcher
from extract import send_single_sample

# mem0 相关导入（可选，如果未安装会优雅降级）
try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    print("警告: mem0 未安装，--method m0 选项将不可用。安装命令: pip install mem0ai")

# os.environ['OPENAI_API_KEY'] = '<your api key>'
# openai.api_key = '<your api key>'


load_dotenv()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 是否使用 Qwen 远程推理
IS_QWEN = False
QWEN_MODEL_ID: Optional[str] = None
# 是否使用 GPT 远程推理
IS_GPT = False
GPT_MODEL_ID: Optional[str] = None
N_SAMPLE = 3  # 每次推理采样3次

TOOL_LINE_RE = re.compile(
    r"^(\d+\.\s*tool name:\s*)([^,]+)(,\s*tool description:\s*)(.*)$",
    re.IGNORECASE,
)

# draft 工具文档路径（由 MetaTool 根目录下的 draft_tool_docs.py 生成）
DRAFT_DESCRIPTION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset",
    "plugin_des_draft.json",
)

# 全局 memory 实例（延迟初始化）
_mem0_memory_instance = None
_draft_descriptions_cache: Optional[Dict[str, str]] = None

def get_mem0_memory():
    """
    获取全局的 mem0 Memory 实例（单例模式）。
    如果尚未初始化，则进行初始化。
    
    Returns:
        Memory 实例，如果初始化失败则返回 None
    """
    global _mem0_memory_instance
    
    if _mem0_memory_instance is not None:
        return _mem0_memory_instance
    
    if not MEM0_AVAILABLE:
        return None
    
    try:
        config = get_mem0_config()
        _mem0_memory_instance = Memory.from_config(config)
        return _mem0_memory_instance
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
                "api_key": "your api key",
                "model": "text-embedding-v3",
                "embedding_dims": 512,
                "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": "your api key",
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


def _load_draft_descriptions() -> Dict[str, str]:
    """
    延迟加载 draft 版工具描述（由 draft_tool_docs.py 生成）。
    如果文件不存在，则返回空 dict，并在控制台给出提示。
    """
    global _draft_descriptions_cache
    if _draft_descriptions_cache is not None:
        return _draft_descriptions_cache

    if not os.path.exists(DRAFT_DESCRIPTION_PATH):
        print(
            f"警告: 未找到 draft 工具描述文件 {DRAFT_DESCRIPTION_PATH}，"
            "在 --method draft 下将退回原始工具文档。"
        )
        _draft_descriptions_cache = {}
        return _draft_descriptions_cache

    try:
        with open(DRAFT_DESCRIPTION_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(
                f"警告: draft 描述文件 {DRAFT_DESCRIPTION_PATH} 内容应为 dict，"
                f"实际类型为 {type(data)}，将忽略。"
            )
            data = {}
    except Exception as e:
        print(f"警告: 读取 draft 描述文件 {DRAFT_DESCRIPTION_PATH} 失败: {e}")
        data = {}

    _draft_descriptions_cache = data  # type: ignore[assignment]
    return _draft_descriptions_cache


def get_mem0_memories_for_tools(
    tool_names: list,
    current_task_id: str,
    max_total_chars: int = 120000,
) -> Dict[str, str]:
    """
    从 mem0 获取指定工具的记忆，排除当前 task 的记录。
    
    Args:
        tool_names: 工具名称列表
        current_task_id: 当前任务的 ID（用于排除）
        
    Returns:
        Dict[str, str]: 工具名称到拼接后的 memory 文本的映射
    """
    memory_map: Dict[str, str] = {name: "" for name in tool_names}
    
    memory = get_mem0_memory()
    if memory is None:
        return memory_map
    
    # 计算每个工具可用的最大字符数（平均分配）
    tool_count = len(tool_names) if tool_names else 1
    per_tool_limit = max_total_chars // tool_count if tool_count > 0 else max_total_chars
    if per_tool_limit <= 0:
        per_tool_limit = max_total_chars

    for tool_name in tool_names:
        try:
            # 获取该工具的所有记忆
            all_memories_result = memory.get_all(agent_id=tool_name)
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
            
            # 将多个 memory 拼接起来，用空格分隔，并按 per_tool_limit 截断
            if memory_texts:
                full_text = (
                    "Besides, this is optional guidance on how to better use the tool. You may refer to it selectively "
                    + " ".join(memory_texts)
                )
                if len(full_text) > per_tool_limit:
                    # 截断到限定长度，避免过长
                    truncated = full_text[:per_tool_limit]
                    # 尽量避免截在非常奇怪的位置，简单回退到上一个空格
                    last_space = truncated.rfind(" ")
                    if last_space > 0:
                        truncated = truncated[:last_space]
                    truncated += " ..."
                    full_text = truncated
                memory_map[tool_name] = full_text
        except Exception as e:
            print(f"警告: 获取工具 {tool_name} 的 mem0 记忆失败: {e}")
            continue
    
    return memory_map


def _rewrite_action_prompt_with_draft(action_prompt: str) -> str:
    """
    在 --method draft 下，对 action_prompt 中的工具描述进行替换：
    - 识别形如 `1. tool name: X, tool description: Y` 的行
    - 若 draft 描述中存在对应的 X，则用 draft 描述替换 Y
    - 其它行保持不变
    """
    draft_des = _load_draft_descriptions()
    if not draft_des:
        return action_prompt

    lines = action_prompt.splitlines()
    new_lines = []

    for line in lines:
        # 保留前导空白
        leading_ws_match = re.match(r"(\s*)(.*)", line)
        leading_ws = leading_ws_match.group(1) if leading_ws_match else ""
        content = leading_ws_match.group(2) if leading_ws_match else line

        m = TOOL_LINE_RE.match(content.strip())
        if not m:
            new_lines.append(line)
            continue

        tool_name = m.group(2).strip()
        new_desc = draft_des.get(tool_name)
        if not new_desc:
            new_lines.append(line)
            continue

        # m.group(1): 前缀 "1. tool name: "
        # m.group(3): 中间 ", tool description: "
        new_content = f"{m.group(1)}{tool_name}{m.group(3)}{new_desc}"
        new_lines.append(f"{leading_ws}{new_content}")

    return "\n".join(new_lines)

def extract_user_query_from_context(context: str) -> str:
    """
    从 context 中提取 [User's Query Start] 和 [User's Query End] 之间的内容。
    
    Args:
        context: 包含用户查询的完整 context 字符串
        
    Returns:
        str: 提取出的用户查询内容，如果找不到标记则返回空字符串
    """
    if not context:
        return ""
    
    start_marker = "[User's Query Start]"
    end_marker = "[User's Query End]"
    
    start_idx = context.find(start_marker)
    if start_idx == -1:
        return ""
    
    start_idx += len(start_marker)
    end_idx = context.find(end_marker, start_idx)
    if end_idx == -1:
        return ""
    
    query = context[start_idx:end_idx].strip()
    return query

def send_single_sample_to_mem0(
    item: Dict[str, Any],
    model_name: Optional[str] = None,
) -> bool:
    """
    将单个样本发送到 mem0 进行记忆存储。
    参考 extract.py 中的 send_to_mem0 函数。
    
    Args:
        item: 包含 id, action_res, tool, action_prompt 等字段的样本字典
        model_name: 模型名称，用于读取原始 action_prompt（如果需要）
    
    Returns:
        bool: 发送是否成功
    """
    memory = get_mem0_memory()
    if memory is None:
        print("错误: mem0 未安装或初始化失败，无法发送数据到 mem0")
        return False
    
    try:
        # 构造工具调用记录（复用 extract.py 中的逻辑）
        from extract import (
            get_subtask_id,
            normalize_tool_field,
            extract_used_tools_from_action_res,
            SampleInfo,
            compute_sample_success,
            build_tool_call_records,
        )
        
        sid = str(item.get("id", ""))
        st = get_subtask_id(item)
        if st is None:
            st = 1
        
        gt_set, gt_raw = normalize_tool_field(item.get("tool"))
        if st == 3:
            gt_set = set()
            gt_raw = []
        
        used_set, used_raw = extract_used_tools_from_action_res(item.get("action_res"))
        
        sample = SampleInfo(
            raw=item,
            sample_id=sid,
            subtask=st,
            used_tools=used_set,
            gt_tools=gt_set,
            used_tool_names_raw=used_raw,
            gt_tool_names_raw=gt_raw,
        )
        compute_sample_success(sample)
        
        # 构造 tool_call_records
        records = build_tool_call_records([sample], model_name=model_name)
        
        if not records:
            print(f"样本 {sid} 没有需要发送的工具记录。")
            return False
        
        # 发送每条记录到 mem0
        success_count = 0
        for record in records:
            try:
                tool_name = record.get("tool_name", "")
                context_full = record.get("context", "")
                input_data = record.get("input", {})
                output = record.get("output", "")
                success = record.get("success", False)
                token_cost = record.get("token_cost", 0)
                time_cost = record.get("time_cost", 0.0)
                source_task = record.get("source_task", "")
                
                # 从 context 中提取用户查询
                user_query = extract_user_query_from_context(context_full)
                if not user_query:
                    user_query = context_full
                
                # 构建 call_info
                call_info = {
                    'success': success,
                    'token_cost': token_cost,
                    'time_cost': time_cost
                }
                
                # 构建消息（参考 test.py 的格式）
                message_content = (
                    f"There is an experience that needs to be remembered: when using the {tool_name} tool, "
                    f"the context is {user_query}, input is {input_data}, and tool response is {output}, "
                    f"with call info are {call_info}. "
                    f"Remember this rule so that you can improve the next time you use this tool."
                )
                
                messages = [
                    {"role": "user", "content": message_content}
                ]
                
                # 发送到 mem0
                memory.add(messages, agent_id=tool_name, user_id=source_task)
                success_count += 1
            except Exception as e:
                print(f"❌ 记录发送到 mem0 失败 (tool: {record.get('tool_name', 'unknown')}, task: {record.get('source_task', 'unknown')}): {e}")
        
        if success_count > 0:
            print(f"✅ 样本 {sid} 已发送到 mem0，成功 {success_count}/{len(records)} 条")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ 样本 {item.get('id', 'unknown')} 发送到 mem0 异常: {e}")
        traceback.print_exc()
        return False

def get_best_device():
    """
    检测并返回最佳可用设备。
    优先级：MPS (Mac) > CUDA > CPU
    """
    # 优先使用 MPS (Mac Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    # 其次使用 CUDA
    elif torch.cuda.is_available():
        return torch.device('cuda')
    # 最后使用 CPU
    else:
        return torch.device('cpu')

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

def _should_retry_vllm(exception):
    """判断 vLLM 请求是否应该重试该异常"""
    error_msg = str(exception).lower()
    # 检查是否是速率限制相关的错误消息
    rate_limit_keywords = ['rate limit', 'rate_limit', '429', 'too many requests', 'quota']
    if any(keyword in error_msg for keyword in rate_limit_keywords):
        return True
    # 检查是否是临时性错误
    recoverable_keywords = ['timeout', 'connection', 'server error', '503', '502', '500', 'connection refused']
    if any(keyword in error_msg for keyword in recoverable_keywords):
        return True
    # OpenAI 客户端异常
    if 'rate limit' in error_msg or 'timeout' in error_msg:
        return True
    return False

def get_res_vllm(string: str, model_name: str, base_url: str = "http://127.0.0.1:57723/v1", temperature: float = 0.5, max_retries: int = 3) -> str:
    """
    使用本地 vLLM 服务进行推理（通过 OpenAI 兼容接口）。
    用于调用经过微调的模型（toolllama、qwenft、llama）。
    支持自动重试和指数退避。
    
    Args:
        string: 输入的 prompt
        model_name: 模型名称（例如 "toolqwen3"）
        base_url: vLLM 服务的 base URL，默认为 "http://127.0.0.1:57723/v1"
        temperature: 生成温度，默认为 0.5
        max_retries: 最大重试次数
        
    Returns:
        str: 模型的最终回答（已提取思考标签后的内容）
    """
    if not OPENAI_CLIENT_AVAILABLE:
        return "[ERROR] OpenAI 客户端未安装，无法调用 vLLM 服务。安装命令: pip install openai"
    
    # 打印最终发送给 vLLM 的 prompt
    print("===== MODEL PROMPT (vLLM) START =====")
    print(string)
    print("===== MODEL PROMPT (vLLM) END =====")
    
    # 创建 OpenAI 客户端
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY"  # vLLM 默认不会校验，用随便字符串即可
    )
    
    # 重试逻辑（指数退避）
    max_attempts = max_retries + 1  # 初始尝试 + 重试次数
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            resp = client.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": string}
                ],
                temperature=temperature,
                extra_body={
                    "enable_thinking": False
                }
            )
            
            # 获取完整内容
            full_content = resp.choices[0].message.content
            
            # 提取最终回答（移除思考标签）
            final_response = extract_final_response(full_content)
            
            print(f"完整返回内容: {full_content[:200]}...")
            print(f"提取的最终回答: {final_response[:200]}...")
            
            return final_response
            
        except Exception as e:
            last_error = e
            # 检查是否应该重试
            if _should_retry_vllm(e) and attempt < max_attempts - 1:
                # 计算等待时间（指数退避）
                wait_time = min(2 ** attempt, 60)  # 最多等待60秒
                error_msg = f"[ERROR] vLLM 请求失败（尝试 {attempt + 1}/{max_attempts}），{wait_time}秒后重试: {type(e).__name__}: {str(e)}"
                print(error_msg)
                time.sleep(wait_time)
            else:
                # 不可恢复的错误或已达到最大重试次数
                if attempt < max_attempts - 1:
                    error_msg = f"[ERROR] vLLM 请求失败，不可恢复: {type(e).__name__}: {str(e)}"
                else:
                    error_msg = f"[ERROR] vLLM 请求失败，已达到最大重试次数（{max_attempts}）: {type(e).__name__}: {str(e)}"
                print(error_msg)
                return error_msg
    
    # 如果所有重试都失败，返回最后的错误信息
    return f"[ERROR] vLLM 请求失败: {str(last_error)}" if last_error else "[ERROR] 未知错误"


def get_res_vllm_batch(
    strings: List[str],
    model_name: str,
    base_url: str = "http://127.0.0.1:57723/v1",
    temperature: float = 0.5,
    max_retries: int = 3,
    max_workers: int = 8
) -> List[str]:
    """
    使用本地 vLLM 服务进行批量推理（通过 OpenAI 兼容接口）。
    用于批量调用经过微调的模型（toolllama、qwenft、llama）。
    支持并行请求和自动重试。
    
    Args:
        strings: 输入的 prompt 列表
        model_name: 模型名称（例如 "toolqwen3"）
        base_url: vLLM 服务的 base URL，默认为 "http://127.0.0.1:57723/v1"
        temperature: 生成温度，默认为 0.5
        max_retries: 最大重试次数
        max_workers: 最大并发线程数，默认为 8
        
    Returns:
        List[str]: 每个请求的模型回答列表（已提取思考标签后的内容）
    """
    if not OPENAI_CLIENT_AVAILABLE:
        return ["[ERROR] OpenAI 客户端未安装，无法调用 vLLM 服务。安装命令: pip install openai"] * len(strings)
    
    if not strings:
        return []
    
    print(f"===== 批量推理开始，共 {len(strings)} 个请求 =====")
    
    # 使用线程池进行并发请求
    results = [None] * len(strings)
    
    def process_single(index: int, string: str) -> Tuple[int, str]:
        """处理单个请求的辅助函数（内部已包含重试机制）"""
        try:
            result = get_res_vllm(
                string=string,
                model_name=model_name,
                base_url=base_url,
                temperature=temperature,
                max_retries=max_retries
            )
            return index, result
        except Exception as e:
            error_msg = f"[ERROR] 批量推理中单个请求失败: {type(e).__name__}: {str(e)}"
            logging.error(f"  请求 {index + 1} 失败: {error_msg}")
            return index, error_msg
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, idx, s): idx
            for idx, s in enumerate(strings)
        }
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(strings), desc="批量推理进度", ncols=100) as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                pbar.update(1)
                if result and result.startswith("[ERROR]"):
                    pbar.set_postfix({"错误": 1})
    
    print(f"===== 批量推理完成 =====")
    return results


def _augment_action_prompt_with_memory(action_prompt: str, source_task: str) -> str:
    """
    根据工具名称获取 memory，将返回的 memory 追加到对应工具描述后面。
    - 如果 method == "mem": 调用 ToolMemory 服务
    - 如果 method == "m0": 从 mem0 获取记忆（排除当前 task 的记录）
    - source_task 使用当前样本的 id
    """
    method = getattr(args, "method", None)
    # 未开启 memory 功能直接返回
    if method not in ("mem", "m0"):
        return action_prompt

    lines = action_prompt.splitlines()

    # 1. 从工具列表中解析出所有 tool name
    tool_names = []
    for line in lines:
        stripped = line.strip()
        m = TOOL_LINE_RE.match(stripped)
        if m:
            tool_name = m.group(2).strip()
            if tool_name:
                tool_names.append(tool_name)

    if not tool_names:
        return action_prompt

    # 2. 获取 memory（根据 method 选择不同的方法）
    memory_map: Dict[str, str] = {name: "" for name in tool_names}
    
    if method == "mem":
        # 使用 ToolMemory 服务
        fetcher = ToolMemoryFetcher(
            workspace_id="toole-ds",
            source_task=str(source_task),
            model_name=str(getattr(args, "model_path", "")),
        )
        fetched = fetcher.collect_memory(tool_names)

        # 使用获取到的 memory（fetched 现在返回 Dict[str, Dict[str, str]]，包含 content 和 add_to）
        for name in tool_names:
            tool_memory = (fetched or {}).get(name, {})
            if isinstance(tool_memory, dict):
                value = tool_memory.get("content", "") or ""
            else:
                # 兼容旧格式（字符串）
                value = tool_memory or ""
            memory_map[name] = value
    elif method == "m0":
        # 从 mem0 获取记忆（排除当前 task 的记录）
        memory_map = get_mem0_memories_for_tools(tool_names, source_task)

    # 如果全部为空，直接返回原始 prompt
    if not any(memory_map.values()):
        return action_prompt

    # 3. 将 memory 拼接到每一行对应的 tool description 后面
    new_lines = []
    for line in lines:
        leading_ws_match = re.match(r"(\s*)(.*)", line)
        leading_ws = leading_ws_match.group(1) if leading_ws_match else ""
        content = leading_ws_match.group(2) if leading_ws_match else line
    
        m = TOOL_LINE_RE.match(content.strip())
        if not m:
            new_lines.append(line)
            continue
    
        tool_name = m.group(2).strip()
        memory_text = memory_map.get(tool_name, "") or ""
        if not memory_text:
            new_lines.append(line)
            continue
    
        original_desc = m.group(4).rstrip()
        augmented_desc = f"{original_desc} {memory_text}"
        new_content = f"{m.group(1)}{tool_name}{m.group(3)}{augmented_desc}"
        new_lines.append(f"{leading_ws}{new_content}")
    
    return "\n".join(new_lines)


def generation(input,tokenizer,model):
    msg = input
    conv = get_conversation_template(args.model_path)
    conv.set_system_message('')
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # 打印最终发送给本地/HF 模型的 prompt
    print("===== MODEL PROMPT START =====")
    print(prompt)
    print("===== MODEL PROMPT END =====")
    inputs = tokenizer([prompt])
    
    # 检测最佳设备（优先使用 MPS for Mac, 然后是 CUDA, 最后是 CPU）
    device = get_best_device()
    print(f"使用设备: {device}")
    
    # 确保模型在正确的设备上
    # 检查模型当前设备，如果不在目标设备上，则移动
    try:
        model_device = next(model.parameters()).device
        if model_device != device:
            print(f"将模型从 {model_device} 移动到 {device}")
            model = model.to(device)
    except Exception as e:
        print(f"警告: 无法检测模型设备，尝试直接移动到 {device}: {e}")
        try:
            model = model.to(device)
        except Exception as e2:
            print(f"错误: 无法将模型移动到 {device}: {e2}")
            # 如果移动失败，尝试使用 CPU
            if device.type != 'cpu':
                print("回退到 CPU")
                device = torch.device('cpu')
                model = model.to(device)
    
    inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
    
    # 尝试生成，如果 MPS 出错则回退到 CPU
    try:
        output_ids = model.generate(
            **inputs,
            do_sample=True if args.temperature > 1e-5 else False,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )
    except (RuntimeError, KeyError) as e:
        # 如果是 MPS 相关的错误，回退到 CPU
        if device.type == 'mps' and ('lm_head' in str(e) or 'weight' in str(e) or 'MPS' in str(e)):
            print(f"MPS 生成失败 ({e})，回退到 CPU")
            device = torch.device('cpu')
            model = model.to(device)
            # 重新将 inputs 移动到 CPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_ids = model.generate(
                **inputs,
                do_sample=True if args.temperature > 1e-5 else False,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            # 其他错误直接抛出
            raise

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs



def tool_test_thought(model_name, model, tokenizer):
    model_name = model_name.lower()
    output_dir = os.path.join('tool/new_test_res', model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析指定的样本ID列表
    sample_ids_set = None
    if hasattr(args, 'sample_ids') and args.sample_ids:
        sample_ids_set = set([sid.strip() for sid in args.sample_ids.split(',') if sid.strip()])
        print(f"指定运行样本ID: {sample_ids_set}")
    
    file_list = os.listdir('tool/test_data')

    for file in file_list:
        if file.endswith('.json'):
            if os.path.isfile(os.path.join('tool/test_data', file)):
                # 生成3个输出文件名
                base_name = os.path.splitext(file)[0]
                ext = os.path.splitext(file)[1]
                # 根据 method 添加后缀
                method = getattr(args, "method", None)
                if method:
                    suffix = f"_{method}"
                    output_files = [os.path.join(output_dir, f"{base_name}_{i}{suffix}{ext}") for i in range(N_SAMPLE)]
                else:
                    output_files = [os.path.join(output_dir, f"{base_name}_{i}{ext}") for i in range(N_SAMPLE)]
                
                # 读取已存在的结果（如果有）
                existing_results_list = [{} for _ in range(N_SAMPLE)]
                for sample_idx, output_file in enumerate(output_files):
                    if os.path.exists(output_file):
                        try:
                            with open(output_file, 'r') as f:
                                existing_data = json.load(f)
                                # 创建索引：使用 id 作为唯一标识符
                                for item in existing_data:
                                    if 'id' in item:
                                        key = item['id']
                                        existing_results_list[sample_idx][key] = item
                        except:
                            existing_results_list[sample_idx] = {}
                
                # 读取测试数据
                with open(os.path.join('tool/test_data', file), 'r') as f:
                    data = json.load(f)
                
                # 加载已保存的结果
                all_data_list = []
                for sample_idx in range(N_SAMPLE):
                    all_data_list.append(list(existing_results_list[sample_idx].values()) if existing_results_list[sample_idx] else [])
                
                processed_count = len(all_data_list[0]) if all_data_list[0] else 0
                
                for idx, el in enumerate(data):
                    # 如果指定了样本ID，检查当前样本是否在列表中
                    if sample_ids_set is not None:
                        sample_id = el.get('id', '')
                        if sample_id not in sample_ids_set:
                            print(f"跳过未指定的样本 {idx + 1}/{len(data)}: id={sample_id}")
                            continue
                    
                    # 检查任务ID
                    key = el.get('id', '') if 'id' in el else None
                    if not key:
                        print(f"警告: 任务 {idx + 1} 没有 id，跳过")
                        continue
                    
                    # 检查每个采样是否已经完成
                    results = []
                    need_process = False
                    for sample_idx in range(N_SAMPLE):
                        if key in existing_results_list[sample_idx]:
                            # 该采样已完成，使用已有结果
                            existing_item = existing_results_list[sample_idx][key]
                            res = existing_item.get('res', '')
                            results.append(res)
                            print(f"  采样 {sample_idx + 1}/{N_SAMPLE} 已存在，跳过")
                        else:
                            # 该采样未完成，需要处理
                            need_process = True
                            results.append(None)  # 占位符
                    
                    # 如果所有采样都已完成，跳过该任务
                    if not need_process:
                        print(f"跳过已完全处理的任务 {idx + 1}/{len(data)}: id={key}")
                        continue
                    
                    print(f"处理任务 {idx + 1}/{len(data)} (已处理: {processed_count}, id={key})")
                    
                method = getattr(args, "method", None)
                # mem 方法不使用并行
                use_batch_inference = method in ("toolllama", "qwenft", "llama")
                use_qwen_batch = IS_QWEN and method != "mem"  # Qwen 模型也支持批量推理，但 mem 不使用并行
                use_gpt_batch = IS_GPT and method != "mem"  # GPT 模型也支持批量推理，但 mem 不使用并行
                
                # 对于 vLLM 方法、Qwen/GPT 模型，收集所有未完成的采样进行批量推理
                # 但 mem 方法不使用并行
                if (use_batch_inference or use_qwen_batch or use_gpt_batch) and method != "mem":
                    batch_prompts = []
                    batch_indices = []
                    for sample_idx in range(N_SAMPLE):
                        if results[sample_idx] is None:
                            batch_prompts.append(el['thought_prompt'])  # 相同的 prompt，但会被处理多次以支持采样
                            batch_indices.append(sample_idx)
                    
                    if batch_prompts:
                        print(f"  开始批量推理，共 {len(batch_prompts)} 个采样...")
                        
                        if use_batch_inference:
                            # vLLM 批量推理
                            model_name_map = {
                                "toolllama": "toolllama",
                                "qwenft": "toolqwen3",
                                "llama": "toolllama"
                            }
                            vllm_model_name = model_name_map.get(method, "toolqwen3")
                            temperature = getattr(args, 'temperature', 0.5)
                            
                            batch_results = get_res_vllm_batch(
                                strings=batch_prompts,
                                model_name=vllm_model_name,
                                temperature=temperature
                            )
                        elif use_gpt_batch:
                            # GPT 批量推理
                            batch_results = get_res_gpt_batch(
                                strings=batch_prompts,
                                gpt_model=GPT_MODEL_ID,
                                max_retries=3,
                                timeout=60
                            )
                        elif use_qwen_batch:
                            # Qwen 批量推理
                            batch_results = get_res_qwen_batch(
                                strings=batch_prompts,
                                qwen_model=QWEN_MODEL_ID,
                                max_retries=3,
                                timeout=60
                            )
                        else:
                            batch_results = []
                        
                        # 将结果填入 results
                        for i, sample_idx in enumerate(batch_indices):
                            results[sample_idx] = batch_results[i] if i < len(batch_results) else "[ERROR] 批量推理结果不足"
                            print(f"  结果 {sample_idx + 1}: {results[sample_idx][:100]}...")
                else:
                    # 非批量推理方法，保持原有逻辑
                    # 对每个未完成的采样进行推理
                    for sample_idx in range(N_SAMPLE):
                        # 如果该采样已有结果，跳过
                        if results[sample_idx] is not None:
                            continue
                        
                        try:
                            print(f"  采样 {sample_idx + 1}/{N_SAMPLE}...")
                            if IS_GPT:
                                # GPT 远程推理（单个请求，内部有重试机制）
                                res = get_res_gpt(string=el['thought_prompt'], gpt_model=GPT_MODEL_ID)
                            elif IS_QWEN:
                                # Qwen 远程推理（单个请求，内部有重试机制）
                                res = get_res_qwen(string=el['thought_prompt'], qwen_model=QWEN_MODEL_ID)
                            else:
                                # 本地/HF 模型推理
                                res = generation(model=model, tokenizer=tokenizer, input=el['thought_prompt'])
                            print(f"  结果 {sample_idx + 1}: {res[:100]}...")
                            results[sample_idx] = res
                        except Exception as e:
                            error_msg = f"[ERROR] 推理失败: {str(e)}"
                            print(f"  错误 {sample_idx + 1}: {error_msg}")
                            results[sample_idx] = error_msg
                    

                    # 将结果分别保存到对应的文件（只保存新完成的采样）
                    for sample_idx in range(N_SAMPLE):
                        # 如果该采样之前已存在，跳过保存（避免重复）
                        if key in existing_results_list[sample_idx]:
                            continue
                        
                        el_copy = el.copy()
                        el_copy['res'] = results[sample_idx]
                        all_data_list[sample_idx].append(el_copy)
                        # 每处理一条就保存一次
                        save_json(all_data_list[sample_idx], output_files[sample_idx])
                    
                    processed_count += 1


def tool_test_action(model_name, model, tokenizer):
    model_name = model_name.lower()
    output_dir = os.path.join('tool/new_test_res', model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析指定的样本ID列表
    sample_ids_set = None
    if hasattr(args, 'sample_ids') and args.sample_ids:
        sample_ids_set = set([sid.strip() for sid in args.sample_ids.split(',') if sid.strip()])
        print(f"指定运行样本ID: {sample_ids_set}")
    
    file_list = os.listdir('tool/test_data/{}'.format(model_name))
    
    for file in file_list:
        #if file.endswith('general_test.json'):
        if 1:
            # 生成3个输出文件名
            base_name = os.path.splitext(file)[0]
            ext = os.path.splitext(file)[1]
            # 根据 method 添加后缀
            method = getattr(args, "method", None)
            if method:
                suffix = f"_{method}"
                output_files = [os.path.join(output_dir, f"{base_name}_{i}{suffix}{ext}") for i in range(N_SAMPLE)]
            else:
                output_files = [os.path.join(output_dir, f"{base_name}_{i}{ext}") for i in range(N_SAMPLE)]
            
            # 读取已存在的结果（如果有）
            existing_results_list = [{} for _ in range(N_SAMPLE)]
            for sample_idx, output_file in enumerate(output_files):
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r') as f:
                            existing_data = json.load(f)
                            # 创建索引：使用 id 作为唯一标识符
                            for item in existing_data:
                                if 'id' in item:
                                    key = item['id']
                                    existing_results_list[sample_idx][key] = item
                    except:
                        existing_results_list[sample_idx] = {}
            
            # 读取测试数据
            with open(os.path.join('tool/test_data', model_name, file), 'r') as f:
                data = json.load(f)
            
            # 加载已保存的结果
            all_data_list = []
            for sample_idx in range(N_SAMPLE):
                all_data_list.append(list(existing_results_list[sample_idx].values()) if existing_results_list[sample_idx] else [])
            
            processed_count = len(all_data_list[0]) if all_data_list[0] else 0
            
            for idx, el in enumerate(data):
                # 如果指定了样本ID，检查当前样本是否在列表中
                if sample_ids_set is not None:
                    sample_id = el.get('id', '')
                    if sample_id not in sample_ids_set:
                        print(f"跳过未指定的样本 {idx + 1}/{len(data)}: id={sample_id}")
                        continue
                
                # 检查任务ID
                key = el.get('id', '') if 'id' in el else None
                if not key:
                    print(f"警告: 任务 {idx + 1} 没有 id，跳过")
                    continue
                
                # 先检查每个采样是否已经完成（先查找是否已运行过，避免不必要的memory请求）
                results = []
                need_process = False
                for sample_idx in range(N_SAMPLE):
                    if key in existing_results_list[sample_idx]:
                        # 该采样已完成，使用已有结果
                        existing_item = existing_results_list[sample_idx][key]
                        res = existing_item.get('action_res', '')
                        results.append(res)
                        print(f"  采样 {sample_idx + 1}/{N_SAMPLE} 已存在，跳过")
                    else:
                        # 该采样未完成，需要处理
                        need_process = True
                        results.append(None)  # 占位符
                
                # 如果所有采样都已完成，跳过该任务（包括跳过memory获取）
                if not need_process:
                    print(f"跳过已完全处理的任务 {idx + 1}/{len(data)}: id={key}")
                    continue
                
                # 根据 method 对当前样本的工具描述进行改写或追加 memory（只在需要处理时才获取）
                action_prompt = el["action_prompt"]
                method = getattr(args, "method", None)
                if method == "draft":
                    # 使用 draft 版工具文档替换 prompt 中的 tool description
                    action_prompt = _rewrite_action_prompt_with_draft(action_prompt)
                elif method in ("mem", "m0"):
                    # 在工具描述后追加 memory 信息
                    source_task = el.get("id", f"{file}-{idx}")
                    action_prompt = _augment_action_prompt_with_memory(
                        action_prompt, source_task=source_task
                    )
                
                print(f"处理任务 {idx + 1}/{len(data)} (已处理: {processed_count}, id={key})")
                
                method = getattr(args, "method", None)
                # mem 方法不使用并行
                use_batch_inference = method in ("toolllama", "qwenft", "llama")
                use_qwen_batch = IS_QWEN and method != "mem"  # Qwen 模型也支持批量推理，但 mem 不使用并行
                use_gpt_batch = IS_GPT and method != "mem"  # GPT 模型也支持批量推理，但 mem 不使用并行
                
                # 对于 vLLM 方法、Qwen/GPT 模型，收集所有未完成的采样进行批量推理
                # 但 mem 方法不使用并行
                if (use_batch_inference or use_qwen_batch or use_gpt_batch) and method != "mem":
                    batch_prompts = []
                    batch_indices = []
                    for sample_idx in range(N_SAMPLE):
                        if results[sample_idx] is None:
                            batch_prompts.append(action_prompt)  # 相同的 prompt，但会被处理多次以支持采样
                            batch_indices.append(sample_idx)
                    
                    if batch_prompts:
                        print(f"  开始批量推理，共 {len(batch_prompts)} 个采样...")
                        
                        if use_batch_inference:
                            # vLLM 批量推理
                            model_name_map = {
                                "toolllama": "toolllama",
                                "qwenft": "toolqwen3",
                                "llama": "toolllama"
                            }
                            vllm_model_name = model_name_map.get(method, "toolqwen3")
                            temperature = getattr(args, 'temperature', 0.5)
                            
                            batch_results = get_res_vllm_batch(
                                strings=batch_prompts,
                                model_name=vllm_model_name,
                                temperature=temperature
                            )
                        elif use_gpt_batch:
                            # GPT 批量推理
                            batch_results = get_res_gpt_batch(
                                strings=batch_prompts,
                                gpt_model=GPT_MODEL_ID,
                                max_retries=3,
                                timeout=60
                            )
                        elif use_qwen_batch:
                            # Qwen 批量推理
                            batch_results = get_res_qwen_batch(
                                strings=batch_prompts,
                                qwen_model=QWEN_MODEL_ID,
                                max_retries=3,
                                timeout=60
                            )
                        else:
                            batch_results = []
                        
                        # 将结果填入 results
                        for i, sample_idx in enumerate(batch_indices):
                            results[sample_idx] = batch_results[i] if i < len(batch_results) else "[ERROR] 批量推理结果不足"
                            print(f"  结果 {sample_idx + 1}: {results[sample_idx][:100]}...")
                else:
                    # 非批量推理方法，保持原有逻辑
                    # 对每个未完成的采样进行推理
                    for sample_idx in range(N_SAMPLE):
                        # 如果该采样已有结果，跳过
                        if results[sample_idx] is not None:
                            continue
                        
                        try:
                            print(f"  采样 {sample_idx + 1}/{N_SAMPLE}...")
                            if IS_GPT:
                                # GPT 远程推理（单个请求，内部有重试机制）
                                res = get_res_gpt(string=action_prompt, gpt_model=GPT_MODEL_ID)
                            elif IS_QWEN:
                                # Qwen 远程推理（单个请求，内部有重试机制）
                                res = get_res_qwen(string=action_prompt, qwen_model=QWEN_MODEL_ID)
                            else:
                                # 本地/HF 模型推理
                                res = generation(model=model, tokenizer=tokenizer, input=action_prompt)
                            print(f"  结果 {sample_idx + 1}: {res[:100]}...")
                            results[sample_idx] = res
                        except Exception as e:
                            error_msg = f"[ERROR] 推理失败: {str(e)}"
                            print(f"  错误 {sample_idx + 1}: {error_msg}")
                            results[sample_idx] = error_msg
                
                # 将结果分别保存到对应的文件（只保存新完成的采样）
                for sample_idx in range(N_SAMPLE):
                    # 如果该采样之前已存在，跳过保存（避免重复）
                    if key in existing_results_list[sample_idx]:
                        continue
                    
                    el_copy = el.copy()
                    el_copy['action_res'] = results[sample_idx]
                    # 如果使用了 memory，需要保存修改后的 prompt
                    method = getattr(args, "method", None)
                    if method in ("mem", "m0"):
                        el_copy['action_prompt'] = action_prompt
                    all_data_list[sample_idx].append(el_copy)
                    # 每处理一条就保存一次
                    save_json(all_data_list[sample_idx], output_files[sample_idx])
                
                # 如果启用了 memory，在每个样本完成后立即发送第一个采样结果到 memory
                method = getattr(args, "method", None)
                if method == "mem":
                    try:
                        # 使用第一个采样结果（index 0）发送到 memory
                        sample_to_send = el.copy()
                        sample_to_send['action_res'] = results[0]
                        sample_to_send['action_prompt'] = action_prompt
                        print(f"  发送样本 {el.get('id', idx)} 到 memory...")
                        send_single_sample(
                            item=sample_to_send,
                            workspace_id="toole-ds",
                            endpoint="http://localhost:8002/add_tool_call_result",
                            timeout=30.0,
                            model_name=model_name,
                        )
                    except Exception as e:
                        print(f"  ⚠️ 发送样本到 memory 失败: {e}")
                elif method == "m0":
                    try:
                        # 使用第一个采样结果（index 0）发送到 mem0
                        sample_to_send = el.copy()
                        sample_to_send['action_res'] = results[0]
                        sample_to_send['action_prompt'] = action_prompt
                        print(f"  发送样本 {el.get('id', idx)} 到 mem0...")
                        send_single_sample_to_mem0(
                            item=sample_to_send,
                            model_name=model_name,
                        )
                    except Exception as e:
                        print(f"  ⚠️ 发送样本到 mem0 失败: {e}")
                
                processed_count += 1



    

def get_res_chatgpt(string, gpt_model):
    completion = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[
            {"role": "user",
             "content": string}
        ]
    )
    print(completion.choices[0].message['content'])
    return completion.choices[0].message['content']


def _should_retry_qwen(exception, status_code=None):
    """判断 Qwen 请求是否应该重试该异常"""
    # 检查状态码
    if status_code:
        # 429: 速率限制，503: 服务不可用，502: 网关错误，500: 服务器错误
        if status_code in [429, 503, 502, 500]:
            return True
    
    error_msg = str(exception).lower()
    # 检查是否是速率限制相关的错误消息
    rate_limit_keywords = ['rate limit', 'rate_limit', '429', 'too many requests', 'quota']
    if any(keyword in error_msg for keyword in rate_limit_keywords):
        return True
    # 检查是否是临时性错误
    recoverable_keywords = ['timeout', 'connection', 'server error', '503', '502', '500']
    if any(keyword in error_msg for keyword in recoverable_keywords):
        return True
    # 检查是否是连接错误
    if isinstance(exception, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    return False

def get_res_qwen(string, qwen_model, max_retries=3, timeout: int = 60):
    """
    使用 Qwen 的 OpenAI 兼容推理接口。
    支持自动重试和指数退避。
    需要在环境变量中设置：
      - QWEN_API_KEY
      - （可选）QWEN_API_BASE，例如: https://dashscope.aliyuncs.com/compatible-mode/v1
    
    如果失败，返回错误信息字符串而不是抛出异常。
    """
    # 打印最终发送给 Qwen 远程接口的 prompt
    print("===== MODEL PROMPT (QWEN) START =====")
    print(string)
    print("===== MODEL PROMPT (QWEN) END =====")

    # 从环境变量获取 API key，如果没有则使用硬编码的值
    api_key = os.getenv("QWEN_API_KEY", 'your api key')
    api_base = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not api_key:
        return "[ERROR] QWEN_API_KEY 未设置，无法调用 Qwen 接口"

    # Qwen 模型名称映射：将简化的模型名映射到完整的 API 模型名
    qwen_model_mapping = {
        # 这里可以按需补充模型别名映射，例如：
        # "qwen-max": "qwen-max",
    }
    
    # 如果模型名在映射中，使用映射后的名称；否则直接使用原名称
    api_model_name = qwen_model_mapping.get(qwen_model.lower(), qwen_model)
    
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": api_model_name,
        "messages": [
            {
                "role": "user",
                "content": string,
            }
        ],
        "enable_thinking": False,
    }
    
    # 重试逻辑（指数退避）
    max_attempts = max_retries + 1  # 初始尝试 + 重试次数
    last_error = None
    last_status_code = None
    
    for attempt in range(max_attempts):
        try:
            print(f"[QWEN] 第 {attempt + 1}/{max_attempts} 次请求")
            print(f"[QWEN] URL: {url}")
            print(f"[QWEN] 模型: {api_model_name} (原始: {qwen_model})")
            print(f"[QWEN] timeout: {timeout}s")

            start_t = time.time()
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            elapsed = time.time() - start_t
            last_status_code = resp.status_code
            print(f"[QWEN] 本次请求耗时: {elapsed:.2f}s, 状态码: {resp.status_code}")
            
            # 如果请求失败，检查是否应该重试
            if resp.status_code != 200:
                error_detail = resp.text
                error_msg = f"[ERROR] Qwen API 错误 (状态码 {resp.status_code}): {error_detail}"
                last_error = Exception(error_msg)
                
                # 检查是否应该重试
                if _should_retry_qwen(last_error, status_code=resp.status_code) and attempt < max_attempts - 1:
                    wait_time = min(2 ** attempt, 60)  # 指数退避，最多60秒
                    print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                    print(f"  {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                    return error_msg
            
            data = resp.json()
            if "choices" not in data or not data["choices"]:
                error_msg = f"[ERROR] Qwen 接口返回异常: {data}"
                last_error = Exception(error_msg)
                if attempt < max_attempts - 1:
                    wait_time = min(2 ** attempt, 60)
                    print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                    print(f"  {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    return error_msg
            
            content = data["choices"][0]["message"]["content"]
            print(content)
            return content
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"[ERROR] HTTP 错误: {str(e)}"
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                error_msg += f"\n响应内容: {e.response.text}"
                if hasattr(e.response, 'status_code'):
                    last_status_code = e.response.status_code
            last_error = e
            
            if _should_retry_qwen(e, status_code=last_status_code) and attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 60)
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                print(f"  {wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                return error_msg
                
        except requests.exceptions.Timeout as e:
            error_msg = f"[ERROR] Qwen 请求超时 (timeout={timeout}s): {str(e)}"
            last_error = e
            if _should_retry_qwen(e) and attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 60)
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                print(f"  URL: {url}")
                print(f"  模型: {api_model_name} (原始: {qwen_model})")
                print(f"  {wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                return error_msg
                
        except requests.exceptions.ConnectionError as e:
            error_msg = f"[ERROR] Qwen 连接错误: {str(e)}"
            last_error = e
            if _should_retry_qwen(e) and attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 60)
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                print(f"  {wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                return error_msg
                
        except Exception as e:
            error_msg = f"[ERROR] 请求异常: {type(e).__name__}: {str(e)}"
            last_error = e
            if _should_retry_qwen(e) and attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 60)
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                print(f"  {wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                return error_msg
    
    # 如果所有重试都失败，返回最后的错误信息
    return f"[ERROR] Qwen 请求失败: {str(last_error)}" if last_error else "[ERROR] 未知错误"


def get_res_qwen_batch(
    strings: List[str],
    qwen_model: str,
    max_retries: int = 3,
    timeout: int = 60,
    max_workers: int = 8
) -> List[str]:
    """
    使用 Qwen 的 OpenAI 兼容推理接口进行批量并行请求。
    支持并行请求和自动重试。
    
    Args:
        strings: 输入的 prompt 列表
        qwen_model: Qwen 模型名称
        max_retries: 最大重试次数
        timeout: 单个请求的超时时间（秒）
        max_workers: 最大并发线程数，默认为 8
        
    Returns:
        List[str]: 每个请求的模型回答列表
    """
    if not strings:
        return []
    
    print(f"===== Qwen 批量推理开始，共 {len(strings)} 个请求 =====")
    
    # 使用线程池进行并发请求
    results = [None] * len(strings)
    
    def process_single(index: int, string: str) -> Tuple[int, str]:
        """处理单个请求的辅助函数（内部已包含重试机制）"""
        try:
            result = get_res_qwen(
                string=string,
                qwen_model=qwen_model,
                max_retries=max_retries,
                timeout=timeout
            )
            return index, result
        except Exception as e:
            error_msg = f"[ERROR] 批量推理中单个请求失败: {type(e).__name__}: {str(e)}"
            logging.error(f"  请求 {index + 1} 失败: {error_msg}")
            return index, error_msg
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, idx, s): idx
            for idx, s in enumerate(strings)
        }
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(strings), desc="Qwen 批量推理进度", ncols=100) as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                pbar.update(1)
                if result and result.startswith("[ERROR]"):
                    pbar.set_postfix({"错误": 1})
    
    print(f"===== Qwen 批量推理完成 =====")
    return results


def get_res_gpt(string, gpt_model, max_retries=3, timeout: int = 60):
    """
    使用 GPT 的 OpenAI 兼容推理接口（参考 evaluator.py 的做法）。
    支持自动重试和指数退避。
    需要在环境变量中设置：
      - GPT_API_KEY（或 OPENAI_API_KEY）
      - （可选）GPT_API_BASE（或 OPENAI_API_BASE），例如: https://api.gpt.ge/v1
    
    如果失败，返回错误信息字符串而不是抛出异常。
    """
    # 打印最终发送给 GPT 远程接口的 prompt
    print("===== MODEL PROMPT (GPT) START =====")
    print(string)
    print("===== MODEL PROMPT (GPT) END =====")

    # 从环境变量获取 API key，参考 evaluator.py 的默认值
    api_key = os.getenv("GPT_API_KEY", os.getenv("OPENAI_API_KEY", "your api key"))
    api_base = os.getenv("GPT_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.gpt.ge/v1/"))
    # 确保 api_base 格式正确（移除末尾的斜杠）
    api_base = api_base.rstrip('/')
    # 如果已经包含 /v1，就使用它；否则添加 /v1
    if not api_base.endswith('/v1'):
        api_base = f"{api_base}/v1"
    
    if not api_key:
        return "[ERROR] GPT_API_KEY 或 OPENAI_API_KEY 未设置，无法调用 GPT 接口"

    # GPT 模型名称映射：将简化的模型名映射到完整的 API 模型名
    gpt_model_mapping = {
        "chatgpt": "gpt-3.5-turbo",
        "gpt-4": "gpt-4",
    }
    
    # 如果模型名在映射中，使用映射后的名称；否则直接使用原名称
    api_model_name = gpt_model_mapping.get(gpt_model.lower(), gpt_model)
    
    url = f"{api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": api_model_name,
        "messages": [
            {
                "role": "user",
                "content": string,
            }
        ],
    }
    
    # 重试逻辑（指数退避）
    max_attempts = max_retries + 1  # 初始尝试 + 重试次数
    last_error = None
    last_status_code = None
    
    for attempt in range(max_attempts):
        try:
            print(f"[GPT] 第 {attempt + 1}/{max_attempts} 次请求")
            print(f"[GPT] URL: {url}")
            print(f"[GPT] 模型: {api_model_name} (原始: {gpt_model})")
            print(f"[GPT] timeout: {timeout}s")

            start_t = time.time()
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            elapsed = time.time() - start_t
            last_status_code = resp.status_code
            print(f"[GPT] 本次请求耗时: {elapsed:.2f}s, 状态码: {resp.status_code}")
            
            # 如果请求失败，检查是否应该重试
            if resp.status_code != 200:
                error_detail = resp.text
                error_msg = f"[ERROR] GPT API 错误 (状态码 {resp.status_code}): {error_detail}"
                last_error = Exception(error_msg)
                
                # 检查是否应该重试
                if _should_retry_qwen(last_error, status_code=resp.status_code) and attempt < max_attempts - 1:
                    wait_time = min(2 ** attempt, 60)  # 指数退避，最多60秒
                    print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                    print(f"  {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                    return error_msg
            
            data = resp.json()
            if "choices" not in data or not data["choices"]:
                error_msg = f"[ERROR] GPT 接口返回异常: {data}"
                last_error = Exception(error_msg)
                if attempt < max_attempts - 1:
                    wait_time = min(2 ** attempt, 60)
                    print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                    print(f"  {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    return error_msg
            
            content = data["choices"][0]["message"]["content"]
            print(content)
            return content
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"[ERROR] HTTP 错误: {str(e)}"
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                error_msg += f"\n响应内容: {e.response.text}"
                if hasattr(e.response, 'status_code'):
                    last_status_code = e.response.status_code
            last_error = e
            
            if _should_retry_qwen(e, status_code=last_status_code) and attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 60)
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                print(f"  {wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                return error_msg
                
        except requests.exceptions.Timeout as e:
            error_msg = f"[ERROR] GPT 请求超时 (timeout={timeout}s): {str(e)}"
            last_error = e
            if _should_retry_qwen(e) and attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 60)
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                print(f"  URL: {url}")
                print(f"  模型: {api_model_name} (原始: {gpt_model})")
                print(f"  {wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                return error_msg
                
        except requests.exceptions.ConnectionError as e:
            error_msg = f"[ERROR] GPT 连接错误: {str(e)}"
            last_error = e
            if _should_retry_qwen(e) and attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 60)
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                print(f"  {wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                return error_msg
                
        except Exception as e:
            error_msg = f"[ERROR] 请求异常: {type(e).__name__}: {str(e)}"
            last_error = e
            if _should_retry_qwen(e) and attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 60)
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                print(f"  {wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                print(f"尝试 {attempt + 1}/{max_attempts}: {error_msg}")
                return error_msg
    
    # 如果所有重试都失败，返回最后的错误信息
    return f"[ERROR] GPT 请求失败: {str(last_error)}" if last_error else "[ERROR] 未知错误"


def get_res_gpt_batch(
    strings: List[str],
    gpt_model: str,
    max_retries: int = 3,
    timeout: int = 60,
    max_workers: int = 8
) -> List[str]:
    """
    使用 GPT 的 OpenAI 兼容推理接口进行批量并行请求。
    支持并行请求和自动重试。
    
    Args:
        strings: 输入的 prompt 列表
        gpt_model: GPT 模型名称
        max_retries: 最大重试次数
        timeout: 单个请求的超时时间（秒）
        max_workers: 最大并发线程数，默认为 8
        
    Returns:
        List[str]: 每个请求的模型回答列表
    """
    if not strings:
        return []
    
    print(f"===== GPT 批量推理开始，共 {len(strings)} 个请求 =====")
    
    # 使用线程池进行并发请求
    results = [None] * len(strings)
    
    def process_single(index: int, string: str) -> Tuple[int, str]:
        """处理单个请求的辅助函数（内部已包含重试机制）"""
        try:
            result = get_res_gpt(
                string=string,
                gpt_model=gpt_model,
                max_retries=max_retries,
                timeout=timeout
            )
            return index, result
        except Exception as e:
            error_msg = f"[ERROR] 批量推理中单个请求失败: {type(e).__name__}: {str(e)}"
            logging.error(f"  请求 {index + 1} 失败: {error_msg}")
            return index, error_msg
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, idx, s): idx
            for idx, s in enumerate(strings)
        }
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(strings), desc="GPT 批量推理进度", ncols=100) as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                pbar.update(1)
                if result and result.startswith("[ERROR]"):
                    pbar.set_postfix({"错误": 1})
    
    print(f"===== GPT 批量推理完成 =====")
    return results


def run_single_test(args):
    model_mapping = {"baichuan-inc/Baichuan-13B-Chat": "baichuan-13b",
                     "baichuan-inc/Baichuan2-13B-chat": "baichuan2-13b",  
                     "THUDM/chatglm2-6b": "chatglm2",
                     "lmsys/vicuna-13b-v1.3": "vicuna-13b",
                     "lmsys/vicuna-7b-v1.3": "vicuna-7b",
                     "lmsys/vicuna-33b-v1.3": "vicuna-33b",
                     "meta-llama/Llama-2-7b-chat-hf": "llama2-7b",
                     "meta-llama/Llama-2-13b-chat-hf": "llama2-13b",
                     "llama-2": "llama2-7b",  # 支持简化的模型名，映射到本地模型
                     'TheBloke/koala-13B-HF': "koala-13b",
                     "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": "oasst-12b",
                     "WizardLM/WizardLM-13B-V1.2": "wizardlm-13b",
                     'ernie': "ernie",
                     "chatgpt": 'chatgpt',
                     'gpt-4': 'gpt-4'}
    
    # 保存原始模型名用于后续映射（在转换为路径之前）
    original_model_path = args.model_path
    
    # 如果模型名为 "llama-2"，转换为本地模型路径
    if args.model_path == "llama-2":
        # 获取当前脚本所在目录，然后构建到模型目录的路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 方法1: 尝试相对路径（从 experiment/MetaTool/src/generation 到 experiment/Llama-2-7b-chat-hf）
        local_model_path = os.path.join(script_dir, "../../../Llama-2-7b-chat-hf")
        local_model_path = os.path.normpath(local_model_path)
        
        # 方法2: 从 workspace root 构建绝对路径
        # 从 experiment/MetaTool/src/generation 上4级到达 workspace root
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
        absolute_model_path = os.path.join(workspace_root, "experiment", "Llama-2-7b-chat-hf")
        absolute_model_path = os.path.normpath(absolute_model_path)
        
        # 优先使用相对路径，如果不存在则使用绝对路径
        if os.path.exists(local_model_path):
            print(f"检测到模型名为 'llama-2'，使用本地模型路径: {local_model_path}")
            args.model_path = local_model_path
        elif os.path.exists(absolute_model_path):
            print(f"检测到模型名为 'llama-2'，使用本地模型路径: {absolute_model_path}")
            args.model_path = absolute_model_path
        else:
            print(f"警告: 未找到本地 llama-2 模型，尝试使用原始路径 '{args.model_path}'")
            print(f"  尝试过的相对路径: {local_model_path}")
            print(f"  尝试过的绝对路径: {absolute_model_path}")

    global IS_QWEN, QWEN_MODEL_ID, IS_GPT, GPT_MODEL_ID
    
    # 如果使用微调模型（toolllama、qwenft、llama），不需要加载本地模型
    method = getattr(args, "method", None)
    if method in ("toolllama", "qwenft", "llama"):
        if not OPENAI_CLIENT_AVAILABLE:
            raise ImportError("OpenAI 客户端未安装，无法使用 --method toolllama/qwenft/llama 选项。请运行: pip install openai")
        print(f"使用微调模型方法: {method}，将通过 vLLM 服务调用")
        model = None
        tokenizer = None
        IS_QWEN = False
        QWEN_MODEL_ID = None
    else:
        # 判断是否为 GPT 系列模型（用于走远程 GPT 接口）
        model_path_lower = (args.model_path or "").lower()
        IS_GPT = "gpt" in model_path_lower or args.model_path in ['chatgpt', 'gpt-4']
        GPT_MODEL_ID = args.model_path if IS_GPT else None
        
        # 判断是否为 Qwen 系列模型（用于走远程 Qwen 接口）
        # 支持 qwen、deepseek 等远程 API 模型（GPT 已单独处理）
        IS_QWEN = (
            ("qwen" in model_path_lower or "deepseek" in model_path_lower) and not IS_GPT
        )
        QWEN_MODEL_ID = args.model_path if IS_QWEN else None
        
        # 打印调试信息
        if IS_GPT:
            print(f"检测到 GPT 远程 API 模型: {args.model_path}，将使用 GPT 远程接口，不加载本地模型")
        elif IS_QWEN:
            print(f"检测到 Qwen/DeepSeek 远程 API 模型: {args.model_path}，将使用 Qwen 远程接口，不加载本地模型")
        else:
            print(f"模型路径: {args.model_path}，将尝试加载本地模型")

        if args.model_path != 'ernie' and args.model_path != 'chatgpt' and args.model_path != 'gpt-4' and not IS_QWEN and not IS_GPT:
            # 在 Mac 上使用 MPS 时，num_gpus 应该设置为 0（MPS 不是通过 num_gpus 控制的）
            # 检测是否使用 MPS
            use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if use_mps:
                print("检测到 Mac MPS 可用，将使用 MPS 进行推理")
                num_gpus_to_use = 0  # MPS 不需要 num_gpus
            else:
                num_gpus_to_use = args.num_gpus
            
            model, tokenizer = load_model(
                args.model_path,
                num_gpus=num_gpus_to_use,
                max_gpu_memory=args.max_gpu_memory,
                load_8bit=args.load_8bit,
                cpu_offloading=args.cpu_offloading,
                revision=args.revision,
                debug=args.debug,
            )
            
            # 模型加载后，确保移动到正确的设备
            # 对于 MPS，需要显式移动模型
            if use_mps:
                try:
                    device = get_best_device()
                    print(f"将模型移动到设备: {device}")
                    model = model.to(device)
                    print("模型已成功移动到 MPS 设备")
                except Exception as e:
                    print(f"警告: 无法将模型移动到 MPS，将使用 CPU: {e}")
                    model = model.to('cpu')
        else:
            model = None
            tokenizer = None
    
    test_type = args.test_type
    # 如果使用微调模型（toolllama、qwenft、llama），model_name 直接从 method 映射
    if method in ("toolllama", "qwenft", "llama"):
        # 从 method 映射到 model_name（用于输出目录和测试数据目录）
        method_to_model_name = {
            "toolllama": "toolllama",
            "qwenft": "qwenft",
            "llama": "llama"
        }
        model_name = method_to_model_name.get(method, method)
    # 如果是 Qwen 模型或其他不在映射中的模型，使用 model_path 作为 model_name
    # 使用原始模型名进行映射（如果路径被转换了，仍能正确映射）
    elif IS_QWEN or original_model_path not in model_mapping:
        model_name = original_model_path
    else:
        model_name = model_mapping[original_model_path]
    print(test_type)
    if test_type == 'tool_test_thought':
        tool_test_thought(model_name=model_name, model=model, tokenizer=tokenizer)
    elif test_type == 'tool_test_action':
        tool_test_action(model_name=model_name, model=model, tokenizer=tokenizer)
    else:
        print("Invalid test_type. Please provide a valid test_type.")
        return None
    return "OK"





@torch.inference_mode()
def main(args, max_retries=3, retry_interval=3):
    """
    主函数：执行测试任务
    如果失败，记录错误信息但继续执行，不中断程序
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            state = run_single_test(args)
            message = f"测试任务成功完成 (尝试 {attempt + 1}/{max_retries})"
            logging.info(message)
            print(message)
            return state  
        except Exception as e:
            error_msg = f"测试任务失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
            traceback.print_exc()
            logging.error(error_msg)
            print(error_msg)
            last_error = error_msg
            
            if attempt < max_retries - 1:
                print(f"等待 {retry_interval} 秒后重试...")
                time.sleep(retry_interval)
            else:
                # 最后一次尝试失败，记录错误但继续执行
                final_error = f"[FINAL ERROR] 所有 {max_retries} 次尝试均失败。最后错误: {str(e)}"
                print(final_error)
                logging.error(final_error)
                # 不返回 None，而是返回错误信息，让调用者知道发生了什么
                return final_error

    # 如果所有重试都失败，返回最后的错误信息
    return last_error if last_error else "[ERROR] 未知错误"  




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, default='', help="模型路径。注意：当使用 --method toolllama/qwenft/llama 时，此参数不需要提供（模型名称由 method 自动确定）")
    parser.add_argument("--test_type", type=str, default='tool')
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["mem", "m0", "toolllama", "qwenft", "llama", "draft"],
        help="选择使用的方法：'mem' 表示在工具描述后追加 memory 信息（需要本地 memory 服务可用），'m0' 表示使用 mem0 进行记忆管理（需要安装 mem0ai: pip install mem0ai），'toolllama'/'qwenft'/'llama' 表示使用本地 vLLM 服务调用微调模型（需要安装 openai: pip install openai，且需要本地 vLLM 服务运行在 http://127.0.0.1:57723/v1，使用这些方法时不需要提供 --model_path 参数），'draft' 表示在推理前将 prompt 中的工具文档替换为 draft_tool_docs.py 生成的改写版本",
    )
    parser.add_argument(
        "--sample-ids",
        type=str,
        default=None,
        help="指定要运行的样本ID，用逗号分隔（例如：'id1,id2,id3'）。如果不指定，则运行所有样本。",
    )
    args = parser.parse_args()
    # Generate a unique timestamp for the log file
    timestamp = time.strftime("%Y%m%d%H%M%S")
    log_filename = f"test_log_{timestamp}.txt"

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    state = main(args,)
    print(state)
    
    

