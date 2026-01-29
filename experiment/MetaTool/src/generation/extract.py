#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Task2.json 中：
1）按 subtask1-4 统计 Similar / Scenario / Reliability / Multi-tool 以及平均准确率；
2）根据每条样本生成工具记忆 payload，并按批次发送到指定 endpoint。

注意：
- 假设 Task2.json 是一个列表，每个元素是一个 dict，至少包含：
  - id: 唯一 ID
  - source_subtask 或 subtask: 用于区分 subtask1-4
  - action_res: 模型真实使用的工具信息（字符串 / 列表 / dict 均可，脚本内做了尽量通用的解析）
  - tool: ground truth 工具（字符串或字符串列表）
  - thought_prompt: 作为 context
- 工具名比较一律用小写（用于匹配），但保留原始大小写用于输出（若能获取）。
- 本脚本只是示例，请根据你实际 Task2.json 的字段情况做适当调整。
"""

import argparse
import json
from tqdm import tqdm
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    print("警告: mem0 未安装，无法使用 mem0 相关功能。请安装: pip install mem0ai")


@dataclass
class SampleInfo:
    raw: Dict[str, Any]
    sample_id: str
    subtask: int
    used_tools: Set[str]          # 小写
    gt_tools: Set[str]            # 小写
    used_tool_names_raw: List[str]
    gt_tool_names_raw: List[str]
    success: bool = False         # 根据 1 中的规则计算后再填


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Task2.json 统计指标并发送工具调用结果"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Task2.json 路径（包含所有 subtask1-4 样本）",
    )
    parser.add_argument(
        "--workspace-id",
        default="toole",
        help="POST body 中使用的 workspace_id",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="每个 HTTP 请求中包含的 tool_call_results 数量",
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
        "--method",
        type=str,
        default="m0",
        choices=["mem", "m0"],
        help="选择发送方式：'mem' 表示发送到 HTTP 服务，'m0' 表示发送到 mem0（默认）",
    )
    return parser.parse_args()


def get_subtask_id(item: Dict[str, Any]) -> Optional[int]:
    """
    从 item 中解析 subtask:
    - 优先使用 item['subtask']（若存在）；
    - 否则从 source_subtask（如 'Task2-Subtask3.json'）中解析数字 1-4。
    """
    if "subtask" in item:
        try:
            return int(item["subtask"])
        except Exception:
            pass

    src = item.get("source_subtask") or ""
    m = re.search(r"Subtask(\d+)", str(src))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def normalize_tool_field(tool_field: Any) -> Tuple[Set[str], List[str]]:
    """
    将 tool 字段统一解析为：
    - 小写集合（用于比较）
    - 保留原始字符串列表（用于输出）
    tool 字段可以是：
    - 字符串：'Calculator'
    - 列表：['ToolSearcher', 'QueryStock']
    """
    raw_names: List[str] = []
    if tool_field is None:
        return set(), raw_names

    if isinstance(tool_field, str):
        raw_names = [tool_field.strip()]
    elif isinstance(tool_field, list):
        raw_names = [str(x).strip() for x in tool_field if x]
    else:
        # 其他类型先转成字符串
        raw_names = [str(tool_field).strip()]

    lowered = {name.lower() for name in raw_names if name}
    return lowered, raw_names


def extract_used_tools_from_action_res(action_res: Any) -> Tuple[Set[str], List[str]]:
    """
    从 action_res 中抽取"真实使用的工具名"，处理以下情况：
    1. 包含 "none"（大小写不敏感）→ 返回空集合
    2. 列表格式 '[Tool1, Tool2]' → 提取多个工具名
    3. 逗号分隔格式 'NewsTool, FinanceTool' → 提取多个工具名（用于 subtask4 多工具任务）
    4. 单工具格式 'timeport' → 直接作为工具名
    返回：
    - 小写集合（用于比较）
    - 原始字符串列表（保留原始大小写）
    """
    raw_names: List[str] = []
    
    # 将 action_res 转为字符串并转小写用于判断
    if action_res is None:
        return set(), []
    
    s = str(action_res).strip()
    s_lower = s.lower()
    
    # 情况1: 包含 "none" → 返回空
    if "none" == s_lower:
        return set(), []
    
    # 情况2: 列表格式 '[Tool1, Tool2]' 或 '[Tool1,Tool2]'
    if s.startswith("[") and s.endswith("]"):
        # 去掉首尾的 []
        inner = s[1:-1].strip()
        if inner:
            # 按逗号分割
            parts = [p.strip() for p in inner.split(",")]
            for part in parts:
                # 去掉可能的引号
                part = part.strip('"\'')
                # 允许工具名包含字母、数字、下划线、&、- 等常见符号
                if part and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_&\-]*", part):
                    raw_names.append(part)
        if raw_names:
            lowered = {name.lower() for name in raw_names}
            return lowered, raw_names
    
    # 情况3: 逗号分隔格式 'NewsTool, FinanceTool'（用于 subtask4 多工具任务）
    # 如果字符串包含逗号，但不是列表格式，尝试按逗号分割
    if "," in s and not (s.startswith("[") and s.endswith("]")):
        parts = [p.strip() for p in s.split(",")]
        for part in parts:
            # 去掉可能的引号
            part = part.strip('"\'')
            # 允许工具名包含字母、数字、下划线、&、- 等常见符号
            if part and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_&\-]*", part):
                raw_names.append(part)
        if raw_names:
            lowered = {name.lower() for name in raw_names}
            return lowered, raw_names
    
    # 情况4: 单工具格式，直接作为工具名
    # 去掉可能的引号
    s_clean = s.strip('"\'')
    # 允许工具名包含字母、数字、下划线、&、- 等常见符号
    if s_clean and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_&\-]*", s_clean):
        raw_names.append(s_clean)
        lowered = {s_clean.lower()}
        return lowered, raw_names
    # print(s_lower)
    # 如果都不匹配，返回空
    return set(s_lower), [s_lower]


def compute_sample_success(sample: SampleInfo) -> None:
    """
    根据题意，对每条样本在其所属 subtask 上给出 success（True/False）：
    - subtask1 (Similar) / subtask2 (Scenario)：真实使用工具集合 == GT 工具集合 才正确；
    - subtask3 (Reliability)：真实使用工具集合为空才正确；
    - subtask4 (Multi-tool)：真实使用工具数量必须为 2 且与 GT 集合一致（无顺序）。
    """
    st = sample.subtask
    used = sample.used_tools
    gt = sample.gt_tools

    if st in (1, 2):
        sample.success = used == gt
    elif st == 3:
        sample.success = len(used) == 0
    elif st == 4:
        sample.success = (len(used) == 2) and (used == gt)
    else:
        sample.success = False



def compute_metrics(samples: List[SampleInfo]) -> None:
    """
    统计并打印 Similar / Scenario / Reliability / Multi-tool / Average 的准确率。
    """
    stats = {
        1: {"name": "Similar", "correct": 0, "total": 0},
        2: {"name": "Scenario", "correct": 0, "total": 0},
        3: {"name": "Reliability", "correct": 0, "total": 0},
        4: {"name": "Multi-tool", "correct": 0, "total": 0},
    }

    for s in samples:
        if s.subtask not in stats:
            continue
        stats[s.subtask]["total"] += 1
        if s.success:
            stats[s.subtask]["correct"] += 1

    accs = []
    for st_id in (1, 2, 3, 4):
        name = stats[st_id]["name"]
        correct = stats[st_id]["correct"]
        total = stats[st_id]["total"]
        acc = correct / total if total > 0 else 0.0
        accs.append(acc)
        print(f"{name} (subtask{st_id}) 准确率: {acc:.4f} ({correct}/{total})")

    avg_acc = sum(accs) / 4 if accs else 0.0
    print(f"Average 平均准确率: {avg_acc:.4f}")


def build_natural_language_output(sample: SampleInfo, tool_lc: str, tool_raw: str) -> str:
    """
    根据题意，为某一条样本中的某个工具生成自然语言描述：
    - subtask3 或 非 GT 工具但被调用：工具不该被调用；
    - GT 工具却没被调用：在应该被调用时没有被调用；
    - 其它情况简单说明调用是否符合预期。
    """
    st = sample.subtask
    used = sample.used_tools
    gt = sample.gt_tools

    # 统一加一点上下文
    base_prefix = f""

    # 1) 工具被调用且不在 groundtruth 中
    if tool_lc in used and tool_lc not in gt:
        if st == 3:
            return (
                base_prefix
                + "The tool should not be called in this task."
            )
        else:
            return (
                base_prefix
                + f"Using the wrong tool. The tool should not be called in this task."
            )

    # 2) 工具在 groundtruth 中但没有被调用（一般是我们从 tool 字段补的情况）
    if (st != 3) and tool_lc in gt and tool_lc not in used:
        return (
            base_prefix
            + "The tool should have been called in this task, but was not."
        )

    # # 3) 工具在 groundtruth 中且被调用
    # if tool_lc in gt and tool_lc in used:
    #     if st == 4 and len(gt) == 2:
    #         return (
    #             base_prefix
    #             + "该样本属于多工具调用场景（subtask4），"
    #               "该工具在 groundtruth 中应被使用，模型也正确地调用了它，"
    #               "是多工具协同调用的一部分。"
    #         )
    #     else:
    #         return (
    #             base_prefix
    #             + "该工具在 groundtruth 中应被使用，模型也正确地调用了它，"
    #               "整体工具选择与期望一致。"
    #         )

    # 4) 成功描述
    return base_prefix + "Success."


def compute_tool_success(sample: SampleInfo, tool_lc: str) -> bool:
    """
    为单个工具计算 success 值。
    对于多工具任务（subtask4），需要分别判断每个工具是否正确使用：
    - 工具在 GT 中且被使用了 → success = True
    - 工具在 GT 中但没被使用 → success = False
    - 工具不在 GT 中但被使用了 → success = False
    
    对于其他 subtask，也使用相同的逻辑。
    """
    used = sample.used_tools
    gt = sample.gt_tools
    
    # 工具在 GT 中且被使用了 → success = True
    if tool_lc in gt and tool_lc in used:
        return True
    
    # 其他情况都是 False
    return False


def load_original_action_prompt(model_name: str, task_id: str) -> str:
    """
    从原始测试数据文件中加载指定任务的 action_prompt。
    
    Args:
        model_name: 模型名称，例如 'qwen3-8b' 或 'qwen3-14b'
        task_id: 任务 ID
        
    Returns:
        str: action_prompt 内容，如果找不到则返回空字符串
    """
    # 构建文件路径
    # extract.py 位于 tool/ 目录下，test_data 也在 tool/ 目录下
    base_dir = Path(__file__).parent /"tool"/ "test_data"
    file_path = base_dir / model_name / "Task2.json"
    
    if not file_path.exists():
        print(f"警告: 原始测试数据文件不存在: {file_path}")
        return ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"警告: 文件格式错误，应为 JSON 数组: {file_path}")
            return ""
        
        # 查找对应 id 的任务
        for item in data:
            if isinstance(item, dict) and str(item.get("id", "")) == str(task_id):
                action_prompt = item.get("action_prompt", "")
                return str(action_prompt) if action_prompt else ""
        
        print(f"警告: 未找到任务 ID {task_id} 在文件 {file_path} 中")
        return ""
    except Exception as e:
        print(f"警告: 读取原始测试数据文件失败: {e}")
        return ""


def build_tool_call_records(samples: List[SampleInfo], model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    根据 SampleInfo 构造要发送的 tool_call_results 列表。
    每条样本可能生成多条记录（多工具），字段要求：
    - create_time: 当前时间
    - tool_name: 优先使用 action_res 中的工具名；若无使用但 tool 中有，则使用 tool 中的工具
    - context: thought_prompt（已清理掉 guidance 部分）
    - config: 固定为 空
    - input: 置为空 dict
    - output: 按规则生成的自然语言描述
    - token_cost: 0
    - time_cost: 0.0
    - source_task: id
    - success: 为每个工具单独计算 success（对于多工具任务，每个工具有独立的判断）
    """
    records: List[Dict[str, Any]] = []
    config_path = ""

    for s in samples:
        item = s.raw
        # 如果提供了 model_name，从原始测试数据文件读取 action_prompt
        if model_name:
            thought_prompt = load_original_action_prompt(model_name, s.sample_id)
            # 如果从文件读取失败，回退到使用 item 中的 action_prompt
            if not thought_prompt:
                thought_prompt = item.get("action_prompt", "")
        else:
            # 如果没有提供 model_name，使用 item 中的 action_prompt
            thought_prompt = item.get("action_prompt", "")

        # 构建小写 -> 原始名的映射
        # 优先使用 action_res 中的原始名，如果没有则使用 groundtruth 中的原始名
        lc_to_raw = {}
        
        # 先添加使用的工具（优先）
        for raw in s.used_tool_names_raw:
            lc = raw.lower()
            if lc not in lc_to_raw:
                lc_to_raw[lc] = raw
        
        # 再添加 groundtruth 中的工具（如果还没有映射）
        for raw in s.gt_tool_names_raw:
            lc = raw.lower()
            if lc not in lc_to_raw:
                lc_to_raw[lc] = raw
        
        # 收集所有需要报告的工具：
        # 1. 所有使用的工具
        # 2. GT 中但没被使用的工具（如果 GT 工具已被使用，则不再单独记录）
        tools_to_report = s.used_tools | (s.gt_tools - s.used_tools)

        for tool_lc in tools_to_report:
            tool_raw = lc_to_raw.get(tool_lc, tool_lc)
            output_text = build_natural_language_output(s, tool_lc, tool_raw)
            # 为每个工具单独计算 success
            tool_success = compute_tool_success(s, tool_lc)

            records.append(
                {
                    "create_time": "",  # 将在 batch_and_send 中按批次设置
                    "tool_name": tool_raw,
                    "context": str(thought_prompt),
                    "config": config_path,
                    "input": {},
                    "output": output_text,
                    "token_cost": 0,
                    "success": tool_success,
                    "time_cost": 0.0,
                    "source_task": str(s.sample_id),
                }
            )

    return records


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


def send_to_mem0(
    records: List[Dict[str, Any]],
    mem0_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    将 tool_call_records 发送到 mem0 进行记忆存储。
    参考 test.py 中的方法，使用 memory.add() 添加记忆。
    对于"应该被调用但没有被调用"的情况，将跳过不发送。
    
    Args:
        records: tool_call_records 列表，每个记录包含：
            - tool_name: 工具名称
            - context: 包含 [User's Query Start] 和 [User's Query End] 的完整 context
            - input: 工具输入（dict）
            - output: 工具输出（字符串）
            - success: 是否成功
            - token_cost: token 成本
            - time_cost: 时间成本
            - source_task: 源任务 ID
        mem0_config: mem0 配置字典，如果为 None 则使用默认配置
    """
    if not MEM0_AVAILABLE:
        print("错误: mem0 未安装，无法发送数据到 mem0")
        return
    
    # 默认配置（参考 test.py）
    default_config = {
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
    
    config = mem0_config if mem0_config is not None else default_config
    
    try:
        memory = Memory.from_config(config)
    except Exception as e:
        print(f"错误: 初始化 mem0 Memory 失败: {e}")
        return
    
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    for record in tqdm(records):
        try:
            tool_name = record.get("tool_name", "")
            context_full = record.get("context", "")
            input_data = record.get("input", {})
            output = record.get("output", "")
            success = record.get("success", False)
            token_cost = record.get("token_cost", 0)
            time_cost = record.get("time_cost", 0.0)
            source_task = record.get("source_task", "")
            
            # 跳过"应该被调用但没有被调用"的情况
            if "The tool should have been called in this task, but was not." in output:
                skipped_count += 1
                continue
            
            # 从 context 中提取用户查询
            user_query = extract_user_query_from_context(context_full)
            if not user_query:
                # 如果提取失败，使用完整的 context
                user_query = context_full
            
            # 构建消息（参考 test.py 的格式：user, assistant with tool_calls, tool）
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
            
            # 发送到 mem0（使用 procedural_memory 类型）
            memory.add(messages, agent_id=tool_name, user_id=source_task, memory_type="procedural_memory", app_id="toole-ds")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 记录发送到 mem0 失败 (tool: {record.get('tool_name', 'unknown')}, task: {record.get('source_task', 'unknown')}): {e}")
            fail_count += 1
    
    print(f"✅ mem0 发送完成: 成功 {success_count} 条, 失败 {fail_count} 条, 跳过 {skipped_count} 条（应该被调用但未被调用）")


def send_single_sample(
    item: Dict[str, Any],
    workspace_id: str = "toole",
    endpoint: str = "http://localhost:8002/add_tool_call_result",
    timeout: float = 30.0,
    model_name: Optional[str] = None,
) -> bool:
    """
    从单个样本（dict）构造并立即发送 tool_call_results。
    用于在运行过程中逐个发送样本结果。
    
    Args:
        item: 包含 id, action_res, tool, action_prompt 等字段的样本字典
        workspace_id: workspace ID
        endpoint: 接收工具调用结果的 HTTP endpoint
        timeout: HTTP 请求超时时间（秒）
    
    Returns:
        bool: 发送是否成功
    """
    try:
        # 构造 SampleInfo
        sid = str(item.get("id", ""))
        st = get_subtask_id(item)
        if st is None:
            # 如果无法判断 subtask，默认使用 1
            st = 1
        
        gt_set, gt_raw = normalize_tool_field(item.get("tool"))
        # 对于 subtask3，groundtruth 应该为空
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
        
        # 构造 tool_call_records（单个样本），传递 model_name
        records = build_tool_call_records([sample], model_name=model_name)
        
        if not records:
            print(f"样本 {sid} 没有需要发送的工具记录。")
            return False
        
        # 设置当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for record in records:
            record["create_time"] = current_time
        
        # 立即发送
        payload = {
            "workspace_id": workspace_id,
            "tool_call_results": records,
        }
        
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        if resp.status_code != 200:
            print(
                f"❌ 样本 {sid} 请求失败，状态码: {resp.status_code}, 响应: {resp.text}"
            )
            return False
        
        try:
            body = resp.json()
        except Exception:
            body = {}
        
        if not body or not body.get("success", False):
            print(
                f"❌ 样本 {sid} 接口返回失败，工具数量: {len(records)}，响应: {body or resp.text}"
            )
            return False
        
        print(f"✅ 样本 {sid} 已发送，工具数量: {len(records)}")
        return True
        
    except Exception as e:
        print(f"❌ 样本 {item.get('id', 'unknown')} 发送异常: {e}")
        return False


def batch_and_send(
    records: List[Dict[str, Any]],
    workspace_id: str,
    batch_size: int,
    endpoint: str,
    timeout: float,
) -> None:
    """
    将 records 按 batch_size 划分成若干批，并依次 POST 到 endpoint。
    发送逻辑参考 extract_tool_posts_from_jsonl.py，但不使用 include_gt / include_right。
    在发送前会打乱 records 的顺序，所有批次使用相同的 create_time。
    """
    if not records:
        print("没有需要发送的 tool_call_results。")
        return

    # 打乱 records 的顺序
    shuffled_records = records.copy()
    random.shuffle(shuffled_records)
    print("已打乱工具记录顺序。")

    total_tools = len(shuffled_records)
    total_batches = (total_tools + batch_size - 1) // batch_size
    print(f"共 {total_tools} 条工具记录，将分 {total_batches} 个批次发送。")

    # 所有批次使用相同的时间
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_tools)
        batch = shuffled_records[start:end]

        # 为当前批次设置 create_time（所有批次使用相同时间）
        for record in batch:
            record["create_time"] = current_time_str

        payload = {
            "workspace_id": workspace_id,
            "tool_call_results": batch,
        }

        try:
            resp = requests.post(endpoint, json=payload, timeout=timeout)
            if resp.status_code != 200:
                print(
                    f"❌ 批次 {batch_idx + 1} 请求失败，状态码: {resp.status_code}, 响应: {resp.text}"
                )
                continue
            try:
                body = resp.json()
            except Exception:
                body = {}
            if not body or not body.get("success", False):
                print(
                    f"❌ 批次 {batch_idx + 1} 接口返回失败，工具数量: {len(batch)}，响应: {body or resp.text}"
                )
                continue
            print(
                f"✅ 批次 {batch_idx + 1} 已发送，工具数量: {len(batch)}"
            )
        except Exception as e:
            print(f"❌ 批次 {batch_idx + 1} 发送异常: {e}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 从输入文件路径中提取模型名称
    # 假设路径格式：.../test_data/{model_name}/Task2.json
    model_name = None
    try:
        parts = input_path.parts
        if "test_data" in parts:
            test_data_idx = parts.index("test_data")
            if test_data_idx + 1 < len(parts):
                model_name = parts[test_data_idx + 1]
    except Exception:
        pass

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入的 Task2.json 内容应为 JSON 数组（list）。")

    samples: List[SampleInfo] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        sid = str(item.get("id", ""))
        st = get_subtask_id(item)
        if st is None:
            # 无法判断 subtask 的样本直接跳过
            continue

        gt_set, gt_raw = normalize_tool_field(item.get("tool"))
        # 对于 subtask3，groundtruth 应该为空
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
        samples.append(sample)

    # 1. 统计并打印五个指标
    compute_metrics(samples)

    # 2. 构造并发送 Task2 中的工具信息
    tool_call_records = build_tool_call_records(samples, model_name=model_name)
    print(tool_call_records)
    # # 根据 method 选择发送方式
    # if args.method == "m0":
    #     send_to_mem0(tool_call_records)
    # else:
    #     batch_and_send(
    #         tool_call_records,
    #         workspace_id=args.workspace_id,
    #         batch_size=args.batch_size,
    #         endpoint=args.endpoint,
    #         timeout=args.timeout,
    #     )

if __name__ == "__main__":
    # 按要求：这里只定义代码，不执行。你使用时可取消下面注释。
    main()
    # pass