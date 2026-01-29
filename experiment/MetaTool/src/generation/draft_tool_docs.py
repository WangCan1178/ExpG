#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 MetaTool 中已有的数据与 DRAFT 的重写提示，离线生成一份「draft 版」工具文档，
并保存为 `dataset/plugin_des_draft.json`。

设计思路：
- 原始工具文档来源：`dataset/plugin_des.json`（tool_name -> description 字符串）
- 工具调用信息来源：从 Task2.json 中提取，使用 `src/generation/extract.py` 中的
  `build_tool_call_records` 构造 tool_call_records，然后按工具名分组汇总使用经验
- 重写提示词：复用 `experiment/DRAFT/prompts/rewrite_tool_doc.txt`

生成好 draft 文档后，在 `run.py` 中通过 `--method draft` 自动把 prompt 里的
`tool description` 替换为 draft 描述，推理流程的其它逻辑保持不变。

本脚本的模型调用方式与 DRAFT 方法实现参考
`experiment/gorilla/berkeley-function-call-leaderboard/bfcl_eval/scripts/draft_improver.py`：
- 使用 Qwen/DashScope 的 OpenAI 兼容接口
- 使用 DRAFT 的 Analyzer + Rewriter 多轮迭代与收敛判断
"""

import argparse
import json
import os
import random
import time
import traceback
import threading
from typing import Dict, Any, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import OpenAI, RateLimitError


def load_plugin_des(path: str) -> Dict[str, str]:
    """读取原始工具描述（plugin_des.json）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 应该是一个 dict，实际类型为 {type(data)}")
    return data


def load_rewrite_prompt(metatool_root: str) -> str:
    """
    读取 DRAFT 项目的 rewrite_tool_doc 提示词。
    默认为：experiment/DRAFT/prompts/rewrite_tool_doc.txt
    """
    prompt_path = os.path.join(
        os.path.dirname(metatool_root),
        "DRAFT",
        "prompts",
        "rewrite_tool_doc.txt",
    )
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"未找到 DRAFT 重写提示文件：{prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_tool_call_records_from_task2(
    task2_path: str, model_name: str | None = None
) -> List[Dict[str, Any]]:
    """
    从 Task2.json 文件中读取数据，使用 extract.py 中的逻辑构造 tool_call_records。
    
    Args:
        task2_path: Task2.json 文件路径
        model_name: 模型名称（可选，用于读取原始 action_prompt）
    
    Returns:
        List[Dict[str, Any]]: tool_call_records 列表
    """
    # 导入 extract.py 中的相关函数
    import sys
    
    # 添加项目根目录到路径，以便导入 src.generation.extract
    metatool_root = os.path.dirname(os.path.abspath(__file__))
    if metatool_root not in sys.path:
        sys.path.insert(0, metatool_root)
    
    # 从同目录下的 extract.py 导入（以脚本方式运行时更稳妥）
    from extract import (
        get_subtask_id,
        normalize_tool_field,
        extract_used_tools_from_action_res,
        SampleInfo,
        compute_sample_success,
        build_tool_call_records,
    )
    
    with open(task2_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"{task2_path} 内容应为 JSON 数组（list）")
    
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
    
    # 构造 tool_call_records
    tool_call_records = build_tool_call_records(samples, model_name=model_name)
    return tool_call_records


def aggregate_records_by_tool(
    tool_call_records: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    按工具名分组汇总 tool_call_records。
    
    Args:
        tool_call_records: tool_call_records 列表
    
    Returns:
        Dict[str, List[Dict]]: 工具名 -> 该工具的所有 records 列表
    """
    tool_to_records = defaultdict(list)
    for record in tool_call_records:
        tool_name = record.get("tool_name", "")
        if tool_name:
            tool_to_records[tool_name].append(record)
    return dict(tool_to_records)


def summarize_tool_usage(tool_records: List[Dict[str, Any]]) -> str:
    """
    汇总某个工具的使用经验，生成自然语言描述。
    
    Args:
        tool_records: 该工具的所有调用记录
    
    Returns:
        str: 汇总后的使用经验描述
    """
    if not tool_records:
        return ""
    
    # 统计成功和失败的次数
    success_count = sum(1 for r in tool_records if r.get("success", False))
    total_count = len(tool_records)
    
    # 收集所有 output 文本（这些是自然语言描述）
    outputs = []
    for r in tool_records:
        output = r.get("output", "")
        if output and output.strip():
            outputs.append(output.strip())
    
    # 构建汇总文本
    parts = []
    parts.append(
        f"This tool was called {total_count} time(s), "
        f"with {success_count} successful call(s) and {total_count - success_count} failed call(s)."
    )
    
    if outputs:
        parts.append("\nUsage examples and feedback:")
        # 去重并限制数量，避免过长
        unique_outputs = list(dict.fromkeys(outputs))[:10]  # 最多保留10个不同的输出
        for i, output in enumerate(unique_outputs, 1):
            parts.append(f"{i}. {output}")
    
    return "\n".join(parts)


def build_tool_documentation(
    tool_name: str,
    original_desc: str,
    usage_summary: str | None,
) -> str:
    """
    构造喂给 DRAFT 提示词的 {Tool Description} 内容。
    这里我们把原始文档和（可选的）调用经验拼在一起，交给模型综合重写。
    """
    parts = [
        f"Tool name: {tool_name}",
        f"Original description: {original_desc}",
    ]
    if usage_summary:
        parts.append(
            "Usage guides from previous tool calls (summarized from tool_call_records):"
        )
        parts.append(usage_summary)
    return "\n".join(parts)


openai_client = None


def init_openai_client() -> OpenAI:
    """
    初始化 OpenAI 客户端，默认走 DashScope/Qwen 的 OpenAI 兼容接口，
    与 BFCL 的 draft_improver.py 保持一致的使用方式。
    """
    global openai_client
    if openai_client is None:
        api_key = 'your api key'
        if not api_key:
            raise RuntimeError(
                "未找到可用的 API Key，请设置 DASHSCOPE_API_KEY / QWEN_API_KEY / OPENAI_API_KEY 之一"
            )
        base_url = os.getenv(
            "DASHSCOPE_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        openai_client = OpenAI(api_key=api_key, base_url=base_url)
    return openai_client


def openai_embedding(text: str, embedding_model: str = "text-embedding-v3") -> List[float]:
    """
    获取文本的嵌入向量（使用 Qwen/OpenAI 兼容接口），带 429 错误重试逻辑。
    实现方式参考 BFCL 的 draft_improver.py。
    """
    client = init_openai_client()
    for attempt in range(3):
        try:
            resp = client.embeddings.create(model=embedding_model, input=text)
            return resp.data[0].embedding
        except RateLimitError as e:
            if attempt == 0:
                wait_time = 30
            elif attempt == 1:
                wait_time = 60
            else:
                print("嵌入生成失败: 429 错误重试 3 次后仍失败")
                return [0.0] * 1536
            print(f"嵌入生成遇到 429 错误（速率限制），等待 {wait_time} 秒后重试...: {e}")
            time.sleep(wait_time)
        except Exception as e:
            print(f"嵌入生成失败: {e}")
            return [0.0] * 1536


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度。"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def compute_similarity_and_bleu(
    reference_sentence: str,
    candidate_sentence: str,
    embedding_model: str = "text-embedding-v3",
) -> float:
    """
    计算参考句子和候选句子的相似度，参考 DRAFT 原实现：
    - 使用 embedding 余弦相似度
    - 使用 BLEU（带平滑）
    - 最终分数为二者的平均值
    """
    try:
        # 1) embedding 相似度
        reference_embedding = openai_embedding(reference_sentence, embedding_model=embedding_model)
        candidate_embedding = openai_embedding(candidate_sentence, embedding_model=embedding_model)
        similarity = cosine_similarity(reference_embedding, candidate_embedding)

        # 2) BLEU 分数（按词分 token，并使用平滑）
        reference = [reference_sentence.lower().split()]
        candidate = candidate_sentence.lower().split()
        smoothie = SmoothingFunction().method4
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)

        # 3) 综合得分
        delta = (bleu_score + similarity) / 2
        return float(delta)
    except Exception as e:
        print(f"相似度计算失败: {e}")
        return 0.0


def openai_json_request(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> Dict[str, Any]:
    """
    通用的 LLM JSON 调用工具，参考 BFCL 的 openai_response：
    - 使用 OpenAI 兼容接口
    - 带 429 错误重试
    - 自动去除 ```json/``` 包裹并解析 JSON
    """
    client = init_openai_client()

    content: str | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            break
        except RateLimitError as e:
            print(f"API 调用遇到 429 错误（速率限制）: {e}", traceback.format_exc())
            if attempt == 0:
                wait_time = 30
            elif attempt == 1:
                wait_time = 60
            else:
                print("API 调用失败: 429 错误重试 3 次后仍失败")
                raise
            print(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"API 调用失败: {e}")
            raise

    if content is None:
        raise RuntimeError("无法获取 API 响应内容")

    cleaned_text = content.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        print(f"JSON 解析失败，原始内容: {cleaned_text}")
        return {"error": "JSON parse failed", "raw_content": cleaned_text}


def call_llm(prompt: str, model: str) -> Dict[str, Any]:
    """
    调用 OpenAI 兼容接口，要求返回 JSON：
    {
        "tool_description": str
    }

    具体的 API Key / Base URL 由环境变量控制，脚本本身不做绑定。
    使用方式参考 BFCL 的 draft_improver.py：通过 OpenAI 兼容接口调用 Qwen。
    """
    client = init_openai_client()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    content: str | None = None
    # 与 draft_improver 中 openai_response 类似的 429 重试逻辑
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=2000,
            )
            content = response.choices[0].message.content
            break
        except RateLimitError as e:
            print(f"API 调用遇到 429 错误（速率限制）: {e}", traceback.format_exc())
            if attempt == 0:
                wait_time = 30
            elif attempt == 1:
                wait_time = 60
            else:
                raise
            print(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"API 调用失败: {e}", traceback.format_exc())
            raise

    if content is None:
        raise RuntimeError("无法获取 LLM 响应内容")

    cleaned = (
        content.strip("`json\n")
        .strip("`\n")
        .strip("```\n")
        .strip()
    )
    try:
        data = json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"无法解析 LLM 返回的 JSON：{cleaned}") from e
    if not isinstance(data, dict) or "tool_description" not in data:
        raise ValueError(f"LLM 返回 JSON 中缺少 'tool_description' 字段：{data}")
    return data


def load_prompts() -> Dict[str, Any]:
    """
    加载 DRAFT 提示词模板（Explorer / Analyzer / Rewriter），
    实现方式参考 BFCL 的 draft_improver.py。

    这里直接复用 API-Bank 下的 draft_prompts 目录：
    /Users/xiaocan/Toolmem/experiment/api-bank/draft_prompts
    """
    from pathlib import Path

    prompts_dir = Path("/Users/xiaocan/Toolmem/experiment/api-bank/draft_prompts")
    prompts_dir.mkdir(exist_ok=True, parents=True)

    explorer_path = prompts_dir / "Explorer.txt"
    analyzer_path = prompts_dir / "Analyzer.txt"
    rewriter_path = prompts_dir / "Rewriter.txt"

    # 如果提示词文件不存在，写入默认模板（与 BFCL 版本保持一致）
    if not explorer_path.exists():
        explorer_template = """Your task is to answer the user's query as best you can. You have access to the following API, which you can use via API call to help with your response:

{Tool Description}

Now you have the chance to explore this API. You can do this by 1) synthesizing some natural user query that calling the API could help, 2) extracting the parameters needed to call the API from the generated query. Here, you can focus on queries that only require calling the API once.

Now, first input your synthesized user query. You should make the query natural - for example, try to avoid using the provided API descriptions or API names in the query, as the user does not know what APIs you have access to. However, please make sure that the user query you generate includes the parameters required to call the API, for which you need to generate random information. For required parameters like IP address, location, coordinates, etc., provide specific details. Also try to make the query as specific as possible. Next you need to extract the parameters needed to call the API from your generated user queries based on the provided API documentation descriptions.

Here is an example:

```json
{{
    "User Query": "",
    "Parameters": {{}}
}}
```

=========

Below are queries you have already explored:

{Explored queries}

Based on these, try to explore queries that can help you understand the API further; Avoid synthesizing queries that are too close to the existing ones. Here are some suggestions to explore the API:

{Suggestions}

Now you know a bit more about the API. You can synthesize another user query to explore the API a bit further and consolidate your understanding of the API, based on things that you discovered about this API. You should cherish the opportunity to explore, as each time is precious. Therefore, you should generate new explorations that are different from previous ones as much as possible."""
        with open(explorer_path, "w", encoding="utf-8") as f:
            f.write(explorer_template)

    if not analyzer_path.exists():
        analyzer_template = """You task is to provide suggestions for modifying the tool documentation based on the current tool documentation, the explored query and parameters, and the results returned by the tool. You have access to the following tool:

{Tool Description}

Please note that the existing tool documentation may be incomplete or noisy.

Previously, you generated some user queries and required parameters to explore this API based on the API documentation. Now, you will be provided with the output of this API under these parameters. 

You need to consider the following when providing suggestions: For instance, consider whether the current description is consistent with the actual results returned by the tool, whether the description is comprehensive, and whether it is concise and free of irrelevant information. Provide suggestions for modifications based on these aspects.

Below are explored queries, the required parameters and the outputs of the tool:

{usage_example}

Your task is to provide suggestions for modifying the tool documentation. The current tool description is:

{tool_description}

Here is an example:

```json
{{
    "Suggestions for tool description": "your suggestions for modifying the tool description."
}}
```

=========

The following is the history of you modifying the tool description:

{History}

Based on the above information, provide more constructive suggestions."""
        with open(analyzer_path, "w", encoding="utf-8") as f:
            f.write(analyzer_template)

    if not rewriter_path.exists():
        rewriter_template = """Your task is to refine the description of the tool based on the queries and parameters you've explored, the outputs provided by the tool, as well as the suggestions provided. You have access to the following tool:

{Tool Description}

Please note that the existing tool documentation may be incomplete or noisy. The revised description should focus solely on the functionalities of the API, omitting any irrelevant details.

Below are explored queries, the required parameters and the outputs of the tool: 

{usage_example}

Based on the feedback provided, here are some guidelines for updating the tool description:

{Suggestions}

Due to the limited number of explorations you can perform, you need to value each opportunity. What aspects of this API would you like to explore next? Please provide some suggestions for your next query generation. Just give a direction of the next exploration, don't give a full example of the exploration directly.

Here is an example: 

```json
{{
    "Rewritten description": "Tool descriptions you have rewritten in the text.",
    "Suggestions for exploring": "your suggestions on generating exploration query in the text."
}}
```

=========

The following is the history of you modifying the tool description:

{History}

Based on the information above, improve the tool descriptions and provide more constructive suggestions for exploration."""
        with open(rewriter_path, "w", encoding="utf-8") as f:
            f.write(rewriter_template)

    # 读取提示词内容并拆分上下文
    with open(explorer_path, "r", encoding="utf-8") as f:
        explorer_template = f.read()
        explorer_prompt, explorer_follow = explorer_template.split("=========")

    with open(analyzer_path, "r", encoding="utf-8") as f:
        analyzer_template = f.read()
        analyzer_prompt, analyzer_follow = analyzer_template.split("=========")

    with open(rewriter_path, "r", encoding="utf-8") as f:
        rewriter_template = f.read()
        rewriter_prompt, rewriter_follow = rewriter_template.split("=========")

    return {
        "explorer": (explorer_prompt, explorer_follow),
        "analyzer": (analyzer_prompt, analyzer_follow),
        "rewriter": (rewriter_prompt, rewriter_follow),
    }


def improve_tool_description_metatool(
    tool_name: str,
    original_description: str,
    tool_records: List[Dict[str, Any]],
    episodes: int = 5,
    model: str = "qwen3-max",
    temperature: float = 0.2,
    max_tokens: int = 2000,
    convergence_threshold: float = 0.75,
    embedding_model: str = "text-embedding-v3",
) -> str:
    """
    使用 DRAFT 方法改进单个工具的描述（适配 MetaTool 的 tool_call_records 结构）。
    实现思路参考 BFCL 的 improve_tool_description_bfcl：
    - 使用历史工具调用记录作为 usage_example
    - Analyzer 生成修改建议
    - Rewriter 迭代重写工具描述
    - 使用 embedding + BLEU 进行收敛判断
    """
    prompts = load_prompts()

    if not tool_records:
        print(f"警告: 工具 {tool_name} 没有工具调用记录，无法使用 DRAFT，只保留原始描述")
        return original_description

    last_description = original_description
    rewrite_history: List[str] = [last_description]

    # 随机选择 episodes 条记录；如果记录不足则用全部
    num_samples = min(episodes, len(tool_records))
    sample_records = random.sample(tool_records, num_samples)

    print(f"\n开始改进工具: {tool_name}")
    print(f"原始描述: {last_description[:100]}...")
    print(f"随机选择了 {len(sample_records)} 条调用记录进行 DRAFT 迭代")

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")
        if episode >= len(sample_records):
            print("调用记录已用完，提前结束 DRAFT 迭代")
            break

        record = sample_records[episode]

        # 尽量与 BFCL 的字段对齐，但兼容 MetaTool 的记录结构
        historical_context = (
            record.get("context")
            or record.get("user_query")
            or record.get("query")
            or ""
        )
        historical_input = (
            record.get("input")
            or record.get("parameters")
            or record.get("args")
            or {}
        )
        historical_output = (
            record.get("output")
            or record.get("response")
            or record.get("tool_output")
            or "Success"
        )

        example_ans = {
            "User Query": historical_context.strip()
            if isinstance(historical_context, str)
            else str(historical_context),
            "Parameters": historical_input if isinstance(historical_input, dict) else {},
            "API_Response": historical_output,
        }

        print(
            f"使用历史调用: 输入参数={historical_input}, 输出={str(historical_output)[:100]}..."
        )

        tool_info_str = json.dumps(
            {
                "tool_name": tool_name,
                "description": last_description,
            },
            ensure_ascii=False,
            indent=2,
        )

        # 1) Analyzer：基于 usage_example + 当前描述 生成修改建议
        analyzer_head, analyzer_follow = prompts["analyzer"]
        suggestion_prompt_temp = analyzer_head.replace(
            "{Tool Description}", tool_info_str
        )
        suggestion_prompt_temp = suggestion_prompt_temp.replace(
            "{usage_example}",
            json.dumps(example_ans, ensure_ascii=False, indent=2),
        )
        suggestion_prompt_temp = suggestion_prompt_temp.replace(
            "{tool_description}", last_description
        )
        if len(rewrite_history) > 1:
            suggestion_prompt_follow = analyzer_follow.replace(
                "{History}", str(rewrite_history)
            )
            suggestion_prompt_temp += suggestion_prompt_follow

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": suggestion_prompt_temp},
        ]
        suggestion_ans = openai_json_request(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )

        # 2) Rewriter：根据建议重写描述
        rewriter_head, rewriter_follow = prompts["rewriter"]
        rewrite_prompt_temp = rewriter_head.replace(
            "{Tool Description}", tool_info_str
        )
        rewrite_prompt_temp = rewrite_prompt_temp.replace(
            "{usage_example}",
            json.dumps(example_ans, ensure_ascii=False, indent=2),
        )
        rewrite_prompt_temp = rewrite_prompt_temp.replace(
            "{Suggestions}",
            suggestion_ans.get("Suggestions for tool description", ""),
        )
        rewrite_prompt_temp = rewrite_prompt_temp.replace(
            "{tool_description}", last_description
        )
        if len(rewrite_history) > 1:
            rewrite_prompt_follow = rewriter_follow.replace(
                "{History}", str(rewrite_history)
            )
            rewrite_prompt_temp += rewrite_prompt_follow

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": rewrite_prompt_temp},
        ]
        rewrite_ans = openai_json_request(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )

        if "Rewritten description" in rewrite_ans:
            rewrite_history.append(rewrite_ans["Rewritten description"])
            last_description = rewrite_ans["Rewritten description"]
        else:
            print(f"警告: 重写响应格式不正确: {rewrite_ans}")
            continue

        print(f"重写后的描述: {last_description[:100]}...")

        # 3) 收敛判断：与上一轮结果比较
        if len(rewrite_history) > 1:
            reference_sentence = rewrite_history[-2]
            candidate_sentence = rewrite_history[-1]
            delta = compute_similarity_and_bleu(
                reference_sentence,
                candidate_sentence,
                embedding_model=embedding_model,
            )
            print(f"相似度 delta: {delta:.4f}")
            if delta > convergence_threshold:
                print("达到收敛条件，停止迭代")
                break

    print(f"\n工具 {tool_name} 改进完成，最终描述: {last_description[:100]}...")
    return last_description


def main():
    parser = argparse.ArgumentParser(
        description="基于 DRAFT 提示与工具调用记录，生成 draft 版工具文档"
    )
    parser.add_argument(
        "--plugin-des-path",
        type=str,
        default="dataset/plugin_des.json",
        help="原始工具描述文件路径（默认为 dataset/plugin_des.json）",
    )
    parser.add_argument(
        "--task2-path",
        type=str,
        required=True,
        help="Task2.json 文件路径（用于提取工具调用记录）",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="dataset/plugin_des_draft.json",
        help="draft 后的工具描述输出路径（默认为 dataset/plugin_des_draft.json）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-max",
        help="用于重写的模型名称（OpenAI/DashScope 兼容接口，例如 qwen3-max）",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="模型名称（可选，用于从原始测试数据文件读取 action_prompt）",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="每个工具的 DRAFT 迭代轮数（默认 5）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM 生成温度（默认 0.2，与 DRAFT 一致）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="LLM 最大输出 token（默认 2000，与 DRAFT 一致）",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.75,
        help="收敛阈值 delta（默认 0.75，与 DRAFT 一致）",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-v3",
        help="用于相似度计算的 embedding 模型（默认 text-embedding-v3）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="最大并发线程数（默认 4）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="每个工具的最大重试次数（默认 3）",
    )
    args = parser.parse_args()

    # 推断目录：
    # - script_dir: 本脚本所在目录 experiment/MetaTool/src/generation
    # - project_root: MetaTool 工程根目录 experiment/MetaTool
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    plugin_des_path = (
        args.plugin_des_path
        if os.path.isabs(args.plugin_des_path)
        else os.path.join(project_root, args.plugin_des_path)
    )
    plugin_des = load_plugin_des(plugin_des_path)

    # 目前 rewrite_template 未使用，保留兼容占位
    _ = load_rewrite_prompt(project_root)

    # 从 Task2.json 中提取 tool_call_records
    task2_path = (
        args.task2_path
        if os.path.isabs(args.task2_path)
        else os.path.join(project_root, args.task2_path)
    )
    
    print(f"正在从 {task2_path} 读取工具调用记录...")
    try:
        tool_call_records = load_tool_call_records_from_task2(
            task2_path, model_name=args.model_name
        )
        print(f"✅ 成功读取 {len(tool_call_records)} 条工具调用记录")
    except Exception as e:
        print(f"❌ 读取工具调用记录失败：{e}")
        print("将仅基于原始描述进行 draft 改写（不使用工具调用经验）")
        tool_call_records = []
    
    # 按工具名分组汇总
    tool_to_records = aggregate_records_by_tool(tool_call_records)
    print(f"共涉及 {len(tool_to_records)} 个不同的工具")

    # 计算输出路径，并尝试加载已有结果，支持断点续跑
    out_path = (
        os.path.join(project_root, args.output_path)
        if not os.path.isabs(args.output_path)
        else args.output_path
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    existing_output: Dict[str, str] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_output = json.load(f)
            print(f"已加载现有输出配置，包含 {len(existing_output)} 个工具，可用于断点续跑")
        except Exception as e:
            print(f"加载现有输出配置失败，将从头开始生成：{e}")
            existing_output = {}

    draft_des: Dict[str, str] = existing_output.copy()

    # 线程安全锁和原子写入
    config_lock = threading.Lock()
    file_lock = threading.Lock()

    def save_config_immediately():
        """立即保存配置到文件（使用原子写入，线程安全）"""
        with file_lock:
            try:
                tmp_path = out_path + ".tmp"
                with config_lock:
                    data_to_save = draft_des.copy()
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, out_path)
            except Exception as e:
                print(f"⚠️ 保存配置时出错: {e}")

    tool_names = list(plugin_des.keys())
    print(f"共 {len(tool_names)} 个工具，将并发生成 draft 描述（DRAFT 迭代 + 断点续跑）。")

    def process_single_tool(tool_name: str) -> tuple[str, bool]:
        """
        处理单个工具，带断点续跑与重试逻辑。

        Returns:
            (tool_name, success)
        """
        original_desc = plugin_des[tool_name]

        # 检查是否已在输出中且已成功改写
        with config_lock:
            if tool_name in draft_des:
                existing_desc = draft_des[tool_name] or ""
                if existing_desc.strip() and existing_desc != original_desc:
                    print(f"✅ 工具 {tool_name} 已经改进成功，跳过")
                    return tool_name, True
                else:
                    print(f"🔄 工具 {tool_name} 描述未改变或为空，重新改进")

        tool_records = tool_to_records.get(tool_name, [])

        for retry in range(args.max_retries):
            try:
                improved_desc = improve_tool_description_metatool(
                    tool_name=tool_name,
                    original_description=original_desc,
                    tool_records=tool_records,
                    episodes=args.episodes,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    convergence_threshold=args.convergence_threshold,
                    embedding_model=args.embedding_model,
                )
                with config_lock:
                    draft_des[tool_name] = improved_desc
                save_config_immediately()
                print(f"💾 工具 {tool_name} 改进完成，已保存到文件")
                return tool_name, True
            except Exception as e:
                if retry < args.max_retries - 1:
                    wait_time = (retry + 1) * 5
                    print(
                        f"⚠️ 工具 {tool_name} 改进失败（第 {retry + 1}/{args.max_retries} 次尝试），"
                        f"{wait_time} 秒后重试: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    print(
                        f"❌ 工具 {tool_name} 改进失败，已重试 {args.max_retries} 次，放弃: {e}"
                    )
                    traceback.print_exc()
                    return tool_name, False

    failed_tools: List[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_tool = {
            executor.submit(process_single_tool, name): name for name in tool_names
        }
        with tqdm(
            total=len(tool_names), desc="Rewriting tool docs (DRAFT, parallel)"
        ) as pbar:
            for future in as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                try:
                    _name, success = future.result()
                    if not success:
                        failed_tools.append(tool_name)
                except Exception as e:
                    print(f"❌ 工具 {tool_name} 处理时发生未捕获的异常: {e}")
                    failed_tools.append(tool_name)
                finally:
                    pbar.update(1)

    # 最后再保存一次，确保所有内容写入
    save_config_immediately()

    if failed_tools:
        print(f"\n⚠️ 有 {len(failed_tools)} 个工具改进失败: {failed_tools}")
    print(f"\n所有 draft 工具文档已生成并保存到：{out_path}")


if __name__ == "__main__":
    main()

