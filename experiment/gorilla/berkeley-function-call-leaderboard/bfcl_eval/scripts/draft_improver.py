"""
DRAFT 工具文档改进脚本 - 适配 BFCL 数据集
基于 DRAFT 方法，通过探索-分析-重写的迭代循环改进工具描述

注意：BFCL 数据集的工具结构与 API-Bank 不同，本脚本适配 BFCL 的 config.json 格式
"""
import json
import os
import traceback
import re
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from pathlib import Path

import tqdm
import numpy as np
from openai import OpenAI
from openai import RateLimitError
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 初始化 OpenAI 客户端
openai_client = None


class QwenOpenAIWrapper:
    """使用百炼 DashScope OpenAI 兼容接口的 Qwen 封装，保持与 OpenAI client 相同返回结构。
    只使用 Qwen 接口，默认使用 qwen3-max 模型。
    """
    def __init__(self, model="qwen3-max", api_key=None, base_url=None):
        self.model = model
        # 只使用 Qwen/DashScope API
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or "your api key"
        self.base_url = base_url or os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call(self, messages, **kwargs):
        """单个请求的公共接口，带 429 错误重试逻辑"""
        # 构建请求参数
        call_kwargs = {
            "model": self.model,
            "messages": messages,
        }
        # 用户传入的 kwargs 会覆盖默认值
        call_kwargs.update(kwargs)
        
        # 重试逻辑：第一次 429 错误等待 30 秒，第二次等待 60 秒
        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(**call_kwargs)
                # 兼容 OpenAI 返回格式
                return {
                    "choices": [
                        {"message": {"content": completion.choices[0].message.content}}
                    ]
                }
            except RateLimitError as e:
                if attempt == 0:
                    wait_time = 30
                    print(f"遇到 429 错误（速率限制），等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                elif attempt == 1:
                    wait_time = 60
                    print(f"再次遇到 429 错误，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"QwenOpenAIWrapper 调用失败: 429 错误重试 3 次后仍失败")
                    raise
            except Exception as e:
                print(f"QwenOpenAIWrapper 调用失败: {e}")
                raise

def init_openai_client():
    """初始化 OpenAI 客户端（使用 Qwen 接口）"""
    global openai_client
    if openai_client is None:
        # 只使用 Qwen/DashScope API
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or "your api key"
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        openai_client = OpenAI(api_key=api_key, base_url=base_url)
    return openai_client


def openai_embedding(text: str) -> List[float]:
    """获取文本的嵌入向量（使用 Qwen 接口），带 429 错误重试逻辑"""
    client = init_openai_client()
    # 重试逻辑：第一次 429 错误等待 30 秒，第二次等待 60 秒
    for attempt in range(3):
        try:
            response = client.embeddings.create(
                model="text-embedding-v3",
                input=text
            )
            return response.data[0].embedding
        except RateLimitError as e:
            if attempt == 0:
                wait_time = 30
                print(f"嵌入生成遇到 429 错误（速率限制），等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            elif attempt == 1:
                wait_time = 60
                print(f"嵌入生成再次遇到 429 错误，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"嵌入生成失败: 429 错误重试 3 次后仍失败")
                return [0.0] * 1536
        except Exception as e:
            print(f"嵌入生成失败: {e}")
            return [0.0] * 1536


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def openai_response(messages: List[Dict], model: str = "qwen3-max", temperature: float = 0.2, max_tokens: int = 2000) -> Dict:
    """调用 LLM 并解析 JSON 响应，带 429 错误重试逻辑"""
    client = init_openai_client()
    
    # 重试逻辑：第一次 429 错误等待 30 秒，第二次等待 60 秒
    content = None
    for attempt in range(3):
        try:
            if not client:
                # 使用 QwenOpenAIWrapper 作为备选（它内部已经有重试逻辑）
                wrapper = QwenOpenAIWrapper(model=model)
                response = wrapper.call(messages, temperature=temperature, max_tokens=max_tokens)
                content = response['choices'][0]['message']['content']
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
            break  # 成功获取响应，退出重试循环
        except RateLimitError as e:
            print(f"API 调用遇到 429 错误（速率限制）: {e}", traceback.format_exc())
            if attempt == 0:
                wait_time = 30
                print(f"API 调用遇到 429 错误（速率限制），等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            elif attempt == 1:
                wait_time = 60
                print(f"API 调用再次遇到 429 错误，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"API 调用失败: 429 错误重试 3 次后仍失败")
                raise
        except Exception as e:
            # 非 429 错误直接抛出
            print(f"API 调用失败: {e}")
            raise
    
    if content is None:
        raise Exception("无法获取 API 响应内容")
    
    # 清理并解析 JSON
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


def compute_similarity_and_bleu(reference_sentence: str, candidate_sentence: str) -> float:
    """
    计算参考句子和候选句子的相似度，严格对齐 DRAFT 原文：
    - 使用 embedding 余弦相似度
    - 使用 BLEU（带平滑）
    - 最终分数为二者的平均值
    """
    try:
        # 1) embedding 相似度
        reference_embedding = openai_embedding(reference_sentence)
        candidate_embedding = openai_embedding(candidate_sentence)
        similarity = cosine_similarity(reference_embedding, candidate_embedding)

        # 2) BLEU 分数（与 DRAFT 一致：按词分 token，并使用平滑）
        reference = [reference_sentence.lower().split()]
        candidate = candidate_sentence.lower().split()
        smoothie = SmoothingFunction().method4
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)

        # 3) 综合得分
        delta = (bleu_score + similarity) / 2
        return delta
    except Exception as e:
        print(f"相似度计算失败: {e}")
        return 0.0


def load_prompts():
    """加载 DRAFT 提示词模板"""
    # 使用与 API-Bank 相同的提示词模板
    prompts_dir = Path("/Users/xiaocan/Toolmem/experiment/api-bank/draft_prompts")
    prompts_dir.mkdir(exist_ok=True, parents=True)
    
    explorer_path = prompts_dir / "Explorer.txt"
    analyzer_path = prompts_dir / "Analyzer.txt"
    rewriter_path = prompts_dir / "Rewriter.txt"
    
    # 如果提示词文件不存在，使用默认模板
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
        
        with open(explorer_path, 'w', encoding='utf-8') as f:
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
        
        with open(analyzer_path, 'w', encoding='utf-8') as f:
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
        
        with open(rewriter_path, 'w', encoding='utf-8') as f:
            f.write(rewriter_template)
    
    # 读取提示词
    with open(explorer_path, 'r', encoding='utf-8') as f:
        example_prompt_template = f.read()
        example_prompt, example_prompt_follow = example_prompt_template.split('=========')
    
    with open(analyzer_path, 'r', encoding='utf-8') as f:
        suggestion_prompt_template = f.read()
        suggestion_prompt, suggestion_prompt_follow = suggestion_prompt_template.split('=========')
    
    with open(rewriter_path, 'r', encoding='utf-8') as f:
        rewrite_prompt_template = f.read()
        rewrite_prompt, rewrite_prompt_follow = rewrite_prompt_template.split('=========')
    
    return {
        'explorer': (example_prompt, example_prompt_follow),
        'analyzer': (suggestion_prompt, suggestion_prompt_follow),
        'rewriter': (rewrite_prompt, rewrite_prompt_follow),
    }


def improve_tool_description_bfcl(
    tool_hash: str,
    tool_config: dict,
    tool_call_results: List[dict],
    episodes: int = 5,
    model: str = "qwen3-max",
    temperature: float = 0.2,
    max_tokens: int = 2000
) -> Dict[str, Any]:
    """
    使用 DRAFT 方法改进单个工具的描述（BFCL 格式）
    
    Args:
        tool_hash: 工具hash名称（例如："generate_human_image_0a84a8a9e2c5987d14d71e030cee9d73"）
        tool_config: 工具配置字典，包含 description 和 parameters
        tool_call_results: 该工具的历史调用结果列表（用于模拟 API 响应）
        episodes: 迭代轮数
        model: 使用的模型
        temperature: 生成温度
        max_tokens: 最大 token 数
    
    Returns:
        改进后的工具配置字典
    """
    prompts = load_prompts()
    
    # 获取原始描述
    original_description = tool_config.get("description", "")
    
    # 初始化
    last_description = original_description
    rewrite_history = [last_description]
    explored_queries = []
    explored_queries_embeddings = []
    explored_examples = []
    suggestions = []
    suggestion_from_rewrite_agent = ''
    
    print(f"\n开始改进工具: {tool_hash}")
    print(f"原始描述: {last_description[:100]}...")
    
    # 使用历史调用结果进行探索：随机选择episodes个历史调用记录，然后按顺序使用
    if not tool_call_results:
        print(f"警告: 工具 {tool_hash} 没有历史调用结果，无法进行改进")
        return tool_config
    
    # 随机选择episodes个历史调用记录（如果历史记录数量少于episodes，则使用所有记录）
    num_samples = min(episodes, len(tool_call_results))
    sample_results = random.sample(tool_call_results, num_samples)
    
    print(f"随机选择了 {len(sample_results)} 个历史调用记录进行探索")
    
    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")
        
        # 如果历史调用记录不够，跳过剩余的episodes
        if episode >= len(sample_results):
            print(f"历史调用记录已用完，停止探索")
            break
        
        # 1. Explorer: 使用历史调用记录（每一轮使用一个历史调用）
        historical_sample = sample_results[episode]
        
        # 从历史记录中提取用户查询和参数
        historical_context = historical_sample.get('context', '')
        historical_input = historical_sample.get('input', {})
        historical_output = historical_sample.get('output', 'Success')
        
        # 构建探索示例（使用历史调用数据）
        example_ans = {
            'User Query': historical_context.strip() if historical_context else f"Example query for {tool_hash}",
            'Parameters': historical_input if isinstance(historical_input, dict) else {}
        }
        
        # 记录探索的查询（用于后续的相似度检查，虽然这里不再需要检查，但保持结构一致）
        explored_queries.append(example_ans['User Query'])
        if example_ans['User Query']:
            explored_queries_embeddings.append(openai_embedding(example_ans['User Query']))
        
        # 2. API 响应：直接使用历史调用的输出
        example_ans['API_Response'] = historical_output
        explored_examples.append(example_ans)
        
        print(f"使用历史调用: 输入参数={historical_input}, 输出={str(historical_output)[:100]}...")
        
        # 构建工具信息（用于 Analyzer 和 Rewriter 的提示词）
        tool_info_str = json.dumps({
            "description": last_description,
            "parameters": tool_config.get("parameters", {})
        }, ensure_ascii=False, indent=2)
        
        # 3. Analyzer: 分析并生成建议
        suggestion_prompt_temp = prompts['analyzer'][0].replace('{Tool Description}', tool_info_str)
        suggestion_prompt_temp = suggestion_prompt_temp.replace('{usage_example}', json.dumps(example_ans, ensure_ascii=False, indent=2))
        suggestion_prompt_temp = suggestion_prompt_temp.replace('{tool_description}', last_description)
        
        if len(rewrite_history) > 1:
            suggestion_prompt_follow_temp = prompts['analyzer'][1].replace('{History}', str(rewrite_history))
            suggestion_prompt_temp += suggestion_prompt_follow_temp
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": suggestion_prompt_temp}
        ]
        suggestion_ans = openai_response(messages, model, temperature, max_tokens)
        suggestions.append(suggestion_ans)
        
        # 4. Rewriter: 重写描述
        rewrite_prompt_temp = prompts['rewriter'][0].replace('{Tool Description}', tool_info_str)
        rewrite_prompt_temp = rewrite_prompt_temp.replace('{usage_example}', json.dumps(example_ans, ensure_ascii=False, indent=2))
        rewrite_prompt_temp = rewrite_prompt_temp.replace('{Suggestions}', suggestion_ans.get('Suggestions for tool description', ''))
        rewrite_prompt_temp = rewrite_prompt_temp.replace('{tool_description}', last_description)
        
        if len(rewrite_history) > 1:
            rewrite_prompt_follow_temp = prompts['rewriter'][1].replace('{History}', str(rewrite_history))
            rewrite_prompt_temp += rewrite_prompt_follow_temp
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": rewrite_prompt_temp}
        ]
        rewrite_ans = openai_response(messages, model, temperature, max_tokens)
        
        if 'Rewritten description' in rewrite_ans:
            rewrite_history.append(rewrite_ans['Rewritten description'])
            last_description = rewrite_ans['Rewritten description']
            suggestion_from_rewrite_agent = rewrite_ans.get('Suggestions for exploring', '')
        else:
            print(f"警告: 重写响应格式不正确: {rewrite_ans}")
            continue
        
        print(f"重写后的描述: {last_description[:100]}...")
        
        # 5. 检查收敛
        if len(rewrite_history) > 1:
            reference_sentence = rewrite_history[-2]
            candidate_sentence = rewrite_history[-1]
            delta = compute_similarity_and_bleu(reference_sentence, candidate_sentence)
            print(f"相似度 delta: {delta:.4f}")
            if delta > 0.75:
                print("达到收敛条件，停止迭代")
                break
    
    # 返回改进后的配置
    improved_config = tool_config.copy()
    improved_config['description'] = rewrite_history[-1]
    
    print(f"\n改进完成，最终描述: {improved_config['description'][:100]}...")
    
    return improved_config


def improve_config_with_draft(
    config_path: Path,
    tool_call_results_path: Path,
    output_config_path: Path,
    episodes: int = 5,
    model: str = "qwen3-max",
    tool_hashes: Optional[List[str]] = None,
    max_workers: int = 4,
    max_retries: int = 3
) -> Dict[str, dict]:
    """
    使用 DRAFT 方法改进 config.json 中的所有工具描述（支持多线程并发）
    
    Args:
        config_path: 原始 config.json 文件路径
        tool_call_results_path: tool_call_results.jsonl 文件路径（包含历史调用结果）
        output_config_path: 输出改进后的 config.json 文件路径
        episodes: 每个工具的迭代轮数
        model: 使用的模型
        tool_hashes: 要改进的工具hash列表，如果为None则改进所有工具
        max_workers: 最大并发线程数
        max_retries: 每个工具的最大重试次数
    
    Returns:
        改进后的配置字典
    """
    # 初始化 OpenAI 客户端（使用 Qwen 接口，有默认 API key）
    init_openai_client()
    
    # 加载原始配置
    with open(config_path, 'r', encoding='utf-8') as f:
        original_config = json.load(f)
    
    # 加载工具调用结果
    tool_call_results_by_tool = {}
    if tool_call_results_path.exists():
        with open(tool_call_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line)
                        tool_name = result.get('tool_name', '')
                        if tool_name:
                            if tool_name not in tool_call_results_by_tool:
                                tool_call_results_by_tool[tool_name] = []
                            tool_call_results_by_tool[tool_name].append(result)
                    except json.JSONDecodeError:
                        continue
    
    # 加载已有的输出配置（如果存在），用于检查哪些工具已经改进成功
    existing_output_config = {}
    if output_config_path.exists():
        try:
            with open(output_config_path, 'r', encoding='utf-8') as f:
                existing_output_config = json.load(f)
            print(f"已加载现有输出配置，包含 {len(existing_output_config)} 个工具")
        except Exception as e:
            print(f"加载现有输出配置失败: {e}，将重新生成所有工具")
    
    # 确定要处理的工具
    tools_to_improve = tool_hashes if tool_hashes else list(original_config.keys())
    
    # 从已有输出配置开始（如果存在），这样可以保留已成功改进的工具
    improved_config = existing_output_config.copy() if existing_output_config else {}
    
    # 用于线程安全的锁
    config_lock = threading.Lock()
    file_lock = threading.Lock()
    
    def save_config_immediately():
        """立即保存配置到文件（使用原子写入，线程安全）"""
        with file_lock:
            try:
                # 使用临时文件确保原子写入
                temp_path = output_config_path.with_suffix('.tmp')
                with config_lock:
                    config_to_save = improved_config.copy()
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(config_to_save, f, indent=4, ensure_ascii=False)
                # 原子性地替换原文件
                temp_path.replace(output_config_path)
            except Exception as e:
                print(f"⚠️ 保存配置时出错: {e}")
    
    def process_single_tool(tool_hash: str) -> tuple[str, Optional[dict], bool]:
        """
        处理单个工具，带重试机制
        
        Returns:
            (tool_hash, improved_config or None, success)
        """
        if tool_hash not in original_config:
            print(f"⚠️ 工具 {tool_hash} 不在配置中，跳过")
            return (tool_hash, None, False)
        
        # 检查该工具是否已经改进成功（描述是否改变）
        # 需要检查 existing_output_config 和 improved_config（可能其他线程已经完成）
        with config_lock:
            # 先检查 improved_config（可能其他线程刚完成）
            if tool_hash in improved_config:
                original_description = original_config[tool_hash].get('description', '')
                improved_description = improved_config[tool_hash].get('description', '')
                if original_description != improved_description and improved_description.strip():
                    print(f"✅ 工具 {tool_hash} 已由其他线程完成，跳过")
                    return (tool_hash, None, True)
            
            # 再检查 existing_output_config（从文件加载的）
            if tool_hash in existing_output_config:
                original_description = original_config[tool_hash].get('description', '')
                existing_description = existing_output_config[tool_hash].get('description', '')
                # 如果描述已经改变（改进成功），则跳过
                if original_description != existing_description and existing_description.strip():
                    print(f"✅ 工具 {tool_hash} 已经改进成功，跳过（原描述长度: {len(original_description)}, 新描述长度: {len(existing_description)}）")
                    return (tool_hash, None, True)  # 已成功，不需要处理
                else:
                    print(f"🔄 工具 {tool_hash} 描述未改变或为空，重新改进")
        
        # 重试机制
        for retry in range(max_retries):
            try:
                tool_config = original_config[tool_hash]
                tool_results = tool_call_results_by_tool.get(tool_hash, [])
                
                improved_tool_config = improve_tool_description_bfcl(
                    tool_hash=tool_hash,
                    tool_config=tool_config,
                    tool_call_results=tool_results,
                    episodes=episodes,
                    model=model
                )
                
                # 成功改进，立即保存
                with config_lock:
                    improved_config[tool_hash] = improved_tool_config
                save_config_immediately()
                print(f"💾 工具 {tool_hash} 改进完成，已保存到文件")
                return (tool_hash, improved_tool_config, True)
                
            except Exception as e:
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 5  # 递增等待时间：5秒、10秒、15秒
                    print(f"⚠️ 工具 {tool_hash} 改进失败（第 {retry + 1}/{max_retries} 次尝试），{wait_time}秒后重试: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"❌ 工具 {tool_hash} 改进失败，已重试 {max_retries} 次，放弃: {e}")
                    import traceback
                    traceback.print_exc()
                    return (tool_hash, None, False)
        
        return (tool_hash, None, False)
    
    # 使用线程池并发处理
    failed_tools = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_tool = {
            executor.submit(process_single_tool, tool_hash): tool_hash 
            for tool_hash in tools_to_improve
        }
        
        # 使用 tqdm 显示进度
        with tqdm.tqdm(total=len(tools_to_improve), desc="改进工具描述") as pbar:
            for future in as_completed(future_to_tool):
                tool_hash = future_to_tool[future]
                try:
                    result_hash, result_config, success = future.result()
                    if not success and result_config is None:
                        failed_tools.append(result_hash)
                except Exception as e:
                    print(f"❌ 工具 {tool_hash} 处理时发生未捕获的异常: {e}")
                    failed_tools.append(tool_hash)
                finally:
                    pbar.update(1)
    
    # 最后再保存一次，确保所有内容都已写入
    save_config_immediately()
    
    if failed_tools:
        print(f"\n⚠️ 有 {len(failed_tools)} 个工具改进失败: {failed_tools}")
    print(f"\n所有改进完成，结果已保存到: {output_config_path}")
    return improved_config


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='使用 DRAFT 方法改进 BFCL 工具描述（支持多线程并发）')
    parser.add_argument('--config', type=str, required=True, help='原始 config.json 文件路径')
    parser.add_argument('--tool-call-results', type=str, required=True, help='tool_call_results.jsonl 文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出改进后的 config.json 文件路径')
    parser.add_argument('--episodes', type=int, default=5, help='每个工具的迭代轮数')
    parser.add_argument('--model', type=str, default='qwen3-max', help='使用的模型')
    parser.add_argument('--tools', type=str, nargs='+', help='要改进的工具hash列表（如果不指定则改进所有工具）')
    parser.add_argument('--max-workers', type=int, default=4, help='最大并发线程数（默认：4）')
    parser.add_argument('--max-retries', type=int, default=3, help='每个工具的最大重试次数（默认：3）')
    
    args = parser.parse_args()
    
    improve_config_with_draft(
        config_path=Path(args.config),
        tool_call_results_path=Path(args.tool_call_results),
        output_config_path=Path(args.output),
        episodes=args.episodes,
        model=args.model,
        tool_hashes=args.tools,
        max_workers=args.max_workers,
        max_retries=args.max_retries
    )

