"""
DRAFT 工具文档改进脚本 - 适配 API-Bank
基于 DRAFT 方法，通过探索-分析-重写的迭代循环改进工具描述
"""
import json
import os
import random
import re
from typing import Dict, List, Any, Optional

import tqdm
import numpy as np
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from tool_manager import ToolManager
from utils import QwenOpenAIWrapper

# 初始化 OpenAI 客户端（如果需要）
openai_client = None

def init_openai_client():
    """初始化 OpenAI 客户端"""
    global openai_client
    if openai_client is None:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if api_key:
            openai_client = OpenAI(api_key=api_key, base_url=base_url)
    return openai_client


def openai_embedding(text: str) -> List[float]:
    """获取文本的嵌入向量"""
    client = init_openai_client()
    if not client:
        raise ValueError("OpenAI client not initialized. Please set DASHSCOPE_API_KEY or OPENAI_API_KEY")
    
    try:
        # 使用 text-embedding-ada-002 或兼容的模型
        response = client.embeddings.create(
            model="text-embedding-v3",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"嵌入生成失败: {e}")
        # 如果失败，返回零向量
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


def openai_response(messages: List[Dict], model: str = "qwen-plus", temperature: float = 0.2, max_tokens: int = 2000) -> Dict:
    """调用 LLM 并解析 JSON 响应"""
    client = init_openai_client()
    if not client:
        # 使用 QwenOpenAIWrapper 作为备选
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
    prompts_dir = os.path.join(os.path.dirname(__file__), "draft_prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    
    # 创建默认提示词文件（如果不存在）
    explorer_path = os.path.join(prompts_dir, "Explorer.txt")
    analyzer_path = os.path.join(prompts_dir, "Analyzer.txt")
    rewriter_path = os.path.join(prompts_dir, "Rewriter.txt")
    
    if not os.path.exists(explorer_path):
        # 创建适配 API-Bank 的提示词
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
    
    if not os.path.exists(analyzer_path):
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
    
    if not os.path.exists(rewriter_path):
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


def improve_api_description(
    api_name: str,
    tool_manager: ToolManager,
    episodes: int = 5,
    model: str = "qwen-plus",
    temperature: float = 0.2,
    max_tokens: int = 2000
) -> Dict[str, Any]:
    """
    使用 DRAFT 方法改进单个 API 的描述
    
    Args:
        api_name: API 名称
        tool_manager: ToolManager 实例
        episodes: 迭代轮数
        model: 使用的模型
        temperature: 生成温度
        max_tokens: 最大 token 数
    
    Returns:
        改进后的 API 描述信息
    """
    prompts = load_prompts()
    
    # 获取原始 API 信息
    api_info = tool_manager.get_api_by_name(api_name)
    original_description = json.loads(tool_manager.get_api_description(api_name))
    
    # 初始化
    last_description = original_description['description']
    rewrite_history = [last_description]
    explored_queries = []
    explored_queries_embeddings = []
    explored_examples = []
    suggestions = []
    suggestion_from_rewrite_agent = ''
    
    print(f"\n开始改进 API: {api_name}")
    print(f"原始描述: {last_description}")
    
    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")
        
        # 1. Explorer: 生成探索查询
        tool_info_str = json.dumps(original_description, ensure_ascii=False, indent=2)
        # 更新描述为最新版本
        current_tool_info = original_description.copy()
        current_tool_info['description'] = last_description
        tool_info_str = json.dumps(current_tool_info, ensure_ascii=False, indent=2)
        
        explore_prompt = prompts['explorer'][0].replace('{Tool Description}', tool_info_str)
        
        if len(explored_queries) > 0:
            explore_prompt_follow = prompts['explorer'][1].replace('{Explored queries}', str(explored_queries))
            explore_prompt_follow = explore_prompt_follow.replace('{Suggestions}', suggestion_from_rewrite_agent)
            explore_prompt = explore_prompt + explore_prompt_follow
            
            # 尝试生成不同的查询（避免相似）
            for t in range(3):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": explore_prompt}
                ]
                example_ans = openai_response(messages, model, temperature, max_tokens)
                
                if 'User Query' not in example_ans:
                    print(f"警告: 响应格式不正确: {example_ans}")
                    break
                
                cur_embedding = openai_embedding(example_ans['User Query'])
                similarity = [cosine_similarity(emb, cur_embedding) for emb in explored_queries_embeddings]
                if all(sim < 0.9 for sim in similarity):
                    break
                else:
                    explore_prompt += f"\nYour last generate query '{example_ans['User Query']}' is too similar to the previous ones. Please generate a different query."
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": explore_prompt}
            ]
            example_ans = openai_response(messages, model, temperature, max_tokens)
        
        if 'User Query' not in example_ans or 'Parameters' not in example_ans:
            print(f"警告: 无法生成有效的探索查询: {example_ans}")
            continue
        
        explored_queries.append(example_ans['User Query'])
        explored_queries_embeddings.append(openai_embedding(example_ans['User Query']))
        
        # 2. 实际调用 API 获取响应
        try:
            parameters = example_ans.get('Parameters', {})
            # 对参数做简单的安全截断，避免 LLM 生成极长字符串导致某些工具（如 Calculator）计算过慢
            MAX_PARAM_STR_LEN = 256
            safe_parameters = {}
            for k, v in parameters.items():
                if isinstance(v, str) and len(v) > MAX_PARAM_STR_LEN:
                    print(f"[WARN] parameter '{k}' is too long (len={len(v)}), truncate to {MAX_PARAM_STR_LEN} chars")
                    safe_parameters[k] = v[:MAX_PARAM_STR_LEN]
                else:
                    safe_parameters[k] = v

            # 调用 API（api_call 使用 **kwargs，直接传递参数字典）
            if safe_parameters:
                result = tool_manager.api_call(api_name, **safe_parameters)
            else:
                # 如果没有参数，尝试调用（某些 API 可能不需要参数）
                result = tool_manager.api_call(api_name)
            example_ans['API_Response'] = result
        except Exception as e:
            print(f"API call failure: {e}")
            import traceback
            traceback.print_exc()
            example_ans['API_Response'] = {"error": str(e)}
        
        explored_examples.append(example_ans)
        
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
    
    # 返回改进后的描述
    improved_description = original_description.copy()
    improved_description['description'] = rewrite_history[-1]
    
    print(f"\n改进完成，最终描述: {improved_description['description'][:100]}...")
    
    return improved_description


def improve_all_apis(
    output_file: str = "draft_improved_descriptions.json",
    episodes: int = 5,
    model: str = "qwen-plus",
    api_names: Optional[List[str]] = None
):
    """
    改进所有 API 的描述
    
    Args:
        output_file: 输出文件路径
        episodes: 每个 API 的迭代轮数
        model: 使用的模型
        api_names: 要改进的 API 名称列表，如果为 None 则改进所有 API
    """
    tool_manager = ToolManager()
    
    if api_names is None:
        api_names = tool_manager.list_all_apis()
    
    improved_descriptions = {}
    
    for api_name in tqdm.tqdm(api_names):
        try:
            improved_desc = improve_api_description(
                api_name=api_name,
                tool_manager=tool_manager,
                episodes=episodes,
                model=model
            )
            improved_descriptions[api_name] = improved_desc
        except Exception as e:
            print(f"改进 {api_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            # 保存原始描述作为备选
            original_desc = json.loads(tool_manager.get_api_description(api_name))
            improved_descriptions[api_name] = original_desc
    
    # 保存改进后的描述
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(improved_descriptions, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有改进完成，结果已保存到: {output_file}")
    return improved_descriptions


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='使用 DRAFT 方法改进 API-Bank 工具描述')
    parser.add_argument('--output', type=str, default='draft_improved_descriptions.json',
                        help='输出文件路径')
    parser.add_argument('--episodes', type=int, default=5,
                        help='每个 API 的迭代轮数')
    parser.add_argument('--model', type=str, default='qwen-plus',
                        help='使用的模型')
    parser.add_argument('--api', type=str, nargs='+',
                        help='要改进的 API 名称列表，如果不指定则改进所有 API')
    
    args = parser.parse_args()
    
    improve_all_apis(
        output_file=args.output,
        episodes=args.episodes,
        model=args.model,
        api_names=args.api
    )

