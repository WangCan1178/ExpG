from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    wait_exponential,
    retry_if_exception,
)

import requests
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from requests.exceptions import ConnectionError, Timeout
import os
import time
from typing import List, Dict, Any, Optional, Callable
from openai import OpenAI
from openai import RateLimitError, APIConnectionError, APIError, APITimeoutError

class RateLimitReached(Exception):
    pass

class OfficialError(Exception):
    pass

class RecoverableError(Exception):
    pass

class KeysBusyError(Exception):
    pass

class ChatGPTWrapper:
    def __init__(self, api_key='', proxies=None) -> None:
        # Set the request parameters
        self.url = 'https://api.openai.com/v1/chat/completions'
        # Set the header
        self.header = {
            "Content-Type": "application/json",
            "Authorization": 'Bearer {}'.format(api_key)
        }
        self.proxies = proxies

    @retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((RateLimitReached, RecoverableError, OfficialError, ConnectionError)))
    def call(self, messages, **kwargs):
        query = {
            "model": "gpt-3.5-turbo",
            "messages": messages
        }
        query.update(kwargs)

        # Make the request
        if self.proxies:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query), proxies=self.proxies)
        else:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query))
        response = response.json()
        if 'error' in response and 'Rate limit reached' in response['error']['message']:
            raise RateLimitReached()
        elif 'choices' in response:
            return response
        else:
            if 'error' in response:
                print(response['error']['message'])
                if response['error']['message'] == 'The server had an error while processing your request. Sorry about that!':
                    raise RecoverableError(response['error']['message'])
                else:
                    raise OfficialError(response['error']['message'])
            else:
                raise Exception('Unknown error occured. Json: {}'.format(response))
            
            
class DavinciWrapper:
    def __init__(self, api_key='') -> None:
        # Set the request parameters
        self.url = 'https://api.openai.com/v1/completions'
        # Set the header
        self.header = {
            "Content-Type": "application/json",
            "Authorization": 'Bearer {}'.format(api_key)
        }

    @retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((RateLimitReached, RecoverableError, OfficialError, ConnectionError)))
    def call(self, messages, **kwargs):
        # messages to prompt
        prompt = ''
        for message in messages:
            prompt += message['role'] + ': ' + message['content'] + '\n'

        query = {
            "model": "davinci",
            "prompt": prompt
        }
        query.update(kwargs)

        # Make the request
        response = requests.post(self.url, headers=self.header, data=json.dumps(query))
        response = response.json()
        if 'error' in response and 'Rate limit reached' in response['error']['message']:
            raise RateLimitReached()
        elif 'choices' in response:
            return response
        else:
            if 'error' in response:
                print(response['error']['message'])
                if response['error']['message'] == 'The server had an error while processing your request. Sorry about that!':
                    raise RecoverableError(response['error']['message'])
                else:
                    raise OfficialError(response['error']['message'])
            else:
                raise Exception('Unknown error occured. Json: {}'.format(response))

class GPT4Wrapper(ChatGPTWrapper):
    @retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((RateLimitReached, RecoverableError, OfficialError, ConnectionError)))
    def call(self, messages, **kwargs):
        query = {
            "model": "gpt-4-0314",
            "messages": messages
        }
        query.update(kwargs)

        # Make the request
        if self.proxies:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query), proxies=self.proxies)
        else:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query))        
        response = response.json()
        if 'error' in response and 'Rate limit reached' in response['error']['message']:
            raise RateLimitReached()
        elif 'choices' in response:
            return response
        else:
            if 'error' in response:
                print(response['error']['message'])
                if response['error']['message'] == 'The server had an error while processing your request. Sorry about that!':
                    raise RecoverableError(response['error']['message'])
                else:
                    raise OfficialError(response['error']['message'])
            else:
                raise Exception('Unknown error occured. Json: {}'.format(response))


class QwenOpenAIWrapper:
    """使用百炼 DashScope OpenAI 兼容接口的 Qwen 封装，保持与 ChatGPTWrapper 相同返回结构。
    支持并行请求和自动重试机制。
    """
    # export DASHSCOPE_API_KEY=your api key
    def __init__(self, model="qwen-plus", api_key=None, base_url=None, max_workers=8, max_retries=3):
        self.model = model
        model_name_lower = model.lower() if model else ""
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        # 根据模型名自动选择 API key 和 base_url
        if "gpt" in model_name_lower:
            # 如果模型名包含 gpt，优先使用环境变量或参数，否则使用默认值
            self.api_key = api_key or os.getenv("GPT_API_KEY")
            # 支持从环境变量读取 base_url，默认使用 api.gpt.ge
            self.base_url = base_url or os.getenv("GPT_API_BASE", "https://api.gpt.ge/v1/")
        elif "qwen" in model_name_lower or "deepseek" in model_name_lower:
            # 如果模型名包含 qwen 或 deepseek，使用 DashScope API
            self.api_key = api_key or os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY"))
            self.base_url = base_url or os.getenv(
                "QWEN_API_BASE",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        else:
            # 默认使用 DashScope API
            self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            self.base_url = base_url or os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        
        if not self.api_key:
            raise ValueError("API key 未设置，请设置 DASHSCOPE_API_KEY 或 QWEN_API_KEY 环境变量")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _should_retry(self, exception):
        """判断是否应该重试该异常"""
        if isinstance(exception, (RateLimitError, APIConnectionError, APITimeoutError)):
            return True
        # 检查是否是速率限制相关的错误消息
        error_msg = str(exception).lower()
        rate_limit_keywords = ['rate limit', 'rate_limit', '429', 'too many requests', 'quota']
        if any(keyword in error_msg for keyword in rate_limit_keywords):
            return True
        # 检查是否是临时性错误
        recoverable_keywords = ['timeout', 'connection', 'server error', '503', '502', '500']
        if any(keyword in error_msg for keyword in recoverable_keywords):
            return True
        return False

    def _call_single_with_retry(self, messages, **kwargs):
        """单个请求的内部方法，带重试机制"""
        max_attempts = self.max_retries + 1  # 初始尝试 + 重试次数
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                # 构建请求参数
                call_kwargs = {
                    "model": self.model,
                    "temperature": 0.5,
                    "messages": messages,
                }
                # 对于非 GPT 模型，添加 extra_body（GPT API 可能不支持）
                # if "gpt" not in self.model.lower():
                #     call_kwargs["extra_body"] = {"enable_thinking": False}
                # 用户传入的 kwargs 会覆盖默认值
                call_kwargs.update(kwargs)
                
                completion = self.client.chat.completions.create(**call_kwargs)
                # 兼容 evaluator 读取方式
                return {
                    "choices": [
                        {"message": {"content": completion.choices[0].message.content}}
                    ]
                }
            except Exception as e:
                last_exception = e
                # 检查是否应该重试
                if self._should_retry(e) and attempt < max_attempts - 1:
                    # 计算等待时间（指数退避）
                    wait_time = min(2 ** attempt, 60)  # 最多等待60秒
                    logging.warning(
                        f"请求失败（尝试 {attempt + 1}/{max_attempts}），"
                        f"{wait_time}秒后重试: {type(e).__name__}: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    # 不可恢复的错误或已达到最大重试次数
                    if attempt < max_attempts - 1:
                        logging.error(f"请求失败，不可恢复: {type(e).__name__}: {str(e)}")
                    else:
                        logging.error(
                            f"请求失败，已达到最大重试次数（{max_attempts}）: "
                            f"{type(e).__name__}: {str(e)}"
                        )
                    raise
        
        # 如果所有重试都失败，抛出最后的异常
        if last_exception:
            raise last_exception

    def call(self, messages, **kwargs):
        """单个请求的公共接口"""
        return self._call_single_with_retry(messages, **kwargs)

    def call_batch(self, messages_list: List[List[Dict[str, str]]], **kwargs) -> List[Dict[str, Any]]:
        """
        并行批量请求接口
        
        Args:
            messages_list: 消息列表的列表，每个元素是一个 messages 列表
            **kwargs: 传递给每个请求的额外参数
            
        Returns:
            List[Dict]: 每个请求的结果列表，保持与 messages_list 相同的顺序
            如果某个请求失败，对应的位置会包含错误信息
        """
        if not messages_list:
            return []
        
        results = [None] * len(messages_list)
        
        def process_single(index: int, messages: List[Dict[str, str]]) -> tuple:
            """处理单个请求的辅助函数"""
            try:
                result = self._call_single_with_retry(messages, **kwargs)
                return index, result, None
            except Exception as e:
                error_msg = f"[ERROR] 请求失败: {type(e).__name__}: {str(e)}"
                logging.error(f"批量请求中索引 {index} 失败: {error_msg}")
                # 返回错误信息而不是抛出异常
                error_result = {
                    "choices": [{"message": {"content": error_msg}}],
                    "error": True,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                return index, error_result, e
        
        # 使用线程池进行并发请求
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_single, idx, msgs): idx
                for idx, msgs in enumerate(messages_list)
            }
            
            # 使用 tqdm 显示进度
            with tqdm(total=len(messages_list), desc="批量请求进度", ncols=100) as pbar:
                for future in as_completed(futures):
                    idx, result, error = future.result()
                    results[idx] = result
                    pbar.update(1)
                    if error:
                        pbar.set_postfix({"错误": 1})
        
        return results

    def call_batch_with_callback(
        self, 
        messages_list: List[List[Dict[str, str]]], 
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        并行批量请求接口，支持回调函数
        
        Args:
            messages_list: 消息列表的列表，每个元素是一个 messages 列表
            callback: 可选的回调函数，每次请求完成时调用 callback(index, result)
            **kwargs: 传递给每个请求的额外参数
            
        Returns:
            List[Dict]: 每个请求的结果列表
        """
        if not messages_list:
            return []
        
        results = [None] * len(messages_list)
        
        def process_single(index: int, messages: List[Dict[str, str]]) -> tuple:
            """处理单个请求的辅助函数"""
            try:
                result = self._call_single_with_retry(messages, **kwargs)
                if callback:
                    callback(index, result)
                return index, result, None
            except Exception as e:
                error_msg = f"[ERROR] 请求失败: {type(e).__name__}: {str(e)}"
                logging.error(f"批量请求中索引 {index} 失败: {error_msg}")
                error_result = {
                    "choices": [{"message": {"content": error_msg}}],
                    "error": True,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                if callback:
                    callback(index, error_result)
                return index, error_result, e
        
        # 使用线程池进行并发请求
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_single, idx, msgs): idx
                for idx, msgs in enumerate(messages_list)
            }
            
            for future in as_completed(futures):
                idx, result, error = future.result()
                results[idx] = result
        
        return results
