import os
import random
import re
from typing import Any
from types import SimpleNamespace

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from overrides import override
from qwen_agent.llm import get_chat_model
import time
from tqdm import tqdm

class QwenAPIHandler(OpenAICompletionsHandler):
    """
    This is the OpenAI-compatible API handler for Qwen, DeepSeek, and GPT models.
    
    All models use non-streaming responses (matching run.py configuration) by default.
    GPT models use GPT_API_KEY/GPT_API_BASE, while Qwen/DeepSeek models use QWEN_API_KEY/QWEN_API_BASE.
    For Qwen models, you can use a boolean flag `enable_thinking` to turn on/off streaming thinking mode.
    """
    # 类变量：用于跟踪已打印的任务
    _printed_tasks = set()
    _printed_count = 0
    _target_print_count = 5
    _multi_turn_printed = 0
    _single_turn_printed = 0

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        enable_thinking: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        # 只对 Qwen 接口生效的思考开关（GPT 始终不思考）
        self.enable_thinking_for_qwen = enable_thinking
        
        # Check if this is a GPT model (based on registry_name or model_name)
        # GPT models use GPT_API_KEY/GPT_API_BASE, others use QWEN_API_KEY/QWEN_API_BASE
        is_gpt_model = (
            "gpt" in registry_name.lower() or 
            "gpt" in model_name.lower()
        )
        # print(f"is_gpt_model: {is_gpt_model}, registry_name: {registry_name}, model_name: {model_name}")
        if is_gpt_model:
            # GPT series: use GPT_API_KEY and GPT_API_BASE (matching run.py)
            # For GPT models, only use GPT_API_KEY/GPT_API_BASE, do NOT fallback to OPENAI_API_KEY/OPENAI_BASE_URL
            # to avoid accidentally using Qwen/Aliyun credentials
            api_key = os.getenv("GPT_API_KEY") or "your api key"
            base_url = os.getenv("GPT_API_BASE") or "https://api.gpt.ge/v1/"
            # Ensure base_url format is correct (remove trailing slash, add /v1 if needed)
            base_url = base_url.rstrip('/')
            if not base_url.endswith('/v1'):
                base_url = f"{base_url}/v1"
            self.is_gpt_model = True
            # print(f"Using GPT config - base_url: {base_url}, api_key prefix: {api_key[:10]}...")
        else:
            # Qwen/DeepSeek series: use QWEN_API_KEY and QWEN_API_BASE
            base_url = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            api_key = os.getenv("QWEN_API_KEY", "your api key")
            self.is_gpt_model = False
            # print(f"Using Qwen config - base_url: {base_url}, api_key prefix: {api_key[:10]}...")
        
        # print(f"Final client config - base_url: {base_url}, api_key prefix: {api_key[:10]}...")
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def generate_with_backoff(self, **kwargs):
        """
        重写父类方法，添加更完整的错误处理和重试逻辑。
        当远程接口不能接受过快的访问时，会等待然后重试。
        支持重试 429 错误和 quota 相关错误（包括 insufficient_quota）。
        对于 data_inspection_failed 错误（内容不当），不重试，直接抛出异常。
        Qwen 接口是否使用流式+思考，由布尔开关 self.enable_thinking_for_qwen 控制。
        """
        max_retries = 10
        min_wait = 20
        max_wait = 120

        def _set_enable_thinking(extra_body: dict | None, value: bool):
            """
            统一在 extra_body 以及其常见嵌套字段中设置 enable_thinking，避免 dashscope 报
            `parameter.enable_thinking must be set to false for non-streaming calls`。
            """
            if not isinstance(extra_body, dict):
                return
            # 顶层
            if "enable_thinking" in extra_body:
                extra_body["enable_thinking"] = value
            # 常见嵌套字段：parameter / parameters
            for key in ("parameter", "parameters"):
                sub = extra_body.get(key)
                if isinstance(sub, dict) and "enable_thinking" in sub:
                    sub["enable_thinking"] = value

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                print(f"kwargs: {kwargs}")

                # GPT 模型：始终非流式 + 不思考
                if self.is_gpt_model:
                    # 防御性清理流式与思考相关参数
                    kwargs.pop("stream", None)
                    kwargs.pop("stream_options", None)
                    extra_body = kwargs.get("extra_body")
                    _set_enable_thinking(extra_body, False)

                    api_response = self.client.chat.completions.create(**kwargs)
                    end_time = time.time()
                    return api_response, end_time - start_time

                # Qwen 模型：根据 enable_thinking_for_qwen 控制是否思考
                if not self.enable_thinking_for_qwen:
                    # 不思考：统一非流式
                    kwargs.pop("stream", None)
                    kwargs.pop("stream_options", None)
                    extra_body = kwargs.get("extra_body")
                    _set_enable_thinking(extra_body, False)

                    api_response = self.client.chat.completions.create(**kwargs)
                    end_time = time.time()
                    return api_response, end_time - start_time

                # 思考模式：启用流式 + enable_thinking，并复用原来的收集逻辑
                kwargs["stream"] = True
                kwargs["stream_options"] = {"include_usage": True}
                extra_body = kwargs.get("extra_body")
                if isinstance(extra_body, dict):
                    _set_enable_thinking(extra_body, True)
                else:
                    kwargs["extra_body"] = {"enable_thinking": True}

                completion = self.client.chat.completions.create(**kwargs)

                reasoning_content = ""  # 完整思考过程
                answer_content = ""  # 完整回复
                tool_calls = []  # 工具调用列表
                is_answering = False  # 是否进入回复阶段
                usage = None  # 使用统计信息

                # 打印思考过程标题
                print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n", flush=True)

                for chunk in completion:
                    if not chunk.choices:
                        # 包含usage信息的chunk
                        if hasattr(chunk, 'usage') and chunk.usage:
                            usage = chunk.usage
                        continue

                    delta = chunk.choices[0].delta

                    # 收集思考内容
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                        if not is_answering:
                            print(delta.reasoning_content, end="", flush=True)
                        reasoning_content += delta.reasoning_content

                    # 收到content，开始进行回复
                    if hasattr(delta, "content") and delta.content:
                        if not is_answering:
                            print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n", flush=True)
                            is_answering = True
                        print(delta.content, end="", flush=True)
                        answer_content += delta.content

                    # 收集工具调用
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            if tool_call_delta.index is not None:
                                # 确保tool_calls列表足够大
                                while len(tool_calls) <= tool_call_delta.index:
                                    tool_calls.append({
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })

                                tool_call = tool_calls[tool_call_delta.index]
                                if tool_call_delta.id:
                                    tool_call["id"] = tool_call_delta.id
                                if hasattr(tool_call_delta.function, "name") and tool_call_delta.function.name:
                                    tool_call["function"]["name"] = tool_call_delta.function.name
                                if hasattr(tool_call_delta.function, "arguments") and tool_call_delta.function.arguments:
                                    tool_call["function"]["arguments"] += tool_call_delta.function.arguments

                print("\n" + "=" * 50 + "\n", flush=True)
                end_time = time.time()

                # 构建兼容的响应对象
                message = SimpleNamespace()
                message.content = answer_content if answer_content else None
                message.reasoning_content = reasoning_content if reasoning_content else ""

                # 如果有工具调用，构建tool_calls
                if tool_calls:
                    # 将字典转换为类似OpenAI的ToolCall对象
                    tool_call_objects = []
                    for tc in tool_calls:
                        if tc["function"]["name"]:  # 只添加非空的工具调用
                            func_obj = SimpleNamespace()
                            func_obj.name = tc["function"]["name"]
                            func_obj.arguments = tc["function"]["arguments"]

                            tool_call_obj = SimpleNamespace()
                            tool_call_obj.id = tc["id"]
                            tool_call_obj.type = tc["type"]
                            tool_call_obj.function = func_obj
                            tool_call_objects.append(tool_call_obj)

                    if tool_call_objects:
                        message.tool_calls = tool_call_objects
                    else:
                        message.tool_calls = None
                else:
                    message.tool_calls = None

                choice = SimpleNamespace()
                choice.message = message

                response = SimpleNamespace()
                response.choices = [choice]

                # 构建usage对象
                if usage:
                    response.usage = usage
                else:
                    usage_obj = SimpleNamespace()
                    usage_obj.prompt_tokens = 0
                    usage_obj.completion_tokens = 0
                    response.usage = usage_obj

                return response, end_time - start_time
                
            except (RateLimitError, APIError, APIConnectionError) as e:
                error_str = str(e)
                
                # 检查是否是 data_inspection_failed 错误（内容不当），如果是，不重试，直接抛出
                if "data_inspection_failed" in error_str.lower() or "inappropriate content" in error_str.lower():
                    # 不重试，直接抛出异常
                    raise
                
                # 检查是否是429错误或quota相关错误，如果是，进行重试
                is_retryable = (
                    "429" in error_str or 
                    "quota" in error_str.lower() or 
                    "insufficient_quota" in error_str.lower() or
                    "rate_limit" in error_str.lower() or
                    bool(re.search(r"rate.?limit", error_str, re.IGNORECASE)) or
                    bool(re.search(r"exceeded.*quota", error_str, re.IGNORECASE)) or
                    isinstance(e, RateLimitError)
                )
                
                if not is_retryable:
                    # 不可重试的错误，直接抛出
                    raise
                
                # 如果是最后一次尝试，抛出异常
                if attempt == max_retries - 1:
                    raise
                
                # 计算等待时间（指数退避）
                wait_time = min(min_wait * (2 ** attempt) + random.uniform(0, 1), max_wait)
                tqdm.write(f"Attempt {attempt + 1} failed. Sleeping for {wait_time:.2f} seconds before retrying... Error: {e}")
                time.sleep(wait_time)

    #### FC methods ####

    @override
    def _query_FC(self, inference_data: dict, n_sample: int = None):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        kwargs = {
            "messages": inference_data["message"],
            "model": self.model_name.replace("-FC", ""),
            "tools": tools,
            "stream": False,  # All models use non-streaming (matching run.py)
        }
        
        # Support batch sampling
        if n_sample is not None and n_sample > 1:
            kwargs["n"] = n_sample
        
        return self.generate_with_backoff(**kwargs)

    @override
    def _parse_query_response_FC(self, api_response: Any, n_sample: int = None) -> dict:
        # All models now use non-streaming responses (matching run.py)
        # Check if this is a batch response: has choices attribute and choices count > 1
        is_batch_response = (
            n_sample is not None 
            and n_sample > 1 
            and hasattr(api_response, 'choices') 
            and len(api_response.choices) > 1
        )
        
        # All responses are non-streaming now
        if True:  # Always handle as non-streaming
            # 非流式批量采样响应或 GPT 单次响应（类似 OpenAICompletionsHandler）
            # 对于批量采样，返回所有候选结果的列表
            # 对于 GPT 单次请求，返回单个结果（但格式保持一致）
            all_model_responses = []
            all_tool_call_ids = []
            all_messages = []
            all_reasoning_contents = []
            
            # Handle both single and batch responses
            choices = api_response.choices if hasattr(api_response, 'choices') else [api_response]
            
            for choice in choices:
                # For batch responses, choice is a Choice object with message attribute
                # For single responses, choice might be the response itself
                if hasattr(choice, 'message'):
                    message = choice.message
                elif hasattr(choice, 'choices') and len(choice.choices) > 0:
                    message = choice.choices[0].message
                else:
                    # Fallback: assume it's already a message
                    message = choice
                
                reasoning_content = getattr(message, "reasoning_content", "") or ""
                
                if hasattr(message, "tool_calls") and message.tool_calls:
                    model_response = [
                        {tool_call.function.name: tool_call.function.arguments}
                        for tool_call in message.tool_calls
                    ]
                    tool_call_ids = [tool_call.id for tool_call in message.tool_calls]
                    model_response_message_for_chat_history = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in message.tool_calls
                        ],
                    }
                else:
                    model_response = getattr(message, "content", "") or ""
                    tool_call_ids = []
                    model_response_message_for_chat_history = {
                        "role": "assistant",
                        "content": model_response,
                    }
                
                if reasoning_content:
                    model_response_message_for_chat_history["reasoning_content"] = reasoning_content
                
                all_model_responses.append(model_response)
                all_tool_call_ids.append(tool_call_ids)
                all_messages.append(model_response_message_for_chat_history)
                all_reasoning_contents.append(reasoning_content)
            
            # For single requests (not batch), return single result instead of list
            if not is_batch_response and len(all_model_responses) == 1:
                response_data = {
                    "model_responses": all_model_responses[0],  # Single result, not a list
                    "model_responses_message_for_chat_history": all_messages[0],  # Single message
                    "tool_call_ids": all_tool_call_ids[0],  # Single list
                    "input_token": api_response.usage.prompt_tokens if hasattr(api_response, 'usage') else 0,
                    "output_token": api_response.usage.completion_tokens if hasattr(api_response, 'usage') else 0,
                }
                if all_reasoning_contents[0]:
                    response_data["reasoning_content"] = all_reasoning_contents[0]
            else:
                response_data = {
                    "model_responses": all_model_responses,  # 列表的列表
                    "model_responses_message_for_chat_history": all_messages,  # 列表
                    "tool_call_ids": all_tool_call_ids,  # 列表的列表
                    "input_token": api_response.usage.prompt_tokens if hasattr(api_response, 'usage') else 0,
                    "output_token": api_response.usage.completion_tokens if hasattr(api_response, 'usage') else 0,
                }
                if any(all_reasoning_contents):
                    response_data["reasoning_content"] = all_reasoning_contents
            
            return response_data

    #### Prompting methods ####

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        重写父类方法，确保 function 字段被包含在 inference_data 中，
        以便在 _query_prompting 中可以打印修改后的 function descriptions（特别是 draft 方法）。
        """
        import copy
        # 使用深拷贝确保获取到修改后的 functions（包含 memory 内容）
        # 注意：base_handler 在调用此方法之前已经修改了 test_entry["function"] 中的 description
        functions: list = copy.deepcopy(test_entry["function"])
        test_entry_id: str = test_entry["id"]

        # 调试：检查是否有 functions 包含 memory 内容
        method = getattr(self, 'method', None)
        if method in ("mem", "m0"):
            memory_count = sum(1 for fun in functions if "Besides, this is optional guidance on how to better use the tool" in fun.get("description", ""))
            if memory_count > 0:
                tqdm.write(f"🔍 [调试] _pre_query_processing_prompting: 检测到 {memory_count}/{len(functions)} 个函数包含 memory 内容")

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        # 返回包含 function 字段的 inference_data，以便打印逻辑可以访问
        # 注意：这里返回的 functions 应该是已经包含 memory 内容的（因为在 base_handler 中已经修改了 test_entry["function"]）
        return {"message": [], "function": functions}

    @override
    def _query_prompting(self, inference_data: dict, n_sample: int = None):
        message: list[dict] = inference_data["message"]
        inference_data["inference_input_log"] = {"message": repr(message)}

        # 只针对 qwen 系列模型（非 GPT）打印系统提示词
        if not self.is_gpt_model:
            test_entry_id = inference_data.get("test_entry_id", "")
            is_multi_turn = inference_data.get("is_multi_turn", False)
            
            # 判断是否需要打印：确保打印5个任务，且包含多轮和单轮
            should_print = False
            if test_entry_id and test_entry_id not in QwenAPIHandler._printed_tasks:
                # 如果还没打印够5个，且当前任务类型还没打印够，则打印
                if QwenAPIHandler._printed_count < QwenAPIHandler._target_print_count:
                    if is_multi_turn:
                        # 多轮任务：确保至少打印2个
                        if QwenAPIHandler._multi_turn_printed < 2 or QwenAPIHandler._printed_count < QwenAPIHandler._target_print_count:
                            should_print = True
                    else:
                        # 单轮任务：确保至少打印2个
                        if QwenAPIHandler._single_turn_printed < 2 or QwenAPIHandler._printed_count < QwenAPIHandler._target_print_count:
                            should_print = True
                
                # 如果还没打印够，随机选择一些任务打印（确保最终能打印够5个）
                if not should_print and QwenAPIHandler._printed_count < QwenAPIHandler._target_print_count:
                    # 随机选择，但确保最终能打印够5个
                    remaining = QwenAPIHandler._target_print_count - QwenAPIHandler._printed_count
                    total_remaining_tasks = 100  # 假设还有足够多的任务
                    if random.random() < (remaining / max(total_remaining_tasks, remaining)):
                        should_print = True
            
            if should_print:
                # 提取系统提示词
                system_prompt = ""
                for msg in message:
                    if msg.get("role") == "system":
                        system_prompt = msg.get("content", "")
                        break
                
                # 获取当前使用的 method
                method = getattr(self, 'method', None)
                method_str = f" | Method: {method}" if method else " | Method: None (默认)"
                
                # 提取修改后的 function descriptions（从 inference_data 中获取，如果可用）
                function_info = ""
                if "function" in inference_data:
                    functions = inference_data.get("function", [])
                    if functions:
                        function_info = "\n\n修改后的 Function Descriptions (前3个函数):\n" + "-" * 100 + "\n"
                        for i, fun in enumerate(functions[:3]):  # 只显示前3个函数
                            fun_name = fun.get("name", "unknown")
                            fun_desc = fun.get("description", "")
                            
                            # 检查是否包含总结内容（通过检查是否包含特定的标识字符串）
                            has_memory = "Besides, this is optional guidance on how to better use the tool" in fun_desc
                            memory_indicator = " ✅ [包含Memory]" if has_memory else " ❌ [无Memory]"
                            
                            # 如果描述太长，截断
                            if len(fun_desc) > 500:
                                fun_desc = fun_desc[:500] + "... (已截断)"
                            function_info += f"\n[{i+1}] {fun_name}{memory_indicator}:\n{fun_desc}\n"
                        if len(functions) > 3:
                            function_info += f"\n... (还有 {len(functions) - 3} 个函数未显示)\n"
                
                # # 打印系统提示词
                # task_type = "多轮" if is_multi_turn else "单轮"
                # tqdm.write("\n" + "=" * 100)
                # tqdm.write(f"📝 [Qwen Prompt 打印] 任务 ID: {test_entry_id} | 类型: {task_type}{method_str} | 已打印: {QwenAPIHandler._printed_count + 1}/{QwenAPIHandler._target_print_count}")
                # tqdm.write("-" * 100)
                # tqdm.write(f"系统提示词 (System Prompt):")
                # tqdm.write("-" * 100)
                # if system_prompt:
                #     # 如果系统提示词太长，只打印前3000个字符（增加长度以便看到更多内容）
                #     if len(system_prompt) > 3000:
                #         tqdm.write(system_prompt[:3000] + "\n... (已截断，完整长度: " + str(len(system_prompt)) + " 字符)")
                #     else:
                #         tqdm.write(system_prompt)
                # else:
                #     tqdm.write("(无系统提示词)")
                
                # 打印修改后的 function descriptions
                if function_info:
                    tqdm.write(function_info)
                
                tqdm.write("=" * 100 + "\n")
                
                # 更新打印状态
                QwenAPIHandler._printed_tasks.add(test_entry_id)
                QwenAPIHandler._printed_count += 1
                if is_multi_turn:
                    QwenAPIHandler._multi_turn_printed += 1
                else:
                    QwenAPIHandler._single_turn_printed += 1

        kwargs = {
            "messages": inference_data["message"],
            "model": self.model_name,
            "temperature": self.temperature,
            "extra_body": {
                "enable_thinking": False
            },
            "stream": False,  # All models use non-streaming (matching run.py)
        }
        if self.is_gpt_model:
            kwargs={
            "messages": inference_data["message"],
            "model": self.model_name,
            "temperature": self.temperature,
            "stream": False,  # All
            }
        
        # Support batch sampling
        if n_sample is not None and n_sample > 1:
            kwargs["n"] = n_sample
        
        return self.generate_with_backoff(**kwargs)

    @override
    def _parse_query_response_prompting(self, api_response: Any, n_sample: int = None) -> dict:
        # All models now use non-streaming responses (matching run.py)
        # Check if this is a batch response: has choices attribute and choices count > 1
        is_batch_response = (
            n_sample is not None 
            and n_sample > 1 
            and hasattr(api_response, 'choices') 
            and len(api_response.choices) > 1
        )
        
        # All responses are non-streaming now
        if True:  # Always handle as non-streaming
            # 非流式批量采样响应或 GPT 单次响应（类似 OpenAICompletionsHandler）
            # 对于批量采样，返回所有候选结果的列表
            # 对于 GPT 单次请求，返回单个结果（但格式保持一致）
            all_model_responses = []
            all_messages = []
            all_reasoning_contents = []
            
            # Handle both single and batch responses
            choices = api_response.choices if hasattr(api_response, 'choices') else [api_response]
            
            for choice in choices:
                # For batch responses, choice is a Choice object with message attribute
                # For single responses, choice might be the response itself
                if hasattr(choice, 'message'):
                    message = choice.message
                elif hasattr(choice, 'choices') and len(choice.choices) > 0:
                    message = choice.choices[0].message
                else:
                    # Fallback: assume it's already a message
                    message = choice
                
                reasoning_content = getattr(message, "reasoning_content", "") or ""
                answer_content = getattr(message, "content", "") or ""
                
                model_response_message_for_chat_history = {
                    "role": "assistant",
                    "content": answer_content,
                }
                
                if reasoning_content:
                    model_response_message_for_chat_history["reasoning_content"] = reasoning_content
                
                all_model_responses.append(answer_content)
                all_messages.append(model_response_message_for_chat_history)
                all_reasoning_contents.append(reasoning_content)
            
            # For single requests (not batch), return single result instead of list
            if not is_batch_response and len(all_model_responses) == 1:
                response_data = {
                    "model_responses": all_model_responses[0],  # Single result, not a list
                    "model_responses_message_for_chat_history": all_messages[0],  # Single message
                    "input_token": api_response.usage.prompt_tokens if hasattr(api_response, 'usage') else 0,
                    "output_token": api_response.usage.completion_tokens if hasattr(api_response, 'usage') else 0,
                }
                if all_reasoning_contents[0]:
                    response_data["reasoning_content"] = all_reasoning_contents[0]
            else:
                response_data = {
                    "model_responses": all_model_responses,  # 列表
                    "model_responses_message_for_chat_history": all_messages,  # 列表
                    "input_token": api_response.usage.prompt_tokens if hasattr(api_response, 'usage') else 0,
                    "output_token": api_response.usage.completion_tokens if hasattr(api_response, 'usage') else 0,
                }
                if any(all_reasoning_contents):
                    response_data["reasoning_content"] = all_reasoning_contents
            
            return response_data



class QwenAgentThinkHandler(OpenAICompletionsHandler):

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        """
        Note: Need to start vllm server first with command:
        vllm serve xxx \
            --served-model-name xxx \
            --port 8000 \
            --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
            --max-model-len 65536
        """
        
        self.llm = get_chat_model({
        'model': model_name,  # name of the model served by vllm server
        'model_type': 'oai',
        'model_server':'http://localhost:8000/v1', # can be replaced with server host
        'api_key': "none",
        'generate_cfg': {
            'fncall_prompt_type': 'nous',
            'extra_body': {
                'chat_template_kwargs': {
                    'enable_thinking': False
                }
            },
            'temperature': 0.5,
        }
    })

    #### FC methods ####
    @override
    def _query_FC(self, inference_data: dict, n_sample: int = None):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        start_time = time.time()
        if len(tools) > 0:
            responses = None
            for resp in self.llm.quick_chat_oai(message, tools):
                responses = resp  # 保留最后一个完整响应
                
        else:
            responses = None
            for resp in self.llm.quick_chat_oai(message):
                responses = resp
        end_time = time.time()
        
        return responses, end_time-start_time
    
    @override
    def _parse_query_response_FC(self, api_response: Any, n_sample: int = None) -> dict:
        try:
            model_responses = [
                {func_call['function']['name']: func_call['function']['arguments']}
                for func_call in api_response["choices"][0]["message"]["tool_calls"]
            ]
            tool_call_ids = [
                func_call['function']['name'] for func_call in api_response["choices"][0]["message"]["tool_calls"]
            ]
        except:
            model_responses = api_response["choices"][0]["message"]["content"]
            tool_call_ids = []
        
        response_data = {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": api_response["choices"][0]["message"],
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.get("usage", {}).get("prompt_tokens", 0),
            "output_token": api_response.get("usage", {}).get("completion_tokens", 0),
        }
        return response_data
        

    
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        
        
        if isinstance(model_response_data["model_responses_message_for_chat_history"], list):
            inference_data["message"]+=model_response_data["model_responses_message_for_chat_history"]
        else:
            inference_data["message"].append(
                model_response_data["model_responses_message_for_chat_history"]
            )

        return inference_data


class QwenAgentNoThinkHandler(QwenAgentThinkHandler):

    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        """
        Note: Need to start vllm server first with command:
        vllm serve xxx \
            --served-model-name xxx \
            --port 8000 \
            --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
            --max-model-len 65536
        """
        
        self.llm = get_chat_model({
        'model': model_name, # name of the model served by vllm server
        'model_type': 'oai',
        'model_server':'http://localhost:8000/v1', # can be replaced with server host
        'api_key': "none",
        'generate_cfg': {
            'fncall_prompt_type': 'nous',
            'extra_body': {
                'chat_template_kwargs': {
                    'enable_thinking': False
                }
            },
            "thought_in_content": False,
            'temperature': 0.5,
            'top_p': 0.8,
            'top_k': 20,
            'repetition_penalty': 1.0,
            'presence_penalty': 1.5,
            'max_input_tokens': 58000,
            'timeout': 1000,
            'max_tokens': 16384
        }
    })