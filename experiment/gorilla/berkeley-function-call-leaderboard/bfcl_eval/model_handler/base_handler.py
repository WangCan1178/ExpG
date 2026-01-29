import json
import threading
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from bfcl_eval.constants.category_mapping import VERSION_PREFIX
from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
    MAXIMUM_STEP_LIMIT,
)
from bfcl_eval.constants.enums import ModelStyle, ReturnFormat
from bfcl_eval.constants.eval_config import RESULT_PATH
from bfcl_eval.constants.executable_backend_config import (
    OMIT_STATE_INFO_CLASSES,
    STATELESS_CLASSES,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.utils import add_memory_instruction_system_prompt
from bfcl_eval.utils import *  # Includes calculate_tool_hash for unified hash calculation
from overrides import final
from tqdm import tqdm

if TYPE_CHECKING:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_api_metaclass import (
        MemoryAPI,
    )

# 模块缓存，确保 run.py 模块只被导入一次，避免全局变量不共享的问题
_run_module_cache = None
_run_module_lock = threading.Lock()

# 内存缓存：避免对同一个 (model, test_entry_id) 在 n_sample 采样时重复访问向量库
_memory_cache: dict[tuple[str, str], dict] = {}

def _get_run_module():
    """
    获取 run.py 模块的单例实例，确保全局变量共享。
    这样可以避免每次调用都创建新的模块实例，导致全局变量 _global_mem0_memory_instance 不共享。
    """
    global _run_module_cache
    
    if _run_module_cache is not None:
        return _run_module_cache
    
    with _run_module_lock:
        # 双重检查
        if _run_module_cache is not None:
            return _run_module_cache
        
        try:
            from pathlib import Path
            import importlib.util
            run_script_path = Path(__file__).resolve().parents[1] / "scripts" / "run.py"
            if run_script_path.exists():
                spec = importlib.util.spec_from_file_location("run_module", run_script_path)
                run_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(run_module)
                _run_module_cache = run_module
                return run_module
        except Exception as e:
            tqdm.write(f"⚠️ 导入 run.py 模块失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return None


class BaseHandler:
    model_name: str
    is_fc_model: bool
    registry_name: str
    temperature: float
    registry_dir_name: str
    model_name_underline_replaced: str
    model_style: ModelStyle

    def __init__(
        self, model_name, temperature, registry_name, is_fc_model, **kwargs
    ) -> None:
        """
        Args:
            model_name: The name of the model as used in the vendor API or on Hugging Face.
            temperature: The temperature of the model.
            registry_name: The name of the model as used internally in BFCL, used for result directory naming.
            is_fc_model: Whether the model is a function calling model.
            **kwargs: Additional attributes passed via kwargs.
        """
        self.model_name = model_name
        self.is_fc_model = is_fc_model
        self.registry_name = registry_name

        # Replace the dash and dot with underscore for valid variable name
        self.model_name_underline_replaced = (
            model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        )
        # The directory name for the model
        # Replace the slash with underscore to avoid creating subdirectories
        self.registry_dir_name = registry_name.replace("/", "_")
        self.temperature = temperature

        # Set any additional attributes passed via kwargs
        for _key, _value in kwargs.items():
            setattr(self, _key, _value)

    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
        n_sample: int = None,
    ):
        # This method is used to retrive model response for each model.
        
        method = getattr(self, 'method', None)
        # Store memory info dict for later use (will be added to system prompt)
        cache_key = None
        memory_info_dict: dict = {}

        # 先尝试从全局缓存中复用 memory
        # mem 方法：只能在同一任务的多次采样之间复用，缓存 key = (registry_name, test_id)
        # m0 方法：工具级缓存由 run.py 中的 mem0 逻辑负责，这里不做跨任务全局缓存
        if method == "mem":
            test_id = str(test_entry.get("id", ""))
            cache_key = (self.registry_name, test_id)
            if cache_key in _memory_cache:
                memory_info_dict = _memory_cache[cache_key].copy()
            elif isinstance(test_entry.get("_memory_info_dict"), dict):
                # 其次尝试从 test_entry 本身复用（例如多轮对话内部多次调用）
                memory_info_dict = test_entry["_memory_info_dict"]
        elif method == "m0":
            # m0：仅复用当前 entry 上已经计算好的 memory（例如多轮对话内部），
            # 每个工具的跨任务缓存由 run.py 中的 get_mem0_memories_for_tools 自己维护
            if isinstance(test_entry.get("_memory_info_dict"), dict):
                memory_info_dict = test_entry["_memory_info_dict"]

        # Load draft improved descriptions if method is "draft"
        draft_descriptions = {}
        if method == "draft":
            from bfcl_eval.constants.eval_config import PROJECT_ROOT
            import json as json_module
            
            draft_config_path = PROJECT_ROOT / "bfcl_eval" / "data" / "config_draft.json"
            if draft_config_path.exists():
                try:
                    with open(draft_config_path, 'r', encoding='utf-8') as f:
                        draft_config = json_module.load(f)
                    
                    # Calculate hash for each function and look up in draft_config
                    # Note: test_entry["function"] may already have language hints added, so we need to remove them
                    # before calculating hash to match the hash in config_draft.json (which is based on original descriptions)
                    for fun in test_entry.get("function", []):
                        fun_name = fun.get('name', '')
                        if fun_name:
                            # Get tool_config directly from test_entry function field
                            tool_config = {k: v for k, v in fun.items() if k != 'name'}
                            
                            # Remove language-specific hint from description before calculating hash
                            # because config_draft.json was created using original descriptions (without hints)
                            if tool_config and 'description' in tool_config:
                                tool_config_for_hash = tool_config.copy()
                                tool_config_for_hash['description'] = remove_language_specific_hint_from_description(
                                    tool_config['description']
                                )
                            else:
                                tool_config_for_hash = tool_config
                            
                            # Calculate hash for this function using unified function (with cleaned description)
                            if tool_config_for_hash:
                                tool_hash = calculate_tool_hash(fun_name, tool_config_for_hash)
                                
                                # Look up in draft_config
                                if tool_hash in draft_config:
                                    draft_descriptions[fun_name] = draft_config[tool_hash].get("description", "")
                    
                    if draft_descriptions:
                        tqdm.write(f"✅ Loaded DRAFT improved descriptions for {len(draft_descriptions)} tool(s): {list(draft_descriptions.keys())}")
                except Exception as e:
                    tqdm.write(f"⚠️ Error loading DRAFT improved config: {e}, using original descriptions")
        
        # Only hit the vector store when we don't already have cached memory for this entry
        if method == "mem" and not memory_info_dict:
            # Use test_entry id as workspace_id and source_task (e.g., "live_relevance_7-7-0")
            # Use registry_name to match the workspace_id used in run.py (which uses model parameter, i.e., registry_name)
            workspace_id = "bfcl-"+self.registry_name
            source_task = test_entry.get("id")
            
            # Extract tool names from function list and convert to hash names (same as stored)
            # Note: test_entry["function"] may already have language hints added, so we need to remove them
            # before calculating hash to match the hash used in run.py (which is based on original descriptions)
            tool_names_hash = []  # Hash names for retrieval
            tool_name_to_hash = {}  # Mapping from original name to hash name
            for fun in test_entry.get("function", []):
                fun_name = fun.get('name', '')
                if fun_name:
                    # Get tool_config directly from test_entry function field
                    tool_config = {k: v for k, v in fun.items() if k != 'name'}
                    
                    # Remove language-specific hint from description before calculating hash
                    # because run.py stores memory using hash based on original descriptions (without hints)
                    if tool_config and 'description' in tool_config:
                        tool_config_for_hash = tool_config.copy()
                        tool_config_for_hash['description'] = remove_language_specific_hint_from_description(
                            tool_config['description']
                        )
                    else:
                        tool_config_for_hash = tool_config
                    
                    # Calculate hash for this function using unified function (with cleaned description)
                    if tool_config_for_hash:
                        tool_hash = calculate_tool_hash(fun_name, tool_config_for_hash)
                        tool_names_hash.append(tool_hash)
                        tool_name_to_hash[fun_name] = tool_hash
                    else:
                        # If no config, use original name
                        tool_names_hash.append(fun_name)
                        tool_name_to_hash[fun_name] = fun_name
            
            # Use ToolMemoryFetcher (similar to run.py)
            from bfcl_eval.tool_memory import ToolMemoryFetcher
            
            try:
                # Create ToolMemoryFetcher instance
                # workspace_id is fixed as "bfcl"+model_name
                # source_task uses test_entry id to exclude current task's records
                fetcher = ToolMemoryFetcher(
                    workspace_id=workspace_id,
                    source_task=str(source_task) if source_task else "",
                    model_name=str(self.model_name),
                )
                is_hash = True
                # Collect memory for all tools using hash names (returns Dict[str, Dict[str, str]] with 'content' and 'add_to')
                # 分批处理，每批4个工具，防止高并发带来的问题
                if is_hash:
                    batch_size = 4
                    all_fetched = {}
                    for i in range(0, len(tool_names_hash), batch_size):
                        batch = tool_names_hash[i:i + batch_size]
                        if batch:
                            try:
                                batch_fetched = fetcher.collect_memory(batch)
                                if batch_fetched:
                                    all_fetched.update(batch_fetched)
                            except Exception as batch_e:
                                tqdm.write(f"⚠️ 批次 {i//batch_size + 1} 获取 memory 失败: {batch_e}")
                                continue
                    
                    # Extract content from the fetched results and map back to original names
                    for original_name, hash_name in tool_name_to_hash.items():
                        tool_memory = (all_fetched or {}).get(hash_name, {})
                        if isinstance(tool_memory, dict):
                            content = tool_memory.get("content", "") or ""
                        else:
                            # Compatible with old format (string)
                            content = tool_memory or ""
                        
                        if content:
                            memory_info_dict[original_name] = content
                else:
                    # Use original tool names instead of hash names
                    tool_names_original = []
                    for fun in test_entry.get("function", []):
                        fun_name = fun.get('name', '')
                        if fun_name:
                            tool_names_original.append(fun_name)
                    
                    # Collect memory for all tools using original names
                    batch_size = 4
                    all_fetched = {}
                    for i in range(0, len(tool_names_original), batch_size):
                        batch = tool_names_original[i:i + batch_size]
                        if batch:
                            try:
                                batch_fetched = fetcher.collect_memory(batch)
                                if batch_fetched:
                                    all_fetched.update(batch_fetched)
                            except Exception as batch_e:
                                tqdm.write(f"⚠️ 批次 {i//batch_size + 1} 获取 memory 失败: {batch_e}")
                                continue
                    
                    # Extract content from the fetched results using original names
                    for tool_name in tool_names_original:
                        tool_memory = (all_fetched or {}).get(tool_name, {})
                        if isinstance(tool_memory, dict):
                            content = tool_memory.get("content", "") or ""
                        else:
                            # Compatible with old format (string)
                            content = tool_memory or ""
                        
                        if content:
                            memory_info_dict[tool_name] = content
                    
                if memory_info_dict:
                    tqdm.write(f"✅ Retrieved memory for {len(memory_info_dict)} tool(s) using ToolMemoryFetcher (batched): {list(memory_info_dict.keys())}")
            except Exception as e:
                tqdm.write(f"⚠️ Error using ToolMemoryFetcher: {e}, falling back to original method")
                # Fallback to original method
                from bfcl_eval.utils import get_memory_info_for_functions
                memory_info_dict = get_memory_info_for_functions(test_entry.get("function", []), workspace_id=workspace_id)
        
        if method == "m0" and not memory_info_dict:
            # Use mem0 to retrieve tool memories
            # Extract tool names from function list and convert to hash names (same as stored)
            # Note: test_entry["function"] may already have language hints added, so we need to remove them
            # before calculating hash to match the hash used in run.py (which is based on original descriptions)
            tool_names_hash = []  # Hash names for retrieval
            tool_name_to_hash = {}  # Mapping from original name to hash name
            for fun in test_entry.get("function", []):
                fun_name = fun.get('name', '')
                if fun_name:
                    # Get tool_config directly from test_entry function field
                    tool_config = {k: v for k, v in fun.items() if k != 'name'}
                    
                    # Remove language-specific hint from description before calculating hash
                    # because run.py stores memory using hash based on original descriptions (without hints)
                    if tool_config and 'description' in tool_config:
                        tool_config_for_hash = tool_config.copy()
                        tool_config_for_hash['description'] = remove_language_specific_hint_from_description(
                            tool_config['description']
                        )
                    else:
                        tool_config_for_hash = tool_config
                    
                    # Calculate hash for this function using unified function (with cleaned description)
                    if tool_config_for_hash:
                        tool_hash = calculate_tool_hash(fun_name, tool_config_for_hash)
                        tool_names_hash.append(tool_hash)
                        tool_name_to_hash[fun_name] = tool_hash
                    else:
                        # If no config, use original name
                        tool_names_hash.append(fun_name)
                        tool_name_to_hash[fun_name] = fun_name
            
            if tool_names_hash:
                try:
                    # 使用缓存的模块实例，确保全局变量共享
                    run_module = _get_run_module()
                    if run_module is not None:
                        # Get mem0 memory instance
                        memory_instance = run_module.get_mem0_memory_instance()
                        if memory_instance:
                            # Get current task ID (for exclusion)
                            current_task_id = test_entry.get("id", "")
                            
                            # Get memories for tools using hash names (exclude current task)
                            run_id = f"bfcl-{self.model_name}"
                            memory_map = run_module.get_mem0_memories_for_tools(tool_names_hash, current_task_id, memory_instance, run_id=run_id)
                            
                            # Map back to original names and add to memory_info_dict
                            for original_name, hash_name in tool_name_to_hash.items():
                                memory_text = memory_map.get(hash_name, "")
                                if memory_text:
                                    memory_info_dict[original_name] = memory_text
                            
                            if memory_info_dict:
                                tqdm.write(
                                    f"✅ [m0] Retrieved memory for {len(memory_info_dict)} tool(s) using mem0: {list(memory_info_dict.keys())}"
                                )
                        else:
                            tqdm.write(f"⚠️ mem0 Memory 实例未初始化，无法获取工具记忆")
                    else:
                        tqdm.write(f"⚠️ 无法找到 run.py 脚本，无法使用 mem0")
                except Exception as e:
                    tqdm.write(f"⚠️ Error using mem0: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Store memory_info_dict in cache & test_entry for use in inference methods
        if method in ("mem", "m0") and memory_info_dict:
            # 对于 qwen 系列 prompt 模型（非 FC），在使用 mem 或 m0 方法时，对 memory 进行裁剪
            # 裁剪方法：模型最大上下文 120000 tokens / 工具数目
            is_qwen_prompt_model = 'qwen' in self.model_name.lower() and not self.is_fc_model
            if is_qwen_prompt_model and memory_info_dict:
                num_tools = len(memory_info_dict)
                if num_tools > 0:
                    # 计算每个工具的 memory 最大 token 数
                    max_tokens_per_tool = 120000 // num_tools
                    # 使用 token 估算：对于中英文混合文本，1 token ≈ 2.5 字符（保守估算）
                    # 这样可以确保不会超过 token 限制
                    max_chars_per_tool = int(max_tokens_per_tool * 2.5)
                    
                    # 对每个工具的 memory 进行裁剪
                    for tool_name, memory_content in memory_info_dict.items():
                        if isinstance(memory_content, str) and len(memory_content) > max_chars_per_tool:
                            # 裁剪 memory 内容
                            original_length = len(memory_content)
                            memory_info_dict[tool_name] = memory_content[:max_chars_per_tool]
                            tqdm.write(f"✂️  已裁剪工具 {tool_name} 的 memory: {original_length} -> {len(memory_info_dict[tool_name])} 字符 (最大 {max_chars_per_tool} 字符, 约 {max_tokens_per_tool} tokens)")
            
            # 写入全局缓存（用于不同 deepcopy 之间的复用）以及当前 entry
            # 只对 mem 方法做全局缓存；m0 的跨任务缓存由 run.py 中的 mem0 逻辑负责
            if method == "mem" and cache_key is not None:
                _memory_cache[cache_key] = memory_info_dict.copy()
            test_entry["_memory_info_dict"] = memory_info_dict
        
        
        for fun in test_entry.get("function", []):
            fun_name = fun['name']
            # Apply draft improved description if available (draft method)
            if method == "draft" and fun_name in draft_descriptions:
                fun["description"] = draft_descriptions[fun_name]
            # Apply memory info if available (mem or m0 method)
            elif method in ("mem", "m0") and fun_name in memory_info_dict:
                marker = "Besides, this is optional guidance on how to better use the tool"
                desc = fun.get("description", "") or ""
                # 防止在对同一个 entry 进行多次采样时重复拼接 memory
                if marker not in desc:
                    fun["description"] = (
                        desc
                        + memory_info_dict[fun_name]
                    )
                    # 针对 m0 方法，再打印一条调试信息，确认该 memory 已经写入到描述中
                    if method == "m0":
                        mem_len = len(memory_info_dict.get(fun_name, "") or "")
                        tqdm.write(
                            f"🧠 [m0] Applied memory to function '{fun_name}' "
                            f"(memory length: {mem_len} chars)"
                        )
        
        # FC model
        # TODO: Let all models have the is_fc_model attribute and remove the "FC" check
        if "FC" in self.registry_name or self.is_fc_model:
            if contain_multi_turn_interaction(test_entry["id"]):
                return self.inference_multi_turn_FC(
                    test_entry, include_input_log, exclude_state_log, n_sample=n_sample
                )
            else:
                return self.inference_single_turn_FC(test_entry, include_input_log, n_sample=n_sample)
        # Prompting model
        else:
            if contain_multi_turn_interaction(test_entry["id"]):
                return self.inference_multi_turn_prompting(
                    test_entry, include_input_log, exclude_state_log, n_sample=n_sample
                )
            else:
                return self.inference_single_turn_prompting(test_entry, include_input_log, n_sample=n_sample)

    @final
    def inference_multi_turn_FC(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
        n_sample: int = None,
    ) -> tuple[list[list], dict]:
        initial_config: dict = test_entry.get("initial_config", {})
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        # This is only for the miss function category
        # A mapping from turn index to function to holdout
        holdout_function: dict[int, list] = test_entry.get("missed_function", {})

        total_input_token_count: list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency: list[list[float]] = []
        all_model_response: list[list] = (
            []
        )  # The model response that will be used for later evaluation
        all_inference_log: list[list[dict]] = (
            []
        )  # The debugging log for human to understand
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed.

        all_reasoning_content: list[list] = []

        # Execute no function call, but just to get a reference to all the instances to get the initial state for logging purpose
        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=("long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )

        if is_memory(test_category):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."

            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            test_entry["question"] = add_memory_instruction_system_prompt(
                test_entry["question"],
                test_category,
                test_entry["scenario"],
                memory_instance,
            )

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
                    continue
                # Avoid modification in future turns
                class_instance = deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if len(state_log) > 0:
                all_inference_log.append(state_log)

        inference_data: dict = {}
        inference_data = self._pre_query_processing_FC(inference_data, test_entry)
        inference_data = self._compile_tools(inference_data, test_entry)

        all_multi_turn_messages: list[list[dict]] = test_entry["question"]
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]

            if str(turn_idx) in holdout_function:
                test_entry["function"].extend(holdout_function[str(turn_idx)])
                # Since we have added new functions, we need to recompile the tools
                inference_data = self._compile_tools(inference_data, test_entry)
                assert (
                    len(current_turn_message) == 0
                ), "Holdout turn should not have user message."
                # TODO: Move this to before pre_query_processing_FC.
                # Shouldn't be happening in the inference loop.
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                    }
                ]

            if turn_idx == 0:
                inference_data = self.add_first_turn_message_FC(
                    inference_data, current_turn_message
                )
            else:
                inference_data = self._add_next_turn_user_message_FC(
                    inference_data, current_turn_message
                )

            current_turn_response = []
            current_turn_inference_log: list[dict] = {
                "begin_of_turn_query": current_turn_message
            }
            current_turn_input_token_count: list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_latency: list[float] = []
            current_turn_reasoning_content = []

            count = 0
            while True:
                tqdm.write("-" * 100)
                tqdm.write(
                    f"ID: {test_entry_id.replace('multi_turn_', '')}, Turn: {turn_idx}, Step: {count}"
                )
                current_step_inference_log: list[dict] = []
                # Add to the current_turn_inference_log at beginning of each step so that we don't need to bother dealing with the break statements
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log

                api_response, query_latency = self._query_FC(inference_data)

                # This part of logging is disabled by default because it is too verbose and will make the result file extremely large
                # It is only useful to see if the inference pipeline is working as expected (eg, does it convert all the inputs correctly)
                if include_input_log:
                    current_step_inference_log.append(
                        {
                            "role": "inference_input",
                            "content": inference_data.get("inference_input_log", ""),
                        }
                    )

                # Try parsing the model response
                model_response_data = self._parse_query_response_FC(api_response)
                model_responses = model_response_data["model_responses"]

                # Add the assistant message to the chat history
                inference_data = self._add_assistant_message_FC(
                    inference_data, model_response_data
                )

                # Process the metadata
                current_turn_input_token_count.append(model_response_data["input_token"])
                current_turn_output_token_count.append(model_response_data["output_token"])
                current_turn_latency.append(query_latency)

                current_turn_response.append(model_responses)

                reasoning_content = model_response_data.get("reasoning_content", "")
                current_turn_reasoning_content.append(reasoning_content)

                log_entry = {
                    "role": "assistant",
                    "content": model_responses,
                }
                if reasoning_content:
                    log_entry["reasoning_content"] = reasoning_content

                current_step_inference_log.append(log_entry)

                # Try decoding the model response
                try:
                    decoded_model_responses = self.decode_execute(
                        model_responses, has_tool_call_tag=False
                    )
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )

                    if is_empty_execute_response(decoded_model_responses):
                        tqdm.write("Empty response from the model. Proceed to next turn.")
                        current_step_inference_log.append(
                            {
                                "role": "handler_log",
                                "content": f"Empty response from the model. Proceed to next turn.",
                                "model_response_decoded": decoded_model_responses,
                            }
                        )
                        break

                except Exception as e:
                    tqdm.write("Failed to decode the model response. Proceed to next turn.")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Error decoding the model response. Proceed to next turn.",
                            "error": str(e),
                        }
                    )
                    break

                # Obtain the execution results
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name_underline_replaced,
                    test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )

                # Add the execution results to the chat history for the next turn
                inference_data = self._add_execution_results_FC(
                    inference_data, execution_results, model_response_data
                )

                for execution_result in execution_results:
                    current_step_inference_log.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                        }
                    )

                count += 1
                # Force quit after too many steps
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                        }
                    )

                    break

            # Add to the total list
            all_model_response.append(current_turn_response)
            all_inference_log.append(current_turn_inference_log)
            all_reasoning_content.append(current_turn_reasoning_content)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_latency.append(current_turn_latency)

            if not exclude_state_log:
                state_log = []
                for class_name, class_instance in involved_instances.items():
                    if (
                        class_name in STATELESS_CLASSES
                        or class_name in OMIT_STATE_INFO_CLASSES
                    ):
                        continue
                    # Avoid modification in future turns
                    class_instance = deepcopy(class_instance)
                    state_log.append(
                        {
                            "role": "state_info",
                            "class_name": class_name,
                            "content": {
                                key: value
                                for key, value in vars(class_instance).items()
                                if not key.startswith("_")
                            },
                        }
                    )
                if len(state_log) > 0:
                    all_inference_log.append(state_log)

            if force_quit:
                break

        # Special handling for the memory category
        # Need to flush the memory to local file at the end of the conversation
        if is_memory_prereq(test_entry_id):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."
            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            memory_instance._flush_memory_to_local_file()

        metadata = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
        }

        if not all(
            all(content == "" for content in single_turn_reasoning_content)
            for single_turn_reasoning_content in all_reasoning_content
        ):
            metadata["reasoning_content"] = all_reasoning_content

        return all_model_response, metadata

    @final
    def inference_multi_turn_prompting(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
        n_sample: int = None,
    ) -> tuple[list[list], dict]:
        initial_config: dict = test_entry.get("initial_config", {})
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        # This is only for the miss function category
        # A mapping from turn index to function to holdout
        holdout_function: dict[int, list] = test_entry.get("missed_function", {})

        total_input_token_count: list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency: list[list[float]] = []
        # The model response that will be used for later evaluation
        all_model_response: list[list] = []
        # Only for reasoning models, reasoning content will be stored as part of metadata and in inference log
        all_reasoning_content: list[list] = []
        # The debugging log for human to understand
        all_inference_log: list[list[dict]] = []
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed.

        # Execute no function call, but just to get a reference to all the instances to get the initial state for logging purpose
        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=("long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )

        if is_memory(test_category):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."

            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            test_entry["question"] = add_memory_instruction_system_prompt(
                test_entry["question"],
                test_category,
                test_entry["scenario"],
                memory_instance,
            )

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
                    continue
                # Avoid modification in future turns
                class_instance = deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if len(state_log) > 0:
                all_inference_log.append(state_log)

        inference_data: dict = self._pre_query_processing_prompting(test_entry)
        # 添加任务 ID 到 inference_data，用于打印系统提示词
        inference_data["test_entry_id"] = test_entry.get("id", "")
        inference_data["is_multi_turn"] = True

        # 在 _pre_query_processing_prompting 之后获取消息列表，因为 _pre_query_processing_prompting 可能会修改 test_entry["question"]
        # 但是需要深拷贝，避免后续处理时修改原始数据
        all_multi_turn_messages: list[list[dict]] = deepcopy(test_entry["question"])
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]
            if str(turn_idx) in holdout_function:
                assert (
                    len(current_turn_message) == 0
                ), "Holdout turn should not have user message."
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                            functions=holdout_function[str(turn_idx)]
                        ),
                    }
                ]

            if turn_idx == 0:
                inference_data = self.add_first_turn_message_prompting(
                    inference_data, current_turn_message
                )
            else:
                inference_data = self._add_next_turn_user_message_prompting(
                    inference_data, current_turn_message
                )

            current_turn_response = []
            current_turn_reasoning_content = []
            current_turn_inference_log: list[dict] = {
                "begin_of_turn_query": current_turn_message
            }
            current_turn_input_token_count: list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_latency: list[float] = []

            count = 0
            while True:
                print("-" * 100)
                print(
                    f"ID: {test_entry_id.replace('multi_turn_', '')}, Turn: {turn_idx}, Step: {count}"
                )
                current_step_inference_log: list[dict] = []
                # Add to the current_turn_inference_log at beginning of each step so that we don't need to bother dealing with the break statements
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log
                # print(f"Inference Data: {inference_data}")
                api_response, query_latency = self._query_prompting(inference_data)
                # print(f"api_response: {api_response}")
                # This part of logging is disabled by default because it is too verbose and will make the result file extremely large
                # It is only useful to see if the inference pipeline is working as expected (eg, does it convert all the inputs correctly)
                if include_input_log:
                    current_step_inference_log.append(
                        {
                            "role": "inference_input",
                            "content": inference_data.get("inference_input_log", ""),
                        }
                    )

                # Try parsing the model response
                model_response_data = self._parse_query_response_prompting(api_response)
                # print(f"model_response_data: {model_response_data}")

                model_responses = model_response_data["model_responses"]

                # Add the assistant message to the chat history
                inference_data = self._add_assistant_message_prompting(
                    inference_data, model_response_data
                )

                # Process the metadata
                current_turn_input_token_count.append(model_response_data["input_token"])
                current_turn_output_token_count.append(model_response_data["output_token"])
                current_turn_latency.append(query_latency)

                current_turn_response.append(model_responses)
                reasoning_content = model_response_data.get("reasoning_content", "")
                current_turn_reasoning_content.append(reasoning_content)

                log_entry = {
                    "role": "assistant",
                    "content": model_responses,
                }
                if reasoning_content:
                    log_entry["reasoning_content"] = reasoning_content

                current_step_inference_log.append(log_entry)

                # Try decoding the model response
                try:
                    decoded_model_responses = self.decode_execute(
                        model_responses, has_tool_call_tag=False
                    )
                    # print(f"decoded_model_responses: {decoded_model_responses}")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )

                    model_response_data["model_responses_decoded"] = decoded_model_responses
                    if is_empty_execute_response(decoded_model_responses):
                        tqdm.write("Empty response from the model. Proceed to next turn.")
                        current_step_inference_log.append(
                            {
                                "role": "handler_log",
                                "content": f"Empty response from the model. Proceed to next turn.",
                                "model_response_decoded": decoded_model_responses,
                            }
                        )
                        break

                except Exception as e:
                    tqdm.write("Failed to decode the model response. Proceed to next turn.")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Error decoding the model response. Proceed to next turn.",
                            "error": str(e),
                        }
                    )
                    break

                # Obtain the execution results
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name_underline_replaced,
                    test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )

                # Add the execution results to the chat history for the next turn
                inference_data = self._add_execution_results_prompting(
                    inference_data, execution_results, model_response_data
                )

                for execution_result in execution_results:
                    current_step_inference_log.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                        }
                    )

                count += 1
                # Force quit after too many steps
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                        }
                    )
                    break

            # Add to the total list
            all_model_response.append(current_turn_response)
            all_reasoning_content.append(current_turn_reasoning_content)
            all_inference_log.append(current_turn_inference_log)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_latency.append(current_turn_latency)

            if not exclude_state_log:
                state_log = []
                for class_name, class_instance in involved_instances.items():
                    if (
                        class_name in STATELESS_CLASSES
                        or class_name in OMIT_STATE_INFO_CLASSES
                    ):
                        continue
                    # Avoid modification in future turns
                    class_instance = deepcopy(class_instance)
                    state_log.append(
                        {
                            "role": "state_info",
                            "class_name": class_name,
                            "content": {
                                key: value
                                for key, value in vars(class_instance).items()
                                if not key.startswith("_")
                            },
                        }
                    )
                if len(state_log) > 0:
                    all_inference_log.append(state_log)

            if force_quit:
                break

        # Special handling for the memory category
        # Need to flush the memory to local file at the end of the conversation
        if is_memory_prereq(test_entry_id):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."
            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            memory_instance._flush_memory_to_local_file()

        metadata = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
        }
        # We only include reasoning content if it exists and is not empty
        if not all(
            all(content == "" for content in single_turn_reasoning_content)
            for single_turn_reasoning_content in all_reasoning_content
        ):
            metadata["reasoning_content"] = all_reasoning_content

        return all_model_response, metadata

    @final
    def inference_single_turn_FC(
        self, test_entry: dict, include_input_log: bool, n_sample: int = None
    ) -> tuple[any, dict]:
        inference_data: dict = {}
        inference_data = self._pre_query_processing_FC(inference_data, test_entry)
        inference_data = self._compile_tools(inference_data, test_entry)
        inference_data = self.add_first_turn_message_FC(
            inference_data, test_entry["question"][0]
        )

        # 打印fc 模型调用
        # print(f"inference_data: {inference_data}")

        api_response, query_latency = self._query_FC(inference_data, n_sample=n_sample)

        # Try parsing the model response
        model_response_data = self._parse_query_response_FC(api_response, n_sample=n_sample)

        # Process the metadata
        metadata = {}
        if include_input_log:
            metadata["inference_log"] = [
                {
                    "role": "inference_input",
                    "content": inference_data.get("inference_input_log", ""),
                }
            ]
        metadata["input_token_count"] = model_response_data["input_token"]
        metadata["output_token_count"] = model_response_data["output_token"]
        metadata["latency"] = query_latency

        if (
            "reasoning_content" in model_response_data
            and model_response_data["reasoning_content"] != ""
        ):
            metadata["reasoning_content"] = model_response_data["reasoning_content"]

        return model_response_data["model_responses"], metadata

    @final
    def inference_single_turn_prompting(
        self, test_entry: dict, include_input_log: bool, n_sample: int = None
    ) -> tuple[any, dict]:
        inference_data: dict = self._pre_query_processing_prompting(test_entry)
        inference_data = self.add_first_turn_message_prompting(
            inference_data, test_entry["question"][0]
        )
        # 添加任务 ID 到 inference_data，用于打印系统提示词
        inference_data["test_entry_id"] = test_entry.get("id", "")
        inference_data["is_multi_turn"] = False
        api_response, query_latency = self._query_prompting(inference_data, n_sample=n_sample)

        # Try parsing the model response
        model_response_data = self._parse_query_response_prompting(api_response, n_sample=n_sample)

        # Process the metadata
        metadata = {}
        if include_input_log:
            metadata["inference_log"] = [
                {
                    "role": "inference_input",
                    "content": inference_data.get("inference_input_log", ""),
                }
            ]
        metadata["input_token_count"] = model_response_data["input_token"]
        metadata["output_token_count"] = model_response_data["output_token"]
        metadata["latency"] = query_latency

        if (
            "reasoning_content" in model_response_data
            and model_response_data["reasoning_content"] != ""
        ):
            metadata["reasoning_content"] = model_response_data["reasoning_content"]

        return model_response_data["model_responses"], metadata

    def decode_ast(self, result, language: ReturnFormat, has_tool_call_tag: bool):
        """
        This method takes raw model output (from `_parse_query_response_xxx`) and convert it to standard AST checker input.
        """
        raise NotImplementedError

    def decode_execute(self, result, has_tool_call_tag: bool):
        """
        This method takes raw model output (from `_parse_query_response_xxx`) and convert it to standard execute checker input.
        """
        raise NotImplementedError

    @final
    def write(self, result, result_dir, update_mode=False, n_sample=None):
        # Use the internal registry name to decide the result directory to avoid
        # collisions between different variants that share the same API model name.
        model_result_dir = result_dir / self.registry_dir_name

        if isinstance(result, dict):
            result = [result]

        # Collect and format each entry for JSON compatibility
        entries_to_write = [make_json_serializable(entry) for entry in result]

        # Group entries by their `test_category` and `sample_idx` for efficient file handling
        file_entries = {}
        for entry in entries_to_write:
            test_category = extract_test_category_from_id(entry["id"])
            # Determine the high-level grouping folder (non_live, live, etc.)
            group_dir_name = get_directory_structure_by_id(entry["id"])
            group_dir_path = model_result_dir / group_dir_name
            group_dir_path.mkdir(parents=True, exist_ok=True)

            # Add suffix based on sample_idx if present, otherwise use n_sample
            if n_sample is not None:
                # If entry has sample_idx, use it to create separate files for each sample
                sample_idx = entry.get("sample_idx")
                if sample_idx is not None:
                    file_path = group_dir_path / f"{VERSION_PREFIX}_{test_category}_result_{sample_idx + 1}.json"
                else:
                    # When n_sample=1 and no sample_idx, use suffix _1.json
                    # When n_sample>1 and no sample_idx, use n_sample as suffix (backward compatibility)
                    file_path = group_dir_path / f"{VERSION_PREFIX}_{test_category}_result_{n_sample}.json"
            else:
                file_path = group_dir_path / f"{VERSION_PREFIX}_{test_category}_result.json"
            file_entries.setdefault(file_path, []).append(entry)

        for file_path, entries in file_entries.items():
            if update_mode:
                # Load existing entries from the file
                existing_entries = {}
                if file_path.exists():
                    existing_entries = {
                        entry["id"]: entry for entry in load_file(file_path)
                    }

                # Update existing entries with new data
                for entry in entries:
                    existing_entries[entry["id"]] = entry

                # Sort entries by `id` and write them back to ensure order consistency
                sorted_entries = sorted(existing_entries.values(), key=sort_key)
                with open(file_path, "w") as f:
                    for entry in sorted_entries:
                        content = json.dumps(entry) + "\n"
                        f.write(content)
                        f.flush()

            else:
                # Normal mode: Append to the end of the file
                # Note: We will sort all the entries at the end of the generation pipeline to ensure the order is consistent
                entries.sort(key=sort_key)
                with open(file_path, "a") as f:
                    for entry in entries:
                        content = json.dumps(entry) + "\n"
                        f.write(content)
                        f.flush()

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        """
        Call the model API in FC mode to get the response.
        Return the response object that can be used to feed into the `_parse_query_response_FC` method.
        """
        raise NotImplementedError

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        This might includes transforming the input user message into the format expected by the model, extract out the system prompt (if any), and any other necessary preprocessing steps. Those steps can also be done in the `add_first_turn_message_FC` and `_add_next_turn_user_message_FC` methods, but it's usually cleaner to do it here.
        The inference_data dict is updated in place and returned.

        Note: This method has different signature from its Prompting version.
        """
        raise NotImplementedError

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """
        [Only for FC mode]
        This method is used to prepare/compile the tools from the test entry and add them to the inference data to use for model query in FC mode.
        Function docs usually need to be transformed to the format expected by the model, done through the `convert_to_tool` function from `model_handler/utils.py`.
        The inference_data dict is updated in place and returned.
        """
        raise NotImplementedError

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - tool_call_ids (list[str]): The IDs of the tool calls that are generated by the model. Optional.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history, in the format that the model expects.

        Args:
            inference_data (dict): The inference data from previous processing steps.
            first_turn_message (list[dict]): The first turn message from the test entry. It has variable length. It might contain one or more of the following roles:
                - "system": The system message. This role will only appear at most once, at the beginning of the first turn. For most entry, this role will not appear.
                - "user": The user message.
                - "assistant": The assistant message. For most entry, this role will not appear.

        Returns:
            inference_data (dict): The updated inference data that will be send to `_query_FC` to call the model API.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next turn user message to the chat history for query.
        user_message is a list of 1 element, which is guaranteed to be a `user` role message.
        """
        raise NotImplementedError

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_FC(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        Some models may need to add additional information to the chat history, such as tool call IDs.
        """
        raise NotImplementedError

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        """
        Call the model API in prompting mode to get the response.
        Return the response object that can be used to feed into the decode method.
        """
        raise NotImplementedError

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        This might includes transforming the input user message into the format expected by the model, extract out the system prompt (if any), and any other necessary preprocessing steps. Those steps can also be done in the `add_first_turn_message_prompting` and `_add_next_turn_user_message_prompting` methods, but it's usually cleaner to do it here.
        The function docs are usually supplied to the prompting models as part of the system prompt, done via the `system_prompt_pre_processing_chat_model` function from `model_handler/utils.py`, unless the model has a different way of handling it.
        Returns a dict that contains all the necessary information for the query method.
        Things like `system_prompt` and `chat_history` are optional, specific to the model.

        Note: This method has different signature from its FC version.
        """
        raise NotImplementedError

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history, in the format that the model expects.

        Args:
            inference_data (dict): The inference data from previous processing steps.
            first_turn_message (list[dict]): The first turn message from the test entry. It has variable length. It might contain one or more of the following roles:
                - "system": The system message. This role will only appear at most once, at the beginning of the first turn.
                - "user": The user message.
                - "assistant": The assistant message. For most entry, this role will not appear.

        Returns:
            inference_data (dict): The updated inference data that will be send to `_query_prompting` to call the model API.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next turn user message to the chat history for query.
        user_message is a list of 1 element, which is guaranteed to be a `user` role message.
        """
        raise NotImplementedError

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        By default, execution results are added back as a `user` role message, as most models don't support the `tool` role in prompting mode.
        """
        raise NotImplementedError
