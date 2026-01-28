import json
import asyncio
from collections import defaultdict
from typing import List
from datetime import datetime
from flowllm import C, BaseAsyncOp
from flowllm.enumeration.role import Role
from flowllm.schema.message import Message
from flowllm.schema.vector_node import VectorNode
from flowllm.utils.common_utils import extract_content
from loguru import logger

from reme_ai.schema.memory import ToolMemory, ToolCallResult, vector_node_to_memory
from reme_ai.utils.op_utils import handle_429_error

MAX_CONTEXT = 65536
@C.register_op()
class ParseToolCallResultOp(BaseAsyncOp):
    file_path: str = __file__

    def __init__(self,
                 max_history_tool_call_cnt: int = 100,
                 evaluation_sleep_interval: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_history_tool_call_cnt: int = max_history_tool_call_cnt
        self.evaluation_sleep_interval: float = evaluation_sleep_interval

    @staticmethod
    def _is_valid_tool_call_result(tool_call_result: ToolCallResult) -> bool:
        """
        Validate if a tool call result is valid and should be processed.
        
        Args:
            tool_call_result: The tool call result to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check if tool_name is provided
        if not tool_call_result.tool_name or not tool_call_result.tool_name.strip():
            logger.warning("Skipping tool_call_result: tool_name is empty")
            return False

        # Check if output is provided (empty output means no useful result)
        if not tool_call_result.output or not str(tool_call_result.output).strip():
            logger.warning(f"Skipping tool_call_result for {tool_call_result.tool_name}: output is empty")
        #     return False
        
        # if not tool_call_result.config or not str(tool_call_result.config).strip():
        #     logger.warning(f"Skipping tool_call_result for {tool_call_result.tool_name}: config is empty")
        #     return False

        # Check if input is provided
        if tool_call_result.input is None:
        # or (
        #         isinstance(tool_call_result.input, str) and not tool_call_result.input.strip()
        # ) or (
        #         isinstance(tool_call_result.input, dict) and not tool_call_result.input
        # ):
            logger.warning(f"Skipping tool_call_result for {tool_call_result.tool_name}: input is empty")
            # return False

        return True

    @staticmethod
    def _estimate_token_cost(tool_call_result: ToolCallResult) -> int:
        """
        Estimate token cost based on input and output length.
        Uses the approximation: token_count ≈ char_count / 4
        
        Args:
            tool_call_result: The tool call result to estimate tokens for
            
        Returns:
            Estimated token count as an integer
        """
        # Convert input to string
        if isinstance(tool_call_result.input, dict):
            input_str = str(tool_call_result.input)
        else:
            input_str = str(tool_call_result.input)

        # Get output string
        output_str = str(tool_call_result.output)

        # Calculate total character count
        total_chars = len(input_str) + len(output_str)

        # Estimate tokens (1 token ≈ 4 characters for English text)
        estimated_tokens = total_chars // 4

        return estimated_tokens

    @staticmethod
    def _parse_create_time(create_time: str) -> datetime:
        """
        Parse create_time string to datetime object.
        Supports common formats like "YYYY-MM-DD HH:MM:SS"
        """
        if not create_time or not create_time.strip():
            return datetime.min
        
        try:
            # Try common format: "YYYY-MM-DD HH:MM:SS"
            return datetime.strptime(create_time.strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                # Try ISO format
                return datetime.fromisoformat(create_time.strip())
            except ValueError:
                logger.warning(f"Unable to parse create_time: {create_time}, using datetime.min")
                return datetime.min

    @staticmethod
    def _clean_tool_calls_by_time_window(tool_call_results: List[ToolCallResult]) -> int:
        """
        Clean tool calls based on time window analysis.
        
        Process:
        1. Group tool calls by create_time
        2. Sort time stamps and calculate average score for each timestamp
        3. For each timestamp, if the following 3 consecutive timestamps have lower average scores,
           delete all tool calls with is_summarized=True at that timestamp
        
        Args:
            tool_call_results: List of tool call results to process
            
        Returns:
            Number of tool calls deleted
        """
        if not tool_call_results:
            return 0
        
        # Group by create_time
        calls_by_timestamp = defaultdict(list)
        
        for call in tool_call_results:
            timestamp = call.create_time
            calls_by_timestamp[timestamp].append(call)
        
        # Sort timestamps
        sorted_timestamps = sorted(
            calls_by_timestamp.keys(),
            key=lambda t: ParseToolCallResultOp._parse_create_time(t)
        )
        
        if len(sorted_timestamps) < 4:  # Need at least 4 timestamps (current + 3 following)
            return 0
        
        # Calculate average score for each timestamp
        timestamp_avg_scores = {}
        for timestamp in sorted_timestamps:
            calls = calls_by_timestamp[timestamp]
            if calls:
                avg_score = sum(call.score for call in calls) / len(calls)
                timestamp_avg_scores[timestamp] = avg_score
        
        # Find timestamps to clean
        timestamps_to_clean = []
        for i in range(len(sorted_timestamps) - 3):
            current_timestamp = sorted_timestamps[i]
            current_avg = timestamp_avg_scores[current_timestamp]
            
            # Check if next 3 consecutive timestamps all have lower average scores
            following_timestamps = sorted_timestamps[i+1:i+4]
            if len(following_timestamps) == 3:
                all_lower = all(
                    timestamp_avg_scores[ts] < current_avg
                    for ts in following_timestamps
                )
                if all_lower:
                    timestamps_to_clean.append(current_timestamp)
        
        # Collect tool calls to delete (using set for O(1) lookup)
        calls_to_delete = set()
        for timestamp in timestamps_to_clean:
            calls = calls_by_timestamp[timestamp]
            for call in calls:
                if call.is_summarized:
                    calls_to_delete.add(id(call))  # Use id() to handle object identity
                    logger.debug(
                        f"Marked for deletion: tool call with is_summarized=True at timestamp {timestamp}, "
                        f"call ={call}"
                    )
        
        # Remove deleted calls from the original list (in-place modification)
        original_count = len(tool_call_results)
        tool_call_results[:] = [
            call for call in tool_call_results
            if id(call) not in calls_to_delete
        ]
        deleted_count = original_count - len(tool_call_results)
        
        if deleted_count > 0:
            logger.info(
                f"Time window cleaning: deleted {deleted_count} tool calls with is_summarized=True "
                f"from {len(timestamps_to_clean)} timestamps"
            )
        
        return deleted_count

    @staticmethod
    def _format_tool_memories_summary(memory_list: List[ToolMemory], deleted_memory_ids: List[str]) -> str:
        """Format tool memories update summary"""
        lines = []
        
        # 统计信息
        total_tools = len(memory_list)
        updated_tools = len(deleted_memory_ids)
        new_tools = total_tools - updated_tools
        
        lines.append(f"Processed {total_tools} tool(s): {updated_tools} updated, {new_tools} newly created\n")
        
        # 详细信息
        for idx, memory in enumerate(memory_list, 1):
            is_updated = memory.memory_id in deleted_memory_ids
            status = "Updated" if is_updated else "New"
            
            lines.append(f"[{status}] {memory.when_to_use}")
            lines.append(f"  Total calls: {len(memory.tool_call_results)}")
            
            # 显示最近添加的调用结果统计
            if memory.tool_call_results:
                recent_results = memory.tool_call_results[-3:]
                success_count = sum(1 for r in recent_results if r.success)
                avg_score = sum(r.score for r in recent_results) / len(recent_results)
                lines.append(f"  Recent calls: {success_count}/{len(recent_results)} successful, avg score: {avg_score:.2f}")
            
            if idx < len(memory_list):
                lines.append("")
        
        return "\n".join(lines)

    async def _evaluate_single_tool_call(self, tool_call_result: ToolCallResult, index: int) -> ToolCallResult:
        await asyncio.sleep(self.evaluation_sleep_interval * index)
        cut_num = MAX_CONTEXT//3
        tool_config = ""
        if len(tool_call_result.config)==0:
            logger.warning(f"Tool call result for {tool_call_result.tool_name} has empty config.")
        else:
            try :
                with open(tool_call_result.config, 'r') as f:
                    tool_configs = json.load(f)
                    tool_config = str(json.dumps(tool_configs.get(tool_call_result.tool_name, {}), ensure_ascii=False))
            except Exception as e:
                logger.error(f"Failed to load tool config from {tool_call_result.config}: {e}")

        # 注意：source_task 字段不参与评估过程，只用于后续的总结过滤
        prompt = self.prompt_format(
            prompt_name="evaluate_tool_call_prompt",
            tool_name=tool_call_result.tool_name,
            context=tool_call_result.context[-cut_num:-1] if len(tool_call_result.context) > cut_num else tool_call_result.context,
            input_params=str(tool_call_result.input),
            output=tool_call_result.output,
            success_flag=str(tool_call_result.success),
            time_cost=tool_call_result.time_cost,
            token_cost=tool_call_result.token_cost,
            config= tool_config[-cut_num:-1] if len(tool_call_result.config) > cut_num else tool_config,
            )
        
        def parse_evaluation(message: Message) -> ToolCallResult:
            content = message.content.strip()
            eval_data = extract_content(content, "json")

            # 更新 tool_call_result - 包含 summary, evaluation 和 score
            # tool_call_result.summary = eval_data.get("summary", "")
            tool_call_result.evaluation = eval_data.get("explanation", "")
            scores = eval_data.get("scores", [0]*10)
            tool_call_result.scores = scores
            tool_call_result.score = sum(scores)
            # 传入的是符合config，也符合上下文的，是config 或者 工具本身的问题
            # if (tool_call_result.score >=6) and (sum(tool_call_result.scores[:6]) == 6) and tool_call_result.success:
            #     tool_call_result.tool_problem = (1 and eval_data.get("is_tool_problem", 0))
            # else:
            #     tool_call_result.tool_problem = 0
            tool_call_result.success = (tool_call_result.success and tool_call_result.score >=9)

            # 验证 score 是否符合 2 档要求 (0.0, 1.0)
            # if tool_call_result.score not in [0.0, 1.0]:
            #     if tool_call_result.score < 0.5:
            #         tool_call_result.score = 0.0
            #     else:
            #         tool_call_result.score = 1.0

            # 打印完整的prompt和result
            # logger.info(f"\n{'='*80}\nLLM Evaluation [Index {index}]\n{'='*80}\n"
            #            f"PROMPT:\n{prompt}\n\n"
            #            f"RESULT:\n{content}\n"
            #            f"{'='*80}\n")
            
            return tool_call_result

        # 调用 LLM 进行评估，处理 429 错误
        async def llm_call():
            return await self.llm.achat(messages=[Message(role=Role.USER, content=prompt)], callback_fn=parse_evaluation)
        
        result = await handle_429_error(llm_call)
        
        return result

    async def async_execute(self):
        tool_call_results: list = self.context.get("tool_call_results", [])
        tool_call_results = [ToolCallResult(**x) if isinstance(x, dict) else x for x in tool_call_results]
        workspace_id: str = self.context.workspace_id

        if not tool_call_results:
            self.context.response.answer = "No valid tool_call_results"
            self.context.response.success = False
            return

        # 过滤和验证 tool_call_results
        original_count = len(tool_call_results)
        valid_tool_call_results = []
        
        for tool_call_result in tool_call_results:
            # 验证数据是否有效
            if not self._is_valid_tool_call_result(tool_call_result):
                continue

            # 确保有 hash 值
            tool_call_result.ensure_hash()

            # 如果 token_cost 未设置（默认值 -1），则自动计算
            if tool_call_result.token_cost < 0:
                token_cost = self._estimate_token_cost(tool_call_result)
                tool_call_result.token_cost = token_cost
                logger.info(f"Auto-calculated token_cost={token_cost} for tool={tool_call_result.tool_name}")

            # source_task 字段会被自动保存（通过 ToolCallResult(**x)），但不参与评估过程
            if tool_call_result.source_task:
                logger.debug(f"Tool call result for {tool_call_result.tool_name} has source_task={tool_call_result.source_task}")

            valid_tool_call_results.append(tool_call_result)

        # 记录过滤统计
        filtered_count = original_count - len(valid_tool_call_results)
        if filtered_count > 0:
            logger.warning(f"Filtered out {filtered_count} invalid tool_call_results out of {original_count}")

        # 如果所有记录都被过滤了
        if not valid_tool_call_results:
            self.context.response.answer = f"All {original_count} tool_call_results were invalid and filtered out"
            self.context.response.success = False
            return

        logger.info(f"Processing {len(valid_tool_call_results)} valid tool_call_results")

        # 预评估去重：基于已有 ToolMemory 的 call_hash，以及本批次内的重复
        # 1) 拉取每个工具已有的 memory（若存在），形成已有哈希集合
        existing_hashes_by_tool: dict[str, set[str]] = {}
        existing_configs_by_tool: dict[str, str] = {}
        tool_names = {r.tool_name for r in valid_tool_call_results}
        for tool_name in tool_names:
            try:
                nodes: List[VectorNode] = await self.vector_store.async_search(
                    query=tool_name, workspace_id=workspace_id, top_k=1
                )
                existing_hashes: set[str] = set()
                if nodes:
                    top_node = nodes[0]
                    memory: ToolMemory = vector_node_to_memory(top_node)
                    if isinstance(memory, ToolMemory) and memory.when_to_use == tool_name:
                        existing_hashes = {r.call_hash for r in memory.tool_call_results if r.call_hash}
                        existing_configs_by_tool[tool_name] = memory.config
                existing_hashes_by_tool[tool_name] = existing_hashes
            except Exception as e:
                logger.exception(f"Error fetching existing memory for tool '{tool_name}': {e}")
                existing_hashes_by_tool[tool_name] = set()

        # 2) 过滤掉重复的调用（包含：与历史重复、本批内互相重复）
        to_evaluate: List[ToolCallResult] = []
        pre_eval_dedup_count = 0
        for r in valid_tool_call_results:
            # 确保已有对应工具的哈希集合
            tool_set = existing_hashes_by_tool.setdefault(r.tool_name, set())
            # 配置变更跳过
            if (existing_configs_by_tool.get(r.tool_name, "")!="") and (r.config != existing_configs_by_tool.get(r.tool_name, "")):
                pre_eval_dedup_count += 1
                logger.info(f"Config changed for {r.tool_name}, please check it.")
                continue
            # 跳过重复
            if r.call_hash in tool_set:
                pre_eval_dedup_count += 1
                logger.info(f"Pre-eval dedup: skip duplicate for {r.tool_name} with hash {r.call_hash}")
                continue
            # 记录到评估列表，并加入集合，避免同批次内重复
            to_evaluate.append(r)
            existing_configs_by_tool.setdefault(r.tool_name, r.config)
            if r.call_hash:
                tool_set.add(r.call_hash)

        if pre_eval_dedup_count > 0:
            logger.info(
                f"Pre-evaluation deduplication skipped {pre_eval_dedup_count} duplicate tool calls; "
                f"remaining {len(to_evaluate)} to evaluate"
            )

        # 使用基类的 submit_async_task 提交所有评估任务（仅非重复项）
        for index, tool_call_result in enumerate(to_evaluate):
            # paper修改
            self.submit_async_task(self._evaluate_single_tool_call, tool_call_result, index)

        # 使用基类的 join_async_task 等待所有任务完成
        # 注意: 基类已经过滤掉异常,返回的只包含成功的结果
        evaluated_results = await self.join_async_task(return_exceptions=True)

        tool_results_by_name = defaultdict(list)
        for result in evaluated_results:
            # 清空 config，节省存储空间
            result.config = ""
            tool_results_by_name[result.tool_name].append(result)

        # 处理每个 tool_name 的结果
        all_memory_list = []
        all_deleted_memory_ids = []
        deduplication_stats = {"total_new": len(valid_tool_call_results), "added": 0, "skipped_pre_evaluation": pre_eval_dedup_count}

        for tool_name, tool_call_results in tool_results_by_name.items():
            nodes: List[VectorNode] = await self.vector_store.async_search(query=tool_name,
                                                                           workspace_id=workspace_id,
                                                                           top_k=1)

            tool_memory: ToolMemory | None = None
            exist_node: bool = False

            if nodes:
                top_node = nodes[0]
                memory: ToolMemory = vector_node_to_memory(top_node)

                # 确保是 ToolMemory 类型且 when_to_use 与 tool_name 匹配
                if isinstance(memory, ToolMemory) and memory.when_to_use == tool_name:
                    tool_memory = memory
                    exist_node = True

            # 如果没有找到匹配的 memory，创建新的
            if tool_memory is None:
                tool_memory = ToolMemory(workspace_id=workspace_id, when_to_use=tool_name, config=existing_configs_by_tool[tool_name])
            
            # 直接追加评估后的结果（已在评估前完成去重）
            tool_memory.tool_call_results.extend(tool_call_results)
            deduplication_stats["added"] += len(tool_call_results)

            # 基于时间窗口清理 tool calls
            # 按 create_time 排序，计算每个时间戳的平均得分
            # 如果某个时间戳后面连续3个时间戳的平均得分都比它低，删除该时间戳下所有 is_summarized=True 的 toolcall
            deleted_by_time_window = self._clean_tool_calls_by_time_window(tool_memory.tool_call_results)
            if deleted_by_time_window > 0:
                logger.info(f"Time window cleaning for {tool_name}: deleted {deleted_by_time_window} tool calls")

            # 保留最近的 n 个
            if len(tool_memory.tool_call_results) > self.max_history_tool_call_cnt:
                tool_memory.tool_call_results = tool_memory.tool_call_results[-self.max_history_tool_call_cnt:]

            # 更新修改时间
            tool_memory.update_modified_time()

            # 如果是更新现有的 memory，需要先删除旧的
            if exist_node:
                all_deleted_memory_ids.append(tool_memory.memory_id)

            all_memory_list.append(tool_memory)

        # 格式化结果信息
        formatted_answer = self._format_tool_memories_summary(all_memory_list, all_deleted_memory_ids)

        # 添加过滤统计信息
        if filtered_count > 0:
            filter_info = (f"\n\nValidation Summary:\n"
                           f"  Total input: {original_count}\n"
                           f"  Valid: {len(valid_tool_call_results)}\n"
                           f"  Filtered (invalid): {filtered_count}")
            formatted_answer += filter_info
        
        # 添加去重统计信息
        dedup_info = (f"\n\nDeduplication Summary:\n"
                      f"  Skipped before evaluation: {deduplication_stats['skipped_pre_evaluation']}\n"
                      f"  Total new calls (evaluated): {deduplication_stats['total_new']}\n"
                      f"  Added to memory: {deduplication_stats['added']}")
        formatted_answer += dedup_info
        
        # 设置返回结果
        self.context.response.answer = formatted_answer
        self.context.response.success = True
        self.context.response.metadata["deleted_memory_ids"] = all_deleted_memory_ids
        self.context.response.metadata["memory_list"] = all_memory_list
        self.context.response.metadata["validation_stats"] = {
            "total_input": original_count,
            "valid": len(valid_tool_call_results),
            "filtered": filtered_count
        }
        self.context.response.metadata["deduplication_stats"] = deduplication_stats


async def main():
    """Simple test for ParseToolCallResultOp"""
    from datetime import datetime

    from reme_ai.app import ReMeApp

    async with ReMeApp():
        op = ParseToolCallResultOp()

        # Create test data with different scenarios
        tool_call_results = [
            # Test 1: Valid with explicit token_cost
            {
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tool_name": "test_tool_with_token",
                "context": "user: search python asyncio docs\nsystem: use web_search tool\nuser: please get official pages only\n",
                "input": {
                    "query": "search for python asyncio documentation",
                    "max_results": 10,
                    "filter_type": "official_docs",
                    "language": "en"
                },
                "output": "Found 10 relevant documentation pages for Python asyncio. Top results include: 1) Official Python docs for asyncio module, 2) Real Python asyncio tutorial, 3) Stack Overflow asyncio examples. All results are from official sources as requested.",
                "token_cost": 150,
                "success": True,
                "time_cost": 2.3
            },
            # Test 2: Valid without token_cost (should auto-calculate)
            {
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tool_name": "test_tool_auto_token",
                "context": "user: get weather for SF\nsystem: clarify units if needed\nuser: use celsius please\n",
                "input": {
                    "query": "get weather information for San Francisco",
                    "units": "celsius"
                },
                "output": "Current weather in San Francisco: Temperature: 18°C, Humidity: 65%, Conditions: Partly cloudy",
                # token_cost not provided, will be auto-calculated
                "success": True,
                "time_cost": 1.5
            },
            # Test 3: Invalid - empty output (should be filtered)
            {
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tool_name": "invalid_tool_empty_output",
                "context": "user: quick test\n",
                "input": {"query": "test"},
                "output": "",  # Empty output
                "success": False,
                "time_cost": 0.1
            },
            # Test 4: Invalid - empty input (should be filtered)
            {
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tool_name": "invalid_tool_empty_input",
                "context": "user: try calling with empty input\n",
                "input": {},  # Empty dict input
                "output": "Some output",
                "success": True,
                "time_cost": 0.2
            },
            # Test 5: Invalid - missing tool_name (should be filtered)
            {
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tool_name": "",  # Empty tool name
                "context": "user: missing tool name\n",
                "input": {"test": "data"},
                "output": "Result",
                "success": True,
                "time_cost": 0.3
            }
        ]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing with {len(tool_call_results)} tool_call_results")
        logger.info(f"Expected: 2 valid, 3 filtered")
        logger.info(f"{'=' * 60}\n")
        workspace_id = "test_workspace1"

        await op.async_call(tool_call_results=tool_call_results, workspace_id=workspace_id)
        logger.info(f"Response: {op.context.response.model_dump_json()}")


if __name__ == "__main__":
    asyncio.run(main())