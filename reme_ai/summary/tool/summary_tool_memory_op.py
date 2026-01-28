import json
import asyncio
from math import floor
import random
from typing import List, Union
from collections import defaultdict

from flowllm import C, BaseAsyncOp
from flowllm.enumeration.role import Role
from flowllm.schema.message import Message
from flowllm.schema.vector_node import VectorNode
from flowllm.utils.common_utils import extract_content
from loguru import logger

from reme_ai.schema.memory import ToolMemory, vector_node_to_memory
from reme_ai.utils.op_utils import handle_429_error




MAX_CONTEXT = 65536   # 上下文长度上限

@C.register_op()
class SummaryToolMemoryOp(BaseAsyncOp):
    file_path: str = __file__

    def __init__(self,
                 data_from: str = "",
                 summary_sleep_interval: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_from: str = data_from
        self.summary_sleep_interval: float = summary_sleep_interval

    def _format_summary_result(self, summarized_memories: List[ToolMemory], skipped_memories: List[ToolMemory]) -> str:
        """Format tool memory summary result"""
        lines = []
        
        # 统计信息
        total_tools = len(summarized_memories) + len(skipped_memories)
        lines.append(f"Processed {total_tools} tool(s): {len(summarized_memories)} summarized, {len(skipped_memories)} skipped\n")
        
        # 显示已总结的工具详细信息
        if summarized_memories:
            for idx, memory in enumerate(summarized_memories, 1):
                lines.append(f"Tool: {memory.when_to_use}")
                lines.append(memory.content)
                
                if idx < len(summarized_memories):
                    lines.append("\n---\n")
        
        return "\n".join(lines)

    # @staticmethod
    # def _sample_calls_by_score_distribution(filtered_calls: List) -> List:
    #     """
    #     根据 scores 分布对 recent_calls 进行采样，每个不同的得分分布随机保留1条。
    #     scores 有10个维度，任何一个维度不同都算不同分布。
        
    #     Args:
    #         filtered_calls: 过滤后的工具调用结果列表
            
    #     Returns:
    #         采样后的调用结果列表，每个不同的 scores 分布随机保留1条
    #     """
    #     if not filtered_calls:
    #         return []
        
    #     # 按 scores 分组
    #     # 将 scores 转换为 tuple 以便作为字典的 key（list 不可哈希）
    #     scores_groups = {}
    #     if len(filtered_calls) <= MIN_CALL_COUNT:
    #         return filtered_calls
            
    #     for call in filtered_calls:
    #         # 获取 scores
    #         scores = getattr(call, 'scores', []) or []
    #         # 将 scores 转换为 tuple 以便作为字典的 key（list 不可哈希）
    #         scores_tuple = tuple(scores)
    #         # 按 scores 分组
    #         if scores_tuple not in scores_groups:
    #             scores_groups[scores_tuple] = []
    #         scores_groups[scores_tuple].append(call)
    #     # 对每个分组随机选择一个
    #     sampled_calls = []
    #     for scores_tuple, calls in scores_groups.items():
    #         if len(calls) >= max(len(filtered_calls)/2, MIN_CALL_COUNT):
    #             sampled_call = random.choices(calls, k=MIN_CALL_COUNT)
    #             sampled_calls.extend(sampled_call)
    #         else:
    #             sampled_call = random.choice(calls)
    #             sampled_calls.append(sampled_call)
    #     return sampled_calls


    @staticmethod
    def _compute_Q(calls: List) -> int:
        """
        计算论文中对应的 Q：
        在总长度不超过 MAX_CONTEXT 的前提下，按「从长到短」贪心能容纳的最大样本数。
        """
        if not calls:
            return 0

        # 假设 len(call) 近似表示该调用在上下文中占用的长度
        calls_with_len = [(call, len(call)) for call in calls]
        calls_with_len.sort(key=lambda x: x[1], reverse=True)

        total_len = 0
        m = 0
        for _, l in calls_with_len:
            if total_len + l <= MAX_CONTEXT:
                total_len += l
                m += 1
            else:
                break
        return m

    @staticmethod
    def _sample_calls_by_score_distribution(filtered_calls: List) -> List:
        """
        根据 scores 分布 + 上下文长度约束进行采样：
        1. 在总长度不超过 MAX_CONTEXT 前提下，尽量用满一个安全上界 m 条样本；
        2. 每种 scores 分布至少保留 1 条（如果该类有样本）；
        3. 在此基础上，各分布样本数量尽量接近原始分布比例。
        
        Args:
            filtered_calls: 过滤后的工具调用结果列表
            
        Returns:
            采样后的调用结果列表
        """
        if not filtered_calls:
            return []

        # ---------- 1. 计算安全上界 m（按长度从长到短贪心） ----------
        # 对应论文中的 Q
        m = SummaryToolMemoryOp._compute_Q(filtered_calls)

        # 如果一个都放不下，直接返回空
        if m == 0 or len(filtered_calls)==1 or m==1:
            return []

        # 如果总数本来就 <= m，那就不必再复杂采样了，直接全返回即可
        if len(filtered_calls) <= m:
            return filtered_calls

        # ---------- 2. 按 scores 分组 ----------
        # scores 作为 tuple 用作 dict key
        score_groups = defaultdict(list)
        for call in filtered_calls:
            scores = getattr(call, 'scores', []) or []
            scores_tuple = tuple(scores)
            score_groups[scores_tuple].append(call)

        # 所有类别（score 组合）
        all_score_keys = list(score_groups.keys())
        c = len(all_score_keys)

        # ---------- 3. 特殊情况：m < c，只能选 m 个类别，各 1 条 ----------
        if m < c:
            # 简单策略：按照类别大小排序，优先选择样本多的类别
            # 你也可以按其他权重排序（比如 avg len，更重要的 score 等）
            sorted_keys = sorted(
                all_score_keys,
                key=lambda k: len(score_groups[k]),
                reverse=True
            )
            chosen_keys = sorted_keys[:m]

            sampled_calls = []
            for k in chosen_keys:
                calls = score_groups[k]
                # 这里随机选 1 条，也可以按 len 选最短的
                sampled_calls.append(random.choice(calls))

            return sampled_calls

        # ---------- 4. m >= c：每类至少 1 条，剩余 R = m - c ----------
        R = m - c

        # 统计每类样本数 count_k，并计算占比 p_k
        counts = []
        for k in all_score_keys:
            counts.append(len(score_groups[k]))  # count_k

        total_n = sum(counts)
        # 避免除零
        if total_n == 0:
            return []

        # 计算 p_k
        p_list = [cnt / total_n for cnt in counts]  # p_k

        # ---------- 5. 根据 p_k 将 R 个“额外名额”分配到各类 ----------
        # 5.1 计算连续理想值 raw_k = R * p_k
        raw = [R * p for p in p_list]

        # 5.2 初步整数部分 extra_k = floor(raw_k)
        extra = [floor(x) for x in raw]
        used = sum(extra)
        remain = R - used

        # 5.3 将 remain 个配额按小数部分分配
        if remain > 0:
            # 计算小数部分
            frac = [raw[i] - extra[i] for i in range(c)]
            # 按小数部分从大到小排序索引
            indices = list(range(c))
            indices.sort(key=lambda i: frac[i], reverse=True)

            # 给前 remain 个类别的 extra_k + 1
            for i in indices[:remain]:
                extra[i] += 1

        # 5.4 得到每类最终配额：quota_k = 1 + extra_k
        quota = [1 + e for e in extra]  # 对应 all_score_keys 的顺序
        # 按构造应满足 sum(quota) == m

        # ---------- 6. 处理每类实际样本数限制：quota_k <= count_k ----------
        # 如果某类 quota_k > count_k，则截断
        for i in range(c):
            count_k = counts[i]
            if quota[i] > count_k:
                quota[i] = count_k

        # 剩余配额（对齐 test_sample_method.py 的写法）：pool = Q - sum(quota)
        # 其中 Q 对应本函数的目标采样总数 m
        pool = m - sum(quota)

        # 若 pool > 0，则尝试在还有空间的类别中重新分配
        # 简单策略：按照 p_k 大的优先，再看剩余容量
        if pool > 0:
            # 可以多分的类别索引集合
            available_indices = [i for i in range(c) if quota[i] < counts[i]]
            if available_indices:
                # 按 p_k 从大到小排序这些有空间的类别
                available_indices.sort(key=lambda i: p_list[i], reverse=True)
                idx_cycle = 0
                # 简单轮询，把 pool 分掉，直到没有空间或 pool 用完
                while pool > 0 and available_indices:
                    i = available_indices[idx_cycle]
                    if quota[i] < counts[i]:
                        quota[i] += 1
                        pool -= 1
                    # 移动到下一个索引
                    idx_cycle = (idx_cycle + 1) % len(available_indices)
                    # 如果这一轮中所有 available 都满了，就停止
                    if all(quota[j] >= counts[j] for j in available_indices):
                        break
            # 如果实在分不出去，pool 剩余无处可去，则总数 < m，但仍不会超过 m

        # ---------- 7. 在每类内部具体采样 ----------
        sampled_calls = []
        for i, key in enumerate(all_score_keys):
            k_quota = quota[i]
            if k_quota <= 0:
                continue
            calls = score_groups[key]

            # 如果该类样本数 <= 配额，全取
            if len(calls) <= k_quota:
                sampled_calls.extend(calls)
            else:
                # 否则随机抽样 k_quota 条
                # 如果想倾向于短的，可以先按 len 排序，再从前面随机 / 直接取前 k_quota
                sampled_calls.extend(random.sample(calls, k_quota))

        # sampled_calls 的条数理论上 <= m，并且在分配时尽量接近 m
        return sampled_calls

    @staticmethod
    def _format_call_summaries_markdown(recent_calls: List) -> str:
        """Format tool call summaries as markdown."""
        if not recent_calls:
            return "No recent calls available."

        lines = []
        for i, call in enumerate(recent_calls, 1):
            lines.append(f"### Call #{i}")
            # lines.append(f"- **Summary**: {call.summary}")
            lines.append(f"- **Evaluation**: {call.evaluation}")
            lines.append(f"- **Score**: {call.score}")
            lines.append(f"- **Success**: {call.success}")
            lines.append(f"- **Time Cost**: {call.time_cost}s")
            lines.append(f"- **Token Cost**: {call.token_cost}")
            lines.append(f"- **Call Input**: ```json\n{call.input}\n```")
            lines.append(f"- **Call Output**: ```txt\n{call.output}\n```")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_statistics_markdown(statistics: dict) -> str:
        """Format statistics as markdown."""
        lines = [f"- **Success Rate**: {statistics.get('success_rate', 0):.2%}",
                 f"- **Average Score**: {statistics.get('avg_score', 0):.3f}",
                 f"- **Average Time Cost**: {statistics.get('avg_time_cost', 0):.3f}s",
                 f"- **Average Token Cost**: {statistics.get('avg_token_cost', 0):.1f}"]

        return "\n".join(lines)

    @staticmethod
    def _calculate_statistics(calls: List) -> dict:
        """
        Calculate statistical information for the given tool calls.
        Returns avg token_cost, success rate, avg time_cost, and avg score.
        """
        if not calls:
            return {
                "avg_token_cost": 0.0,
                "success_rate": 0.0,
                "avg_time_cost": 0.0,
                "avg_score": 0.0
            }
        
        calls_count = len(calls)
        
        # Calculate statistics
        total_token_cost = sum(call.token_cost for call in calls if call.token_cost >= 0)
        valid_token_calls = [call for call in calls if call.token_cost >= 0]
        avg_token_cost = total_token_cost / len(valid_token_calls) if valid_token_calls else 0.0
        
        successful_calls = sum(1 for call in calls if call.success)
        success_rate = successful_calls / calls_count if calls_count > 0 else 0.0
        
        total_time_cost = sum(call.time_cost for call in calls)
        avg_time_cost = total_time_cost / calls_count if calls_count > 0 else 0.0
        
        total_score = sum(call.score for call in calls)
        avg_score = total_score / calls_count if calls_count > 0 else 0.0
        
        return {
            "avg_token_cost": round(avg_token_cost, 2),
            "avg_time_cost": round(avg_time_cost, 3),
            "success_rate": round(success_rate, 4),
            "avg_score": round(avg_score, 3)
        }

    async def _summarize_single_tool(self, tool_memory: ToolMemory, index: int, source_task: str = "") -> ToolMemory:
        await asyncio.sleep(self.summary_sleep_interval * index)
        logger.info(f"summarize_single_tool: {tool_memory.when_to_use}")
        logger.info(f"total_call_count: {len(tool_memory.tool_call_results)}")
        # Filter calls by create_time >= data_from
        if self.data_from:
            recent_calls = [
                call for call in tool_memory.tool_call_results
                if getattr(call, 'create_time', '') >= self.data_from
            ]
        else:
            # If data_from is not provided, use all calls
            recent_calls = tool_memory.tool_call_results

        if not recent_calls:
            logger.warning(f"No tool call results found for tool: {tool_memory.when_to_use}")
            return tool_memory

        for call in recent_calls:
            call.is_summarized = False
        # 如果提供了 source_task，先过滤掉匹配的样本
        if source_task:
            logger.info(source_task)
            filtered_recent_calls = [call for call in recent_calls if getattr(call, 'source_task', '') != source_task]
            if not filtered_recent_calls:
                logger.info(f"All recent calls filtered out by source_task={source_task} for tool {tool_memory.when_to_use}")
                return tool_memory
            recent_calls = filtered_recent_calls

        logger.info(f"Filtered_calls: {len(recent_calls)}")
        # 检查是否满足最小调用数要求
        if len(recent_calls) < 2:
            logger.info(f"Skipping summarization for tool {tool_memory.when_to_use}: "
                       f"only {len(recent_calls)} calls, need at least {2}")
            return tool_memory

        # Log how many unsummarized calls we're processing
        # unsummarized_count = sum(1 for call in recent_calls if not call.is_summarized)
        # logger.info(f"Summarizing tool {tool_memory.when_to_use}: "
                #    f"{unsummarized_count}/{len(recent_calls)} unsummarized calls")

        
        try:
        # 对于超过 min_call_count 条的 recent_calls，根据 scores 分布进行采样
            sampled_calls = self._sample_calls_by_score_distribution(recent_calls)
        except Exception as e:
            logger.error(f"Failed to sample calls by score distribution: {e}", exc_info=True)
            return tool_memory
        logger.info(f"Sampled {len(sampled_calls)} calls from {len(recent_calls)} recent calls "
                   f"for tool {tool_memory.when_to_use} based on score distribution")

        # Get statistics (使用过滤后的 filtered_calls 计算统计信息)
        statistics = self._calculate_statistics(recent_calls)

        tool_memory.score = statistics.get("avg_score", 0.0)
        # Format data as markdown (使用采样后的 calls)
        call_summaries_md = self._format_call_summaries_markdown(sampled_calls)
        statistics_md = self._format_statistics_markdown(statistics)

        tool_config = ""
        if tool_memory.config:
            try:
                with open(tool_memory.config, 'r') as f:
                    tool_configs = json.load(f)
                    tool_config = str(json.dumps(tool_configs.get(tool_memory.when_to_use, {}), ensure_ascii=False))
            except Exception as e:
                logger.error(f"Failed to load tool config from {tool_memory.config}: {e}")

        # Don't include statistics in prompt - only call summaries
        # paper修改
        prompt = self.prompt_format(prompt_name="summarize_tool_usage_prompt",
                                    tool_name=tool_memory.when_to_use,
                                    tool_config=tool_config,
                                    call_summaries=call_summaries_md)
        # logger.info(f"Summarization prompt for tool {index} ({tool_memory.when_to_use}): {prompt}")

        def parse_summary(message: Message) -> ToolMemory:
            content = message.content.strip()
            # Extract content from txt code block
            llm_summary = extract_content(content, "txt")
            
            # Append statistics markdown to LLM result
            tool_memory.content = f"{llm_summary}\n\n## Statistics\n{statistics_md}"

            # 只标记被采样的 calls 为 summarized
            for call in sampled_calls:
                call.is_summarized = True

            # Update modified time
            tool_memory.update_modified_time()

            logger.info(f"Summarized tool {index}: tool_name={tool_memory.when_to_use}, "
                        f"content_length={len(tool_memory.content)}, "
                        f"marked {len(sampled_calls)} calls as summarized")
            return tool_memory

        # Call LLM to generate summary，处理 429 错误
        async def llm_call():
            return await self.llm.achat(messages=[Message(role=Role.USER, content=prompt)], callback_fn=parse_summary)
        
        result = await handle_429_error(llm_call)

        return result

    async def async_execute(self):
        tool_names: Union[str, List[str]] = self.context.get("tool_names", "")
        workspace_id: str = self.context.workspace_id
        source_task: str = self.context.get("source_task", "")
        # 是否仅返回总结结果而不写回向量库（用于并行批量查询，避免并发文件写入）
        no_persist: bool = bool(self.context.get("no_persist", False))

        if not tool_names:
            logger.warning("tool_names is empty, skipping processing")
            self.context.response.answer = "tool_names is required"
            self.context.response.success = False
            return

        # Normalize tool_names to list
        if isinstance(tool_names, str):
            # Split tool names by comma
            tool_name_list = [name.strip() for name in tool_names.split(",") if name.strip()]
        else:
            # Already a list
            tool_name_list = [name.strip() for name in tool_names if name.strip()]

        logger.info(f"workspace_id={workspace_id} processing {len(tool_name_list)} tools: {tool_name_list}, source_task={source_task}")

        # Search for each tool in the vector store
        matched_tool_memories: List[ToolMemory] = []

        for tool_name in tool_name_list:
            nodes: List[VectorNode] = await self.vector_store.async_search(query=tool_name,
                                                                           workspace_id=workspace_id,
                                                                           top_k=1)

            if nodes:
                top_node = nodes[0]
                memory = vector_node_to_memory(top_node)

                # Ensure it's a ToolMemory and when_to_use matches
                if isinstance(memory, ToolMemory) and memory.when_to_use == tool_name:
                    matched_tool_memories.append(memory)
                    logger.info(f"Found tool_memory for tool_name={tool_name}, "
                                f"memory_id={memory.memory_id}, "
                                f"total_calls={len(memory.tool_call_results)}")
                else:
                    logger.warning(f"No exact match found for tool_name={tool_name}")
            else:
                logger.warning(f"No memory found for tool_name={tool_name}")

        if not matched_tool_memories:
            logger.info("No matching tool memories found")
            self.context.response.answer = "No matching tool memories found"
            self.context.response.success = False
            return

        # Check which tools need summarization
        tools_need_summary = []
        tools_skipped = []
        
        for tool_memory in matched_tool_memories:
            recent_calls = tool_memory.tool_call_results
            if not recent_calls:
                tools_skipped.append(tool_memory)
                logger.info(f"Skipping tool {tool_memory.when_to_use}: all recent calls filtered out by source_task={source_task}")
                continue
            
            unsummarized_count = sum(1 for call in recent_calls if not call.is_summarized)
            
            # 检查是否有新的call的scores与已总结的scores都不同
            needs_rerun = False
            
            # 获取所有已总结的calls的scores集合
            summarized_counts = {}
            unsummarized_counts = {}
            for call in recent_calls:
                scores = getattr(call, 'scores', []) or []
                scores_tuple = tuple(scores)
                if call.is_summarized:
                    summarized_counts[scores_tuple] = summarized_counts.get(scores_tuple, 0) + 1
                else:
                    unsummarized_counts[scores_tuple] = unsummarized_counts.get(scores_tuple, 0) + 1

            # logger.info(f"Tool {tool_memory.when_to_use}: "
            #             f"{unsummarized_count} unsummarized calls, "
            #             f"{len(summarized_counts)} unique summarized score distributions")
            
            # 检查是否有未总结的call的scores与所有已总结的scores都不同
            if unsummarized_count > 0:
                # 规则1: 如果没有已总结的calls，只要有未总结的calls就需要总结
                if len(summarized_counts) == 0:
                    needs_rerun = True
                    logger.info(f"Tool {tool_memory.when_to_use} needs rerun: "
                               f"no summarized calls found, {unsummarized_count} unsummarized calls exist")
                else:
                    # 规则2：存在未总结的 call 的 scores 与已总结的所有 scores 都不同（原有逻辑）
                    for call in recent_calls:
                        if not call.is_summarized:
                            scores = getattr(call, 'scores', []) or []
                            scores_tuple = tuple(scores)
                            if scores_tuple not in summarized_counts:
                                needs_rerun = True
                                logger.info(f"Tool {tool_memory.when_to_use} needs rerun: "
                                           f"found unsummarized call with new score distribution {scores_tuple}")
                                break

                    # 规则3：如果某个未总结的分数种类数量 >= 阈值且该种类已总结数量过少，则也需要重新总结
                    # 其中 MIN_CALL_COUNT 在论文中被定义为 Q/2，这里按相同逻辑动态计算：
                    # Q = _compute_Q(recent_calls)
                    Q = self._compute_Q(recent_calls)
                    min_call_count = max(1, Q // 2)  # 论文中的 MIN_CALL_COUNT = Q/2（向下取整，至少为 1）
                    threshold = max(len(recent_calls) / 2, min_call_count)
                    for scores_tuple, ucount in unsummarized_counts.items():
                        s_count = summarized_counts.get(scores_tuple, 0)
                        # 如果某个分数分布在未总结中的数量已经超过阈值，
                        # 但已总结中该分布的样本数仍然小于论文中的 MIN_CALL_COUNT = Q/2，则触发重新总结
                        if ucount >= threshold and s_count < min_call_count:
                            needs_rerun = True
                            logger.info(f"Tool {tool_memory.when_to_use} needs rerun: "
                                        f"unsummarized count {ucount} for score {scores_tuple} >= {threshold} "
                                        f"but summarized count is only {s_count}")
                            break
            needs_rerun = True
            # 如果没有已总结的calls，或者未总结的calls里：1.有新的scores分布，2.分布改变：已经总结的条数只有1，但是相同分布的条数超过半数
            if needs_rerun:
                tools_need_summary.append(tool_memory)
            else:
                tools_skipped.append(tool_memory)
                logger.info(f"Skipping tool {tool_memory.when_to_use}: all recent {len(recent_calls)} calls already summarized, or there are old scores only.")

        # Concurrently summarize tool memories that need it
        if tools_need_summary:
            logger.info(f"Starting concurrent summarization of {len(tools_need_summary)} tool memories")

            # 使用基类的 submit_async_task 提交所有总结任务
            for index, tool_memory in enumerate(tools_need_summary):
                self.submit_async_task(self._summarize_single_tool, tool_memory, index, source_task)

            # 使用基类的 join_async_task 等待所有任务完成
            # 注意: 基类已经过滤掉异常,返回的只包含成功的结果
            valid_summarized_memories = await self.join_async_task(return_exceptions=True)
            logger.info(f"Completed summarization of {len(valid_summarized_memories)} tool memories")
        else:
            valid_summarized_memories = []
            logger.info("All tool memories are up-to-date, no summarization needed")

        # Combine summarized and skipped memories
        all_memories = valid_summarized_memories + tools_skipped

        # Format summary result
        formatted_answer = self._format_summary_result(valid_summarized_memories, tools_skipped)

        # Set response
        self.context.response.answer = formatted_answer
        self.context.response.success = True
        # 始终把总结结果回传给调用方
        self.context.response.metadata["memory_list"] = all_memories
        # 即使是在并行模式（no_persist=True），也设置 deleted_memory_ids
        # UpdateVectorStoreOp 会使用文件锁机制来安全地处理并发写回
        # 这样 is_summarized 字段的修改也能被正确写回文件
        self.context.response.metadata["deleted_memory_ids"] = [m.memory_id for m in all_memories]

        # Log summary for each tool
        for memory in valid_summarized_memories:
            logger.info(f"Tool: {memory.when_to_use}, "
                        f"Content: {memory.content[:100]}...")

